"""
Query Validator for READ-ONLY Database Access.
Ensures only SELECT queries are executed, preventing data modifications.

FIX (subquery-aware validation):
  The validator previously treated all tables found in a SQL query identically —
  whether they appeared in the outer FROM/JOIN or inside a subquery
  (e.g. `WHERE category_id IN (SELECT id FROM categories ...)`).

  This caused valid, LLM-generated SQL to be rejected when:
    1. A category table (buy_sell_categories, kshop_categories) appeared in an
       IN-subquery: the validator flagged it as "may only be joined, not queried
       directly" because it scanned ordered_tables[0] after including subquery tables.
    2. In sqlglot path: find_all(exp.Table) traverses the full AST including
       subqueries — the subquery table could land at ordered_tables[0] depending
       on traversal order, triggering the join_only guard incorrectly.
    3. In fallback regex path: the FROM keyword inside a subquery matched the
       same table_pattern, polluting ordered_tables with subquery sources.

  Fix:
    - Add _is_in_subquery() for the sqlglot path: walks the AST parent chain to
      detect whether a Table node lives inside an exp.Subquery node.
    - Add _extract_subquery_tables() for the fallback path: uses a bracket-
      balanced regex scan to find tables referenced only in IN/EXISTS subqueries.
    - Both paths now keep ordered_tables for OUTER query tables only, so the
      is_join_only_table(ordered_tables[0]) guard fires correctly.
    - Subquery tables still go through is_sql_visible_table() — they must be
      readable — but are exempt from the "primary table cannot be join_only" rule.
"""

import re
from typing import List, Set, Tuple
from app.core.logger import get_database_logger
from app.utils.privacy_policy import get_privacy_policy

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover - used only when dependency is missing
    sqlglot = None
    exp = None

logger = get_database_logger()


class QueryValidator:
    """Validates SQL queries to ensure READ-ONLY access."""

    # SQL keywords that modify data (forbidden)
    FORBIDDEN_KEYWORDS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'REPLACE', 'RENAME', 'GRANT', 'REVOKE',
        'LOCK', 'UNLOCK', 'CALL', 'EXECUTE', 'EXEC'
    }

    # Only SELECT is allowed for chatbot SQL. SHOW/DESCRIBE/EXPLAIN reveal
    # schema details that bypass the sanitised tool layer.
    ALLOWED_KEYWORDS = {
        'SELECT'
    }

    @staticmethod
    def _remove_string_literals(query: str) -> str:
        """
        Remove string literals from query before scanning for forbidden keywords.
        Prevents false positives like LIKE '%Update%' being flagged as UPDATE,
        or LIKE '%delete%' being flagged as DELETE.
        """
        query = re.sub(r"'[^']*'", "''", query)
        query = re.sub(r'"[^"]*"', '""', query)
        return query

    @staticmethod
    def is_read_only(query: str) -> Tuple[bool, str]:
        """
        Check if query is read-only (SELECT only).

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Empty query"

        clean_query = QueryValidator._clean_query(query)
        first_keyword = QueryValidator._get_first_keyword(clean_query)

        if not first_keyword:
            return False, "Unable to determine query type"

        if first_keyword in QueryValidator.ALLOWED_KEYWORDS:
            query_no_literals = QueryValidator._remove_string_literals(clean_query)
            upper_query = query_no_literals.upper()

            for forbidden in QueryValidator.FORBIDDEN_KEYWORDS:
                if re.search(rf'\b{forbidden}\b', upper_query):
                    return False, f"Forbidden operation detected: {forbidden}"

            privacy_ok, privacy_error = QueryValidator._validate_privacy(clean_query)
            if not privacy_ok:
                return False, privacy_error

            logger.debug("✅ Query validated (READ-ONLY)", query=clean_query[:100])
            return True, ""

        else:
            logger.warning("❌ Query validation failed", query=clean_query[:100])
            return False, (
                f"Operation '{first_keyword}' is not allowed. "
                "Only SELECT queries are permitted."
            )

    # ── Privacy validation dispatch ───────────────────────────────────────────

    @staticmethod
    def _validate_privacy(query: str) -> Tuple[bool, str]:
        """Validate table and column access against the schema privacy policy."""
        if QueryValidator._has_select_star(query):
            return False, "SELECT * and table.* are not allowed"

        if sqlglot is not None and exp is not None:
            return QueryValidator._validate_privacy_with_sqlglot(query)
        return QueryValidator._validate_privacy_fallback(query)

    @staticmethod
    def _has_select_star(query: str) -> bool:
        match = re.search(r"\bSELECT\b(.*?)\bFROM\b", query, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return False
        for item in QueryValidator._split_select_items(match.group(1)):
            item = item.strip()
            if item == "*" or re.match(r"^[`\"\w]+\.\*$", item):
                return True
        return False

    @staticmethod
    def _split_select_items(select_part: str) -> List[str]:
        items = []
        depth = 0
        quote = ""
        start = 0
        for idx, ch in enumerate(select_part):
            if quote:
                if ch == quote:
                    quote = ""
                continue
            if ch in ("'", '"', "`"):
                quote = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif ch == "," and depth == 0:
                items.append(select_part[start:idx])
                start = idx + 1
        items.append(select_part[start:])
        return items

    # ── sqlglot path ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_in_subquery(node: object, outer_select: object) -> bool:
        """
        Return True if `node` is nested inside a Subquery node relative
        to `outer_select`.

        Walk the AST parent chain upward. If we hit an exp.Subquery before
        reaching the outer SELECT, the node belongs to a subquery (e.g. the
        table inside `WHERE col IN (SELECT id FROM that_table ...)`).

        This lets the validator distinguish outer FROM/JOIN tables (which must
        obey the join_only rule) from subquery source tables (which must be
        sql_visible but are allowed to be join_only — they are not the primary
        target of the user's query).
        """
        if sqlglot is None or exp is None:
            return False
        parent = getattr(node, "parent", None)
        while parent is not None and parent is not outer_select:
            if isinstance(parent, exp.Subquery):
                return True
            parent = getattr(parent, "parent", None)
        return False

    @staticmethod
    def _validate_privacy_with_sqlglot(query: str) -> Tuple[bool, str]:
        policy = get_privacy_policy()
        try:
            parsed = sqlglot.parse_one(query, read="mysql")
        except Exception as e:
            return False, f"Unable to parse SQL safely: {e}"

        if not isinstance(parsed, exp.Select):
            return False, "Only SELECT statements are allowed"

        alias_to_table = {}
        # ordered_outer_tables: only the tables in the *outer* FROM/JOIN clauses.
        # Subquery tables (e.g. inside IN (SELECT ...)) are excluded from this
        # list so the join_only guard checks the correct primary table.
        ordered_outer_tables = []

        for table in parsed.find_all(exp.Table):
            table_name = table.name.lower()
            in_subquery = QueryValidator._is_in_subquery(table, parsed)

            # All tables — outer or subquery — must be sql_visible.
            if not policy.is_sql_visible_table(table_name):
                return False, f"Table '{table_name}' is not allowed by privacy policy"

            if not in_subquery:
                # Only track outer-query tables for alias resolution and
                # the join_only primary-table check.
                if table_name not in ordered_outer_tables:
                    ordered_outer_tables.append(table_name)
                alias = (table.alias_or_name or table_name).lower()
                alias_to_table[alias] = table_name
            # Subquery tables: visible check passed above; no join_only
            # restriction — they act as read-only filter sources, not
            # primary query targets.

        # The FIRST table in the outer FROM must not be join_only.
        # (Subquery tables are intentionally excluded from this check.)
        if ordered_outer_tables and policy.is_join_only_table(ordered_outer_tables[0]):
            return False, (
                f"Table '{ordered_outer_tables[0]}' may only be joined, "
                "not queried directly"
            )

        # Blocked column names — scan entire query including subqueries.
        for column in parsed.find_all(exp.Column):
            column_name = column.name
            if policy.is_blocked_column_name(column_name):
                return False, f"Column '{column_name}' is blocked by privacy policy"

        # Safe-column check — only for the outer SELECT projection list.
        for projection in parsed.expressions:
            for column in projection.find_all(exp.Column):
                table_name = alias_to_table.get((column.table or "").lower())
                if not policy.is_safe_selected_column(table_name, column.name):
                    source = (
                        f"{table_name}.{column.name}" if table_name else column.name
                    )
                    return False, f"Column '{source}' cannot be selected by the chatbot"

        return True, ""

    # ── regex fallback path ───────────────────────────────────────────────────

    @staticmethod
    def _extract_subquery_tables(query: str) -> Set[str]:
        """
        Return the set of table names that appear ONLY inside
        IN (...) / EXISTS (...) subqueries, not in the outer FROM/JOIN.

        Uses a bracket-balanced scan so nested parentheses are handled
        correctly (e.g. IN (SELECT ... FROM t WHERE col IN (SELECT ...))).

        Why this matters:
          The fallback regex `table_pattern` matches every FROM/JOIN keyword
          in the entire query string, including those inside subqueries.
          Without this helper, a subquery like
            `WHERE category_id IN (SELECT id FROM buy_sell_categories ...)`
          would add 'buy_sell_categories' to ordered_tables, potentially
          triggering the join_only guard on the wrong table.
        """
        subquery_tables: Set[str] = set()

        # Walk the query character-by-character to identify subquery spans.
        # A subquery starts with '(' immediately followed (modulo whitespace)
        # by 'SELECT'.
        i = 0
        n = len(query)
        while i < n:
            if query[i] == '(':
                # Look ahead past whitespace for SELECT
                j = i + 1
                while j < n and query[j] in (' ', '\t', '\n', '\r'):
                    j += 1
                if query[j:j + 6].upper() == 'SELECT':
                    # Extract the full subquery body (bracket-balanced)
                    depth = 1
                    k = i + 1
                    while k < n and depth > 0:
                        if query[k] == '(':
                            depth += 1
                        elif query[k] == ')':
                            depth -= 1
                        k += 1
                    subquery_body = query[i + 1:k - 1]  # content between ( and )
                    # Find FROM <table> inside this subquery body
                    for m in re.finditer(
                        r'\bFROM\s+`?([a-zA-Z_]\w*)`?',
                        subquery_body,
                        re.IGNORECASE,
                    ):
                        subquery_tables.add(m.group(1).lower())
                    i = k
                    continue
            i += 1
        return subquery_tables

    @staticmethod
    def _validate_privacy_fallback(query: str) -> Tuple[bool, str]:
        policy = get_privacy_policy()
        alias_to_table = {}
        ordered_tables = []

        # Identify tables that appear ONLY inside subqueries so we can
        # exclude them from the join_only primary-table check while still
        # verifying they are sql_visible.
        subquery_tables = QueryValidator._extract_subquery_tables(query)

        table_pattern = re.compile(
            r"\b(FROM|JOIN)\s+`?([a-zA-Z_][\w]*)`?(?:\s+(?:AS\s+)?`?([a-zA-Z_][\w]*)`?)?",
            re.IGNORECASE,
        )
        for match in table_pattern.finditer(query):
            table_name = match.group(2).lower()
            alias = (match.group(3) or table_name).lower()
            if alias in {"on", "where", "left", "right", "inner", "join", "order", "limit"}:
                alias = table_name
            alias_to_table[alias] = table_name

            # All tables — outer or subquery — must be sql_visible.
            if not policy.is_sql_visible_table(table_name):
                return False, f"Table '{table_name}' is not allowed by privacy policy"

            # Add to ordered_tables ONLY for outer-query tables.
            # Subquery source tables must not pollute the primary-table check.
            if table_name not in subquery_tables:
                ordered_tables.append(table_name)

        # The first outer-query table must not be join_only.
        if ordered_tables and policy.is_join_only_table(ordered_tables[0]):
            return False, (
                f"Table '{ordered_tables[0]}' may only be joined, "
                "not queried directly"
            )

        # Blocked column check (qualified columns: table.col or alias.col)
        for _, col in re.findall(
            r"`?([a-zA-Z_][\w]*)`?\.`?([a-zA-Z_][\w]*)`?", query
        ):
            if policy.is_blocked_column_name(col):
                return False, f"Column '{col}' is blocked by privacy policy"

        # Safe-column check — outer SELECT projection list only.
        match = re.search(
            r"\bSELECT\b(.*?)\bFROM\b", query, flags=re.IGNORECASE | re.DOTALL
        )
        if not match:
            return False, "Unable to identify SELECT list"

        for item in QueryValidator._split_select_items(match.group(1)):
            item_no_alias = re.split(r"\s+AS\s+", item, flags=re.IGNORECASE)[0].strip()
            qualified = re.findall(
                r"`?([a-zA-Z_][\w]*)`?\.`?([a-zA-Z_][\w]*)`?", item_no_alias
            )
            if qualified:
                for alias, col in qualified:
                    table_name = alias_to_table.get(alias.lower(), alias.lower())
                    if not policy.is_safe_selected_column(table_name, col):
                        return False, (
                            f"Column '{table_name}.{col}' cannot be selected by the chatbot"
                        )
                continue

            bare = re.sub(
                r"\b[A-Z_]+\s*\(|\)|`|\"", "", item_no_alias, flags=re.IGNORECASE
            ).strip()
            if bare and re.match(r"^[a-zA-Z_][\w]*$", bare):
                if not policy.is_safe_selected_column(None, bare):
                    return False, f"Column '{bare}' cannot be selected by the chatbot"

        return True, ""

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _clean_query(query: str) -> str:
        """Clean query by removing comments and normalising whitespace."""
        query = re.sub(r'--[^\n]*', '', query)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        query = ' '.join(query.split())
        return query.strip()

    @staticmethod
    def _get_first_keyword(query: str) -> str:
        """Extract the first SQL keyword from query."""
        match = re.match(r'^\s*(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""

    @staticmethod
    def validate_batch(queries: List[str]) -> Tuple[bool, List[str]]:
        """Validate multiple queries."""
        errors = []
        all_valid = True

        for i, query in enumerate(queries):
            is_valid, error_msg = QueryValidator.is_read_only(query)
            if not is_valid:
                all_valid = False
                errors.append(f"Query {i + 1}: {error_msg}")

        return all_valid, errors

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitise query by removing dangerous patterns."""
        query = QueryValidator._clean_query(query)

        if query.endswith(';'):
            query = query[:-1]

        query = query.replace(';', '')

        return query.strip()


# Singleton instance
query_validator = QueryValidator()


def validate_query(query: str) -> Tuple[bool, str]:
    """Convenience function for query validation."""
    return query_validator.is_read_only(query)


def validate_queries(queries: List[str]) -> Tuple[bool, List[str]]:
    """Convenience function for batch validation."""
    return query_validator.validate_batch(queries)