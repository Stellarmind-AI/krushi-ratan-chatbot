"""
Query Validator for READ-ONLY Database Access.
Ensures only SELECT queries are executed, preventing data modifications.
"""

import re
from typing import List, Tuple
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
    # schema details that bypass the sanitized tool layer.
    ALLOWED_KEYWORDS = {
        'SELECT'
    }

    @staticmethod
    def _remove_string_literals(query: str) -> str:
        """
        Remove string literals from query before scanning for forbidden keywords.
        Prevents false positives like LIKE '%Update%' being flagged as UPDATE,
        or LIKE '%delete%' being flagged as DELETE.

        Args:
            query: SQL query string

        Returns:
            Query with string literal content replaced by empty placeholders
        """
        # Replace single-quoted strings: 'anything here' -> ''
        query = re.sub(r"'[^']*'", "''", query)
        # Replace double-quoted strings: "anything" -> ""
        query = re.sub(r'"[^"]*"', '""', query)
        return query

    @staticmethod
    def is_read_only(query: str) -> Tuple[bool, str]:
        """
        Check if query is read-only (SELECT/SHOW/DESCRIBE only).
        
        Args:
            query: SQL query string
        
        Returns:
            Tuple of (is_valid, error_message)
            is_valid: True if query is read-only, False otherwise
            error_message: Error message if invalid, empty string if valid
        """
        if not query or not query.strip():
            return False, "Empty query"
        
        # Normalize query: remove comments and extra whitespace
        clean_query = QueryValidator._clean_query(query)
        
        # Extract first keyword
        first_keyword = QueryValidator._get_first_keyword(clean_query)
        
        if not first_keyword:
            return False, "Unable to determine query type"
        
        # Check if first keyword is allowed
        if first_keyword in QueryValidator.ALLOWED_KEYWORDS:
            # Strip string literals FIRST so we don't flag keywords inside them.
            # e.g.  LIKE '%Update%'  must NOT trigger the UPDATE forbidden check.
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
            return False, f"Operation '{first_keyword}' is not allowed. Only SELECT queries are permitted."

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
        ordered_tables = []
        for table in parsed.find_all(exp.Table):
            table_name = table.name.lower()
            if table_name not in ordered_tables:
                ordered_tables.append(table_name)
            alias = (table.alias_or_name or table_name).lower()
            alias_to_table[alias] = table_name

            if not policy.is_sql_visible_table(table_name):
                return False, f"Table '{table_name}' is not allowed by privacy policy"

        if ordered_tables and policy.is_join_only_table(ordered_tables[0]):
            return False, f"Table '{ordered_tables[0]}' may only be joined, not queried directly"

        for column in parsed.find_all(exp.Column):
            column_name = column.name
            if policy.is_blocked_column_name(column_name):
                return False, f"Column '{column_name}' is blocked by privacy policy"

        for projection in parsed.expressions:
            for column in projection.find_all(exp.Column):
                table_name = alias_to_table.get((column.table or "").lower())
                if not policy.is_safe_selected_column(table_name, column.name):
                    source = f"{table_name}.{column.name}" if table_name else column.name
                    return False, f"Column '{source}' cannot be selected by the chatbot"

        return True, ""

    @staticmethod
    def _validate_privacy_fallback(query: str) -> Tuple[bool, str]:
        policy = get_privacy_policy()
        alias_to_table = {}
        ordered_tables = []

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
            ordered_tables.append(table_name)
            if not policy.is_sql_visible_table(table_name):
                return False, f"Table '{table_name}' is not allowed by privacy policy"

        if ordered_tables and policy.is_join_only_table(ordered_tables[0]):
            return False, f"Table '{ordered_tables[0]}' may only be joined, not queried directly"

        for _, col in re.findall(r"`?([a-zA-Z_][\w]*)`?\.`?([a-zA-Z_][\w]*)`?", query):
            if policy.is_blocked_column_name(col):
                return False, f"Column '{col}' is blocked by privacy policy"

        match = re.search(r"\bSELECT\b(.*?)\bFROM\b", query, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return False, "Unable to identify SELECT list"

        for item in QueryValidator._split_select_items(match.group(1)):
            item_no_alias = re.split(r"\s+AS\s+", item, flags=re.IGNORECASE)[0].strip()
            qualified = re.findall(r"`?([a-zA-Z_][\w]*)`?\.`?([a-zA-Z_][\w]*)`?", item_no_alias)
            if qualified:
                for alias, col in qualified:
                    table_name = alias_to_table.get(alias.lower(), alias.lower())
                    if not policy.is_safe_selected_column(table_name, col):
                        return False, f"Column '{table_name}.{col}' cannot be selected by the chatbot"
                continue

            bare = re.sub(r"\b[A-Z_]+\s*\(|\)|`|\"", "", item_no_alias, flags=re.IGNORECASE).strip()
            if bare and re.match(r"^[a-zA-Z_][\w]*$", bare):
                if not policy.is_safe_selected_column(None, bare):
                    return False, f"Column '{bare}' cannot be selected by the chatbot"

        return True, ""
    
    @staticmethod
    def _clean_query(query: str) -> str:
        """
        Clean query by removing comments and normalizing whitespace.
        
        Args:
            query: Raw SQL query
        
        Returns:
            Cleaned query string
        """
        # Remove single-line comments (-- ...)
        query = re.sub(r'--[^\n]*', '', query)
        
        # Remove multi-line comments (/* ... */)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query.strip()
    
    @staticmethod
    def _get_first_keyword(query: str) -> str:
        """
        Extract the first SQL keyword from query.
        
        Args:
            query: SQL query string
        
        Returns:
            First keyword in uppercase, or empty string if not found
        """
        match = re.match(r'^\s*(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""
    
    @staticmethod
    def validate_batch(queries: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate multiple queries.
        
        Args:
            queries: List of SQL query strings
        
        Returns:
            Tuple of (all_valid, error_messages)
            all_valid: True if all queries are valid
            error_messages: List of error messages for invalid queries
        """
        errors = []
        all_valid = True
        
        for i, query in enumerate(queries):
            is_valid, error_msg = QueryValidator.is_read_only(query)
            
            if not is_valid:
                all_valid = False
                errors.append(f"Query {i+1}: {error_msg}")
        
        return all_valid, errors
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """
        Sanitize query by removing dangerous patterns.
        This is a safety measure, but queries should still be validated.
        
        Args:
            query: SQL query string
        
        Returns:
            Sanitized query
        """
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
