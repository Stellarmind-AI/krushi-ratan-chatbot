"""
Robust JSON Parser Utility.
Handles malformed JSON responses from LLMs with multiple fallback strategies.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from app.core.logger import get_logger

logger = get_logger("json_parser")


class JSONParser:
    """Robust JSON parser with multiple fallback strategies."""
    
    @staticmethod
    def parse(text: str, expected_type: Optional[type] = None) -> Any:
        """
        Parse JSON with multiple fallback strategies.
        
        Args:
            text: Text potentially containing JSON
            expected_type: Expected type (dict, list, etc.) for validation
        
        Returns:
            Parsed JSON object
        
        Raises:
            ValueError: If all parsing strategies fail
        """
        # Strategy 0: Fix string concatenation (LLM sometimes uses + operators)
        try:
            text = JSONParser._fix_string_concatenation(text)
        except Exception:
            pass
        
        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(text)
            if JSONParser._validate_type(result, expected_type):
                logger.debug("✅ JSON parsed (direct)", strategy="direct")
                return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code blocks
        try:
            result = JSONParser._extract_from_markdown(text)
            if result and JSONParser._validate_type(result, expected_type):
                logger.debug("✅ JSON parsed (markdown)", strategy="markdown")
                return result
        except Exception:
            pass
        
        # Strategy 3: Regex extraction of JSON objects
        try:
            result = JSONParser._regex_extract(text)
            if result and JSONParser._validate_type(result, expected_type):
                logger.debug("✅ JSON parsed (regex)", strategy="regex")
                return result
        except Exception:
            pass
        
        # Strategy 4: Clean and retry
        try:
            cleaned = JSONParser._clean_text(text)
            result = json.loads(cleaned)
            if JSONParser._validate_type(result, expected_type):
                logger.debug("✅ JSON parsed (cleaned)", strategy="cleaned")
                return result
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Extract first valid JSON object/array
        try:
            result = JSONParser._extract_first_valid(text)
            if result and JSONParser._validate_type(result, expected_type):
                logger.debug("✅ JSON parsed (first_valid)", strategy="first_valid")
                return result
        except Exception:
            pass
        
        # All strategies failed
        logger.error("❌ JSON parsing failed", text_preview=text[:200])
        raise ValueError(f"Failed to parse JSON from text: {text[:100]}...")
    
    @staticmethod
    def _extract_from_markdown(text: str) -> Optional[Any]:
        """Extract JSON from markdown code blocks."""
        # Pattern for ```json ... ``` or ``` ... ```
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    @staticmethod
    def _fix_string_concatenation(text: str) -> str:
        """
        Fix LLM responses that use string concatenation operators (+ or newline continuation).
        Pattern: "string1" \n + "string2" or "string1" + \n "string2"
        
        This happens when LLMs format SQL as if it were code (JavaScript/Python style).
        Converts to proper JSON with single concatenated string.
        
        Example input:
            "sql": "SELECT col " + "FROM table"
        
        Example output:
            "sql": "SELECT col FROM table"
        """
        # Pattern 1: "string" + "string" across any whitespace
        # Match: closing quote, optional whitespace, +, optional whitespace, opening quote
        result = text
        
        # Fix pattern: "..." + "..." or "...\n +" or "+ "..."
        # Remove + operators between quoted strings and join them
        result = re.sub(r'"\s*\+\s*"', ' ', result)
        
        # Pattern 2: Handle cases where the + is on next line (with \ continuation)
        # "string" \
        # + "string2"
        result = re.sub(r'\\\s*\+\s*', ' ', result)
        
        # Pattern 3: Clean up any escaped newlines that might still be there
        result = re.sub(r'\\n\s*(?=\+|")', ' ', result)
        
        return result
    
    @staticmethod
    def _regex_extract(text: str) -> Optional[Any]:
        """Use regex to extract JSON objects or arrays."""
        # Try to find JSON object
        obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        obj_matches = re.findall(obj_pattern, text, re.DOTALL)
        
        for match in obj_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON array
        arr_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        arr_matches = re.findall(arr_pattern, text, re.DOTALL)
        
        for match in arr_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text before JSON parsing."""
        # Remove common prefixes/suffixes
        text = text.strip()
        
        # Remove "Here is the JSON:" type prefixes
        prefixes = [
            r'^Here is the JSON:?\s*',
            r'^The JSON is:?\s*',
            r'^JSON:?\s*',
            r'^Result:?\s*',
        ]
        
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        # Remove trailing text after JSON
        # Find the last } or ]
        last_brace = text.rfind('}')
        last_bracket = text.rfind(']')
        last_pos = max(last_brace, last_bracket)
        
        if last_pos > 0:
            text = text[:last_pos + 1]
        
        return text.strip()
    
    @staticmethod
    def _extract_first_valid(text: str) -> Optional[Any]:
        """Extract the first valid JSON object or array from text."""
        # Try to parse progressively larger substrings
        for i in range(len(text)):
            for j in range(i + 1, len(text) + 1):
                substring = text[i:j].strip()
                if substring.startswith(('{', '[')):
                    try:
                        return json.loads(substring)
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    @staticmethod
    def _validate_type(obj: Any, expected_type: Optional[type]) -> bool:
        """Validate that parsed object matches expected type."""
        if expected_type is None:
            return True
        
        return isinstance(obj, expected_type)
    
    @staticmethod
    def safe_parse(text: str, default: Any = None, expected_type: Optional[type] = None) -> Any:
        """
        Safe JSON parse that returns default on failure.
        
        Args:
            text: Text to parse
            default: Default value if parsing fails
            expected_type: Expected type for validation
        
        Returns:
            Parsed JSON or default value
        """
        try:
            return JSONParser.parse(text, expected_type)
        except ValueError:
            logger.warning("JSON parsing failed, returning default", default=str(default))
            return default
    
    @staticmethod
    def extract_tools_from_text(text: str) -> List[str]:
        """
        Extract tool names from LLM response.
        Handles various formats:
        - JSON array: ["tool1", "tool2"]
        - Comma-separated: tool1, tool2, tool3
        - Bullet points: - tool1\n- tool2
        
        Args:
            text: Text containing tool names
        
        Returns:
            List of tool names
        """
        tools = []
        
        # Try JSON array first
        try:
            parsed = JSONParser.parse(text, expected_type=list)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except ValueError:
            pass
        
        # Try comma-separated
        if ',' in text:
            parts = text.split(',')
            tools = [part.strip() for part in parts if part.strip()]
            if tools:
                return tools
        
        # Try bullet points
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Remove bullet markers
            line = re.sub(r'^[-*•]\s*', '', line)
            # Remove numbers
            line = re.sub(r'^\d+\.\s*', '', line)
            
            if line and not line.endswith(':'):
                tools.append(line)
        
        if tools:
            return tools
        
        # Last resort: split by whitespace and filter
        words = text.split()
        tools = [word.strip('[](){}",') for word in words if len(word) > 3]
        
        return tools
    
    @staticmethod
    def extract_queries_from_text(text: str) -> List[Dict[str, str]]:
        """
        Extract SQL queries from LLM response.
        Handles various formats and sanitizes the SQL.
        
        Args:
            text: Text containing SQL queries
        
        Returns:
            List of query dictionaries with 'table_name' and 'sql' keys (sanitized)
        """
        queries = []
        
        # Try JSON array first
        try:
            parsed = JSONParser.parse(text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and 'sql' in item:
                        item = SQLSanitizer.sanitize_query_dict(item)
                        queries.append(item)
                return queries
            elif isinstance(parsed, dict):
                if 'sql' in parsed:
                    parsed = SQLSanitizer.sanitize_query_dict(parsed)
                    return [parsed]
                elif 'queries' in parsed:
                    return SQLSanitizer.sanitize_batch(parsed['queries'])
        except ValueError:
            pass
        
        # Try to extract SQL statements with regex
        sql_pattern = r'SELECT.*?(?:;|\n\n|$)'
        matches = re.findall(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            sql = match.strip().rstrip(';')
            
            # Sanitize: remove noise and fix ambiguities
            sql = SQLSanitizer.sanitize_sql(sql)
            
            # Try to extract table name from SQL
            table_match = re.search(r'FROM\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE)
            table_name = table_match.group(1) if table_match else "unknown"
            
            queries.append({
                "table_name": table_name,
                "sql": sql
            })
        
        return queries


class SQLSanitizer:
    """
    Sanitizes SQL queries generated by LLMs.
    
    Handles two categories of LLM-generated noise:
    1. Formatting artifacts (backslashes, extra whitespace, escaped newlines)
    2. Ambiguity issues (missing table prefixes in SELECT clauses when JOINs present)
    """
    
    @staticmethod
    def sanitize_sql(sql: str) -> str:
        """Clean SQL query by removing formatting noise and fixing ambiguities."""
        if not sql:
            return sql
        
        sql = sql.strip()
        
        # Remove any remaining string concatenation operators (+ signs between quoted strings)
        # This handles edge cases where JSON preprocessing didn't catch them
        sql = re.sub(r'"\s*\+\s*"', ' ', sql)
        sql = re.sub(r"'\s*\+\s*'", ' ', sql)
        
        # Remove line continuation backslashes
        sql = re.sub(r'\\\s*\n', ' ', sql)
        
        # Handle escaped newlines
        sql = sql.replace('\\n', ' ')
        
        # Remove carriage returns
        sql = sql.replace('\r', '')
        
        # Normalize whitespace
        sql = sql.replace('\t', ' ')
        
        # Reduce multiple spaces (preserve quoted strings)
        parts = re.split(r"('[^']*'|\"[^\"]*\")", sql)
        normalized_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Not a quoted string
                part = re.sub(r' +', ' ', part)
            normalized_parts.append(part)
        sql = ''.join(normalized_parts)
        
        # Clean up edges
        sql = sql.strip().rstrip(';').strip()

        # ── DEFENSIVE: Strip JSON-artifact suffix ──────────────────────────────
        # When the LLM puts literal newlines inside a JSON string, json.loads
        # fails and the regex fallback grabs SQL + JSON closing characters
        # (e.g. ...%') ORDER BY id DESC" } ' ).  Detect and strip that suffix.
        #
        # Pattern: closing-quote (single or double) + optional whitespace + } or ]
        # We build the character class via concatenation to avoid raw-string
        # delimiter conflicts — no backslash tricks needed.
        import re as _re
        _json_artifact_re = _re.compile(r"""["']\s*[}\]]""")  # noqa: W605
        json_suffix = _json_artifact_re.search(sql)
        if json_suffix:
            sql = sql[:json_suffix.start()].strip()
        # Strip any stray trailing JSON-artifact junk while keeping SQL valid.
        # Plain `rstrip` was greedy and stripped the closing quote of literals
        # like `LIKE '%ભાવનગર%')` (because the `%` blocks stripping but the
        # next char inward is the legitimate closing quote — rstrip removed
        # `)` then `'`, leaving an unclosed string literal).
        # Balance-aware loop: never strip a quote that would unbalance the
        # string literals; never strip a `)` that would unbalance parentheses.
        _junk_chars = set('"\'\n\r\t }])')
        while sql and sql[-1] in _junk_chars:
            ch = sql[-1]
            candidate = sql[:-1]
            if ch == "'" and candidate.count("'") % 2 != 0:
                break  # stripping would leave an unclosed single-quoted literal
            if ch == '"' and candidate.count('"') % 2 != 0:
                break  # stripping would leave an unclosed double-quoted literal
            if ch == ")" and candidate.count("(") > candidate.count(")"):
                break  # stripping would leave an unmatched open paren
            sql = candidate
        sql = sql.strip()
        
        # Fix missing table prefixes in SELECT when JOINs present
        sql = SQLSanitizer._fix_missing_table_prefixes(sql)
        
        return sql
    
    @staticmethod
    def _fix_missing_table_prefixes(sql: str) -> str:
        """Detect JOINs and ensure SELECT columns are table-qualified."""
        # Early exit: if no JOIN, columns don't need prefixing
        if 'JOIN' not in sql.upper():
            return sql
        
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql
        
        select_clause = select_match.group(1)
        
        # Extract primary table name from FROM clause
        from_match = re.search(r'FROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql, re.IGNORECASE)
        if not from_match:
            return sql
        
        primary_table = from_match.group(1)
        primary_alias = from_match.group(2) or primary_table
        
        # Parse SELECT columns
        columns = [col.strip() for col in select_clause.split(',')]
        
        fixed_columns = []
        for col in columns:
            # Already has prefix, is *, or is a function
            if '.' in col or col == '*' or '(' in col:
                fixed_columns.append(col)
            else:
                # Add primary table alias
                if ' AS ' in col.upper():
                    col_name, as_part = re.split(r'\s+AS\s+', col, flags=re.IGNORECASE)
                    fixed_columns.append(f"{primary_alias}.{col_name.strip()} AS {as_part}")
                else:
                    fixed_columns.append(f"{primary_alias}.{col}")
        
        # Reconstruct SELECT clause
        new_select_clause = ', '.join(fixed_columns)
        new_sql = re.sub(
            r'SELECT\s+.*?\s+FROM',
            f'SELECT {new_select_clause} FROM',
            sql,
            count=1,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        return new_sql
    
    @staticmethod
    def sanitize_query_dict(query_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize SQL in a query dictionary."""
        if not isinstance(query_dict, dict):
            return query_dict
        if 'sql' in query_dict and isinstance(query_dict['sql'], str):
            query_dict['sql'] = SQLSanitizer.sanitize_sql(query_dict['sql'])
        return query_dict
    
    @staticmethod
    def sanitize_batch(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize a batch of query dictionaries."""
        return [SQLSanitizer.sanitize_query_dict(q) for q in (queries or [])]


# Singleton instance
json_parser = JSONParser()
sql_sanitizer = SQLSanitizer()


def parse_json(text: str, expected_type: Optional[type] = None) -> Any:
    """Convenience function for parsing JSON."""
    return json_parser.parse(text, expected_type)


def safe_parse_json(text: str, default: Any = None, expected_type: Optional[type] = None) -> Any:
    """Convenience function for safe JSON parsing."""
    return json_parser.safe_parse(text, default, expected_type)