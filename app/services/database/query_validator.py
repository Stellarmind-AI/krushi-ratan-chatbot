"""
Query Validator for READ-ONLY Database Access.
Ensures only SELECT queries are executed, preventing data modifications.
"""

import re
from typing import List, Tuple
from app.core.logger import get_database_logger

logger = get_database_logger()


class QueryValidator:
    """Validates SQL queries to ensure READ-ONLY access."""
    
    # SQL keywords that modify data (forbidden)
    FORBIDDEN_KEYWORDS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'REPLACE', 'RENAME', 'GRANT', 'REVOKE',
        'LOCK', 'UNLOCK', 'CALL', 'EXECUTE', 'EXEC'
    }
    
    # Allowed keywords (READ operations)
    ALLOWED_KEYWORDS = {
        'SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN'
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
            
            logger.debug("✅ Query validated (READ-ONLY)", query=clean_query[:100])
            return True, ""
        
        else:
            logger.warning("❌ Query validation failed", query=clean_query[:100])
            return False, f"Operation '{first_keyword}' is not allowed. Only SELECT/SHOW/DESCRIBE queries are permitted."
    
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
