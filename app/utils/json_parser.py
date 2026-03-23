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
        Handles various formats.
        
        Args:
            text: Text containing SQL queries
        
        Returns:
            List of query dictionaries with 'table_name' and 'sql' keys
        """
        queries = []
        
        # Try JSON array first
        try:
            parsed = JSONParser.parse(text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and 'sql' in item:
                        queries.append(item)
                return queries
            elif isinstance(parsed, dict):
                if 'sql' in parsed:
                    return [parsed]
                elif 'queries' in parsed:
                    return parsed['queries']
        except ValueError:
            pass
        
        # Try to extract SQL statements with regex
        sql_pattern = r'SELECT.*?(?:;|\n\n|$)'
        matches = re.findall(sql_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            sql = match.strip().rstrip(';')
            
            # Try to extract table name from SQL
            table_match = re.search(r'FROM\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE)
            table_name = table_match.group(1) if table_match else "unknown"
            
            queries.append({
                "table_name": table_name,
                "sql": sql
            })
        
        return queries


# Singleton instance
json_parser = JSONParser()


def parse_json(text: str, expected_type: Optional[type] = None) -> Any:
    """Convenience function for parsing JSON."""
    return json_parser.parse(text, expected_type)


def safe_parse_json(text: str, default: Any = None, expected_type: Optional[type] = None) -> Any:
    """Convenience function for safe JSON parsing."""
    return json_parser.safe_parse(text, default, expected_type)
