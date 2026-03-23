"""
Query Executor for Safe Parallel Query Execution.
Executes validated SQL queries with proper error handling and timing.
"""

import time
import asyncio
from typing import List, Dict, Any
from app.core.database import db_manager
from app.services.database.query_validator import query_validator
from app.models.chat_models import QueryResult
from app.core.logger import get_database_logger

logger = get_database_logger()


class QueryExecutor:
    """Executes validated SQL queries safely."""
    
    def __init__(self):
        """Initialize query executor."""
        self.max_results_per_query = 1000  # Safety limit
        self.query_timeout = 30  # seconds
    
    async def execute_single(
        self,
        sql: str,
        table_name: str,
        params: tuple = None
    ) -> QueryResult:
        """
        Execute a single SQL query.
        
        Args:
            sql: SQL query string
            table_name: Table being queried
            params: Query parameters (for prepared statements)
        
        Returns:
            QueryResult object
        """
        start_time = time.time()
        
        try:
            # Validate query is READ-ONLY
            is_valid, error_msg = query_validator.is_read_only(sql)
            
            if not is_valid:
                logger.error(
                    "❌ Query validation failed",
                    table=table_name,
                    error=error_msg
                )
                raise ValueError(f"Query validation failed: {error_msg}")
            
            # Sanitize query
            clean_sql = query_validator.sanitize_query(sql)
            
            # Add LIMIT if not present (safety measure)
            if 'LIMIT' not in clean_sql.upper():
                clean_sql = f"{clean_sql} LIMIT {self.max_results_per_query}"
            
            logger.sql_generation(query=clean_sql, table=table_name)
            
            # LOG ACTUAL SQL QUERY BEING EXECUTED
            logger.info(f"🔍 EXECUTING SQL: {clean_sql}", table=table_name)
            
            # Execute query
            results = await db_manager.execute_query(
                query=clean_sql,
                params=params,
                fetch_one=False
            )
            
            execution_time = time.time() - start_time
            
            # Convert results to list of dicts (if not already)
            rows = list(results) if results else []

            # 🔥 NEW — Log actual DB result payload (LLM input data)
            try:
                preview_rows = rows[:5]  # limit preview for safety
                
                logger.info(
                    "🧠 DB RESULT PAYLOAD",
                    extra={
                        "table": table_name,
                        "row_count": len(rows),
                        "execution_time": round(execution_time, 3),
                        "preview": preview_rows,
                        "columns": list(rows[0].keys()) if rows else [],
                    }
                )
            except Exception:
                pass


            logger.query_execution(
                query=clean_sql,
                rows=len(rows),
                execution_time=execution_time
            )

            
            return QueryResult(
                table_name=table_name,
                sql=clean_sql,
                rows=rows,
                row_count=len(rows),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error_with_context(e, {
                "action": "execute_single",
                "table": table_name,
                "query": sql[:200]
            })
            
            # Return empty result with error info
            return QueryResult(
                table_name=table_name,
                sql=sql,
                rows=[],
                row_count=0,
                execution_time=execution_time
            )
    
    async def execute_parallel(
        self,
        queries: List[Dict[str, str]]
    ) -> List[QueryResult]:
        """
        Execute multiple queries in parallel.
        
        Args:
            queries: List of query dicts with 'table_name' and 'sql' keys
        
        Returns:
            List of QueryResult objects
        """
        if not queries:
            return []
        
        logger.info(
            "🔄 Executing parallel queries",
            count=len(queries)
        )
        
        # Validate all queries first
        sql_queries = [q['sql'] for q in queries]
        all_valid, errors = query_validator.validate_batch(sql_queries)
        
        if not all_valid:
            logger.error("❌ Batch validation failed", errors=errors)
            raise ValueError(f"Query validation failed: {'; '.join(errors)}")
        
        # Create tasks for parallel execution
        tasks = [
            self.execute_single(
                sql=query['sql'],
                table_name=query.get('table_name', 'unknown')
            )
            for query in queries
        ]
        
        # Execute all queries concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error_with_context(result, {
                        "action": "execute_parallel",
                        "query_index": i,
                        "query": queries[i].get('sql', '')[:200]
                    })
                else:
                    valid_results.append(result)
            
            logger.info(
                "✅ Parallel execution complete",
                successful=len(valid_results),
                failed=len(results) - len(valid_results)
            )
            
            return valid_results
            
        except Exception as e:
            logger.error_with_context(e, {
                "action": "execute_parallel",
                "query_count": len(queries)
            })
            raise
    
    def combine_results(self, results: List[QueryResult]) -> Dict[str, Any]:
        """
        Combine multiple query results into a single structure.
        
        Args:
            results: List of QueryResult objects
        
        Returns:
            Combined results dictionary
        """
        combined = {
            "total_queries": len(results),
            "total_rows": sum(r.row_count for r in results),
            "total_execution_time": sum(r.execution_time for r in results),
            "tables_queried": [r.table_name for r in results],
            "results_by_table": {}
        }
        
        for result in results:
            combined["results_by_table"][result.table_name] = {
                "rows": result.rows,
                "row_count": result.row_count,
                "execution_time": result.execution_time
            }
        
        return combined
    
    async def execute_with_retry(
        self,
        sql: str,
        table_name: str,
        max_retries: int = 3
    ) -> QueryResult:
        """
        Execute query with retry logic.
        
        Args:
            sql: SQL query
            table_name: Table name
            max_retries: Maximum retry attempts
        
        Returns:
            QueryResult
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await self.execute_single(sql, table_name)
                
                # If we got results, return immediately
                if result.row_count > 0 or attempt == max_retries - 1:
                    return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Query attempt {attempt + 1} failed",
                    table=table_name,
                    error=str(e)
                )
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        if last_error:
            raise last_error
        
        return QueryResult(
            table_name=table_name,
            sql=sql,
            rows=[],
            row_count=0,
            execution_time=0.0
        )


# Global query executor instance
query_executor = QueryExecutor()


def get_query_executor() -> QueryExecutor:
    """Get query executor instance."""
    return query_executor
