"""
Database connection pool manager for MySQL.
Provides async connection pool with automatic reconnection and health checks.
"""

import asyncio
import aiomysql
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logger import get_database_logger

logger = get_database_logger()

# How many times to retry on a lost-connection error before giving up
_RECONNECT_RETRIES = 3
_RECONNECT_DELAY  = 0.5   # seconds between retries


def _is_lost_connection(exc: Exception) -> bool:
    """Return True if the exception is a MySQL lost-connection error."""
    msg = str(exc).lower()
    return any(code in msg for code in [
        "2006",   # MySQL server has gone away
        "2013",   # Lost connection to MySQL server during query
        "2055",   # Lost connection to MySQL server at '...'
        "lost connection",
        "server has gone away",
        "forcibly closed",
    ])


class DatabaseManager:
    """Manages MySQL connection pool with async support and auto-reconnect."""
    
    def __init__(self):
        self.pool: Optional[aiomysql.Pool] = None
        self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.pool is not None
    
    async def initialize(self):
        """Initialize database connection pool."""
        if self.is_ready:
            logger.warning("Database pool already initialized")
            return
        
        try:
            logger.info(
                "🔌 Initializing database connection pool",
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database=settings.DB_NAME,
                pool_size=settings.DB_POOL_SIZE
            )
            
            self.pool = await aiomysql.create_pool(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                db=settings.DB_NAME,
                minsize=settings.DB_POOL_SIZE,
                maxsize=settings.DB_POOL_SIZE + settings.DB_MAX_OVERFLOW,
                # ── Reconnection / keep-alive settings ──────────────────────
                pool_recycle=1800,          # Recycle connections every 30 min
                #   (was 3600 – halved so stale connections are recycled sooner)
                connect_timeout=10,         # Fail fast instead of hanging
                # ── Character set ────────────────────────────────────────────
                autocommit=True,
                charset='utf8mb4',
                echo=settings.is_development
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()
            
            self._initialized = True
            logger.info("✅ Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error_with_context(
                e,
                {
                    "action": "initialize_pool",
                    "host": settings.DB_HOST,
                    "database": settings.DB_NAME
                }
            )
            raise

    async def _reinitialize(self):
        """Tear down and recreate the pool after a fatal connection error."""
        logger.warning("🔄 Reinitializing database pool after connection failure…")
        try:
            if self.pool:
                self.pool.close()
                await self.pool.wait_closed()
        except Exception:
            pass
        self.pool = None
        self._initialized = False
        await self.initialize()
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            logger.info("🔌 Closing database connection pool")
            self.pool.close()
            await self.pool.wait_closed()
            self._initialized = False
            logger.info("✅ Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiomysql.Connection, None]:
        """
        Get a database connection from the pool.
        Automatically retries on lost-connection errors.
        """
        if not self.is_ready:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        
        conn = None
        try:
            conn = await self.pool.acquire()
            # Ping the connection to detect stale sockets before using it
            await conn.ping(reconnect=True)
            yield conn
        except Exception as e:
            logger.error_with_context(e, {"action": "get_connection"})
            raise
        finally:
            if conn:
                await self.pool.release(conn)
    
    @asynccontextmanager
    async def get_cursor(self, dictionary=True) -> AsyncGenerator[aiomysql.DictCursor, None]:
        """
        Get a database cursor from the pool.
        
        Args:
            dictionary: If True, returns results as dictionaries (default: True)
        """
        async with self.get_connection() as conn:
            cursor_class = aiomysql.DictCursor if dictionary else aiomysql.Cursor
            async with conn.cursor(cursor_class) as cursor:
                yield cursor
    
    async def execute_query(self, query: str, params: Optional[tuple] = None, fetch_one: bool = False):
        """
        Execute a SELECT query and return results.
        Retries automatically on lost-connection errors.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            fetch_one: If True, returns only one row (default: False)
        
        Returns:
            Query results as list of dictionaries or single dictionary
        """
        last_error = None
        for attempt in range(1, _RECONNECT_RETRIES + 1):
            try:
                async with self.get_cursor() as cursor:
                    await cursor.execute(query, params)
                    if fetch_one:
                        return await cursor.fetchone()
                    return await cursor.fetchall()

            except Exception as e:
                last_error = e
                if _is_lost_connection(e) and attempt < _RECONNECT_RETRIES:
                    logger.warning(
                        f"⚠️  Lost DB connection (attempt {attempt}/{_RECONNECT_RETRIES}), "
                        f"reinitializing pool… | error={e}"
                    )
                    try:
                        await self._reinitialize()
                    except Exception as reinit_err:
                        logger.error_with_context(reinit_err, {"action": "reinitialize_pool"})
                    await asyncio.sleep(_RECONNECT_DELAY * attempt)
                    continue  # retry
                else:
                    logger.error_with_context(
                        e,
                        {
                            "action": "execute_query",
                            "query": query[:200],
                            "params": str(params) if params else None,
                            "attempt": attempt,
                        }
                    )
                    raise

        # All retries exhausted
        raise last_error
    
    async def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if database is healthy, False otherwise
        """
        if not self.is_ready:
            return False
        try:
            async with self.get_cursor() as cursor:
                await cursor.execute("SELECT 1")
                result = await cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error_with_context(e, {"action": "health_check"})
            return False
    
    async def get_table_count(self) -> int:
        """Get the total number of tables in the database."""
        if not self.is_ready:
            return 0
        try:
            query = """
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_schema = %s
            """
            result = await self.execute_query(query, (settings.DB_NAME,), fetch_one=True)
            return result['count'] if result else 0
        except Exception as e:
            logger.error_with_context(e, {"action": "get_table_count"})
            return 0
    
    async def get_table_names(self) -> list:
        """Get list of all table names in the database."""
        if not self.is_ready:
            return []
        try:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                ORDER BY table_name
            """
            results = await self.execute_query(query, (settings.DB_NAME,))
            return [row['table_name'] for row in results] if results else []
        except Exception as e:
            logger.error_with_context(e, {"action": "get_table_names"})
            return []


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    return db_manager


async def init_database():
    """Initialize database connection pool (called at app startup)."""
    await db_manager.initialize()


async def close_database():
    """Close database connection pool (called at app shutdown)."""
    await db_manager.close()
