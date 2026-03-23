"""
Main FastAPI Application Entry Point.
Handles application lifecycle and WebSocket endpoints.
"""

import sys
import os
# Add the project root directory to sys.path so that
# `python app/main.py` works from anywhere without needing
# uvicorn or PYTHONPATH set manually.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_database, close_database
from app.core.logger import get_logger
from app.utils.schema_generator import initialize_schemas
from app.services.agent.orchestrator import initialize_orchestrator
from app.websocket.chat_handler import get_chat_handler
from app.models.chat_models import HealthCheckResponse
from app.fake_detection_api import router as fake_router          # ← ADD
from app.fake_detection_service import (                          # ← ADD
    init_fake_detection,
    close_fake_detection,
)

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("🚀 Starting AI Chatbot Backend...")
    logger.info("=" * 60)

    logger.info("📊 Initializing database connection pool...")
    await init_database()

    logger.info("🔍 Initializing fake detection service...")
    await init_fake_detection()                                   # ← ADD

    logger.info("📁 Loading database schemas and tools...")
    schema_generator = initialize_schemas(
        schemas_dir=settings.SCHEMA_DIR,
        tools_dir=settings.TOOLS_DIR
    )

    logger.info("🤖 Initializing AI agent orchestrator...")
    initialize_orchestrator(schema_generator)

    logger.info("=" * 60)
    logger.info("✅ Application started successfully!")
    logger.info(f"🌐 Server running on {settings.APP_HOST}:{settings.APP_PORT}")
    logger.info(f"📍 Environment: {settings.ENVIRONMENT}")
    logger.info("=" * 60)

    yield

    logger.info("🛑 Shutting down application...")
    await close_database()
    await close_fake_detection()                                  # ← ADD
    logger.info("✅ Application shut down complete")


# Single FastAPI instance — one app, both chatbot + fake detection run together
app = FastAPI(
    title="AI Chatbot Backend",
    description="AI-powered chatbot with text and voice support for agricultural marketplace",
    version="1.0.0",
    lifespan=lifespan
)

# Fake detection router — completely independent from chatbot routing
app.include_router(fake_router, prefix="/api/v1")                # ← ADD

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "AI Chatbot Backend is running",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    import asyncio as _asyncio
    from app.core.database import db_manager
    from app.services.llm.manager import get_llm_manager

    llm_manager = get_llm_manager()

    db_healthy, llm_health = await _asyncio.gather(
        db_manager.health_check(),
        llm_manager.health_check_all(),
    )

    llm_primary  = llm_health.get("groq", False)
    llm_fallback = llm_health.get("openai", False)

    overall_status = (
        "healthy"  if (db_healthy and llm_primary) else
        "degraded" if llm_primary else
        "unhealthy"
    )

    return HealthCheckResponse(
        status=overall_status,
        database=db_healthy,
        llm_primary=llm_primary,
        llm_fallback=llm_fallback,
    )


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    client_id    = str(uuid.uuid4())
    chat_handler = get_chat_handler()
    await chat_handler.handle_connection(websocket, client_id)


@app.get("/api/stats")
async def get_stats():
    from app.core.database import db_manager

    table_count = await db_manager.get_table_count()
    table_names = await db_manager.get_table_names()

    return {
        "database": {
            "name":         settings.DB_NAME,
            "total_tables": table_count,
            "tables":       table_names[:10]
        },
        "llm": {
            "primary_provider":  "groq",
            "primary_model":     settings.GROQ_MODEL,
            "fallback_provider": "openai",
            "fallback_model":    settings.OPENAI_MODEL
        },
        "features": {
            "text_input":  True,
            "voice_input": True,
            "voice_output": True,
            "streaming":   True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower()
    )