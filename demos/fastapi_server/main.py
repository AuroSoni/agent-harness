"""FastAPI server with health check and database integration."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Try importing agent_base from the workspace (installed via uv).
# If not installed, fall back to the local package by adding repo root to path.
try:
    import agent_base  # noqa: F401
except ImportError:
    logger.info("agent_base not found in workspace, falling back to local package.")
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root))

from agent_router import router as agent_router, session_manager
from db import db
from storage import config_adapter, conversation_adapter, run_adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: connect storage adapters
    await config_adapter.connect()
    await conversation_adapter.connect()
    await run_adapter.connect()
    
    yield

    # Shutdown: evict (checkpoint) resident sessions, then close adapters
    await session_manager.shutdown()
    await config_adapter.close()
    await conversation_adapter.close()
    await run_adapter.close()
    await db.close()


app = FastAPI(
    title="Agent Base API",
    description="Demo API for agent-base package",
    version="0.5.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include agent router for streaming endpoints
app.include_router(agent_router)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint with database connectivity status.
    
    Returns:
        Health status including database connectivity check.
    """
    db_health = await db.health_check()
    
    overall_status = "healthy"
    if db_health.get("status") == "down":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "services": {
            "database": db_health,
        },
    }
