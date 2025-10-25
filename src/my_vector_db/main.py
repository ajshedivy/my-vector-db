"""
Vector Database REST API - Main Application

This is the entry point for the FastAPI application.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from my_vector_db.api.routes import router

logger = logging.getLogger("uvicorn")


# ============================================================================
# Application Lifecycle Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.

    Startup: Log application info and provide hooks for initialization
    Shutdown: Log shutdown and provide hooks for cleanup

    TODO:
    - Load persisted data from disk on startup if implementing persistence
    - Warm up indexes for frequently accessed libraries on startup
    - Persist data to disk on shutdown if implementing persistence
    - Close any open connections or resources on shutdown
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Vector Database API Starting")
    logger.info("=" * 60)
    logger.info("Version: 0.1.0")
    logger.info("Health: http://localhost:8000/health")
    logger.info("=" * 60)

    yield  # Application runs between startup and shutdown
    # Shutdown
    logger.info("Vector Database API Shutting Down")


# Create FastAPI application
app = FastAPI(
    title="Vector Database API",
    description="A REST API for indexing and querying documents in a vector database",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware (configure as needed for your use case)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(KeyError)
async def key_error_handler(request, exc):
    """
    Global fallback handler for KeyError exceptions.

    Note: Most KeyErrors are already handled in routes with specific error messages.
    This is a safety net for any unhandled cases.
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Resource not found: {str(exc)}"},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """
    Global fallback handler for ValueError exceptions.

    Note: Most ValueErrors are already handled in routes with specific error messages.
    This is a safety net for any unhandled cases.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"Invalid input: {str(exc)}"},
    )


# ============================================================================
# Health Check Endpoint
# ============================================================================


@app.get("/health", status_code=status.HTTP_200_OK, tags=["health"])
def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        Status indicating the service is running, with storage metrics
    """
    from my_vector_db.api.routes import storage

    # Get basic storage metrics
    num_libraries = len(storage._libraries)
    num_documents = len(storage._documents)
    num_chunks = len(storage._chunks)

    return {
        "status": "healthy",
        "service": "vector-db",
        "version": "0.1.0",
        "storage": {
            "libraries": num_libraries,
            "documents": num_documents,
            "chunks": num_chunks,
        },
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "my_vector_db.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
