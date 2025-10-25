"""
FastAPI REST API for the Vector Database.

This package provides the HTTP API layer with routes and schemas.

- routes: Contains the FastAPI router with all endpoint definitions
- schemas: Request/response models (DTOs) for API operations
"""

from my_vector_db.api.routes import router
from my_vector_db.api.schemas import (
    ChunkResponse,
    CreateChunkRequest,
    CreateDocumentRequest,
    CreateLibraryRequest,
    DocumentResponse,
    LibraryResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    UpdateChunkRequest,
    UpdateDocumentRequest,
    UpdateLibraryRequest,
)

__all__ = [
    # Router
    "router",
    # Library schemas
    "CreateLibraryRequest",
    "UpdateLibraryRequest",
    "LibraryResponse",
    # Document schemas
    "CreateDocumentRequest",
    "UpdateDocumentRequest",
    "DocumentResponse",
    # Chunk schemas
    "CreateChunkRequest",
    "UpdateChunkRequest",
    "ChunkResponse",
    # Query schemas
    "QueryRequest",
    "QueryResponse",
    "QueryResult",
]
