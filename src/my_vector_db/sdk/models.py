"""
Pydantic models for SDK request and response types.

These models provide type safety and validation for all API operations.
The SDK imports domain models directly for entities (Chunk, Library, Document)
and defines request/response DTOs for API interactions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Import domain models directly - single source of truth
# These are re-exported via sdk/__init__.py for user convenience
from my_vector_db.domain.models import (
    Chunk,
    Document,
    IndexType,
    Library,
    SearchFilters,
)

# Re-export domain models (used by sdk/__init__.py)
__all__ = [
    # Domain models (re-exported for convenience)
    "Chunk",
    "Document",
    "IndexType",
    "Library",
    "SearchFilters",
    # SDK request/response models
    "LibraryCreate",
    "LibraryUpdate",
    "DocumentCreate",
    "DocumentUpdate",
    "ChunkCreate",
    "ChunkUpdate",
    "SearchQuery",
    "SearchResult",
    "SearchResponse",
    "BatchChunkCreate",
    "BatchDocumentCreate",
]


# ============================================================================
# Library Request/Response Models (DTOs)
# ============================================================================
# Note: Library entity is imported from domain.models


class LibraryCreate(BaseModel):
    """Request model for creating a library."""

    name: str = Field(..., min_length=1, max_length=255, description="Library name")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata"
    )
    index_type: IndexType = IndexType.FLAT
    index_config: Dict[str, Any] = Field(
        default_factory=dict, description="Index-specific configuration"
    )


class LibraryUpdate(BaseModel):
    """Request model for updating a library."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = None
    index_type: Optional[IndexType] = Field(None)
    index_config: Optional[Dict[str, Any]] = None


# ============================================================================
# Document Request/Response Models (DTOs)
# ============================================================================
# Note: Document entity is imported from domain.models


class DocumentCreate(BaseModel):
    """Request model for creating a document."""

    library_id: UUID = Field(..., description="ID of the parent library")
    name: str = Field(..., min_length=1, max_length=255, description="Document name")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata"
    )


class DocumentUpdate(BaseModel):
    """Request model for updating a document."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Chunk Request/Response Models (DTOs)
# ============================================================================
# Note: Chunk entity is imported from domain.models


class ChunkCreate(BaseModel):
    """Request model for creating a chunk."""

    document_id: UUID = Field(..., description="ID of the parent document")
    text: str = Field(..., min_length=1, description="Text content of the chunk")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata"
    )


class ChunkUpdate(BaseModel):
    """Request model for updating a chunk."""

    text: Optional[str] = Field(None, min_length=1)
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Search Models
# ============================================================================


class SearchQuery(BaseModel):
    """
    Request model for vector search.

    Supports both declarative filters and custom Python functions (SDK only).
    """

    embedding: List[float] = Field(
        ..., min_length=1, description="Query vector embedding"
    )
    k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    filters: Optional[SearchFilters] = Field(
        None,
        description=(
            "Search filters. Supports declarative filters (metadata, time-based, document IDs) "
            "and custom Python functions (SDK only, not via REST API)."
        ),
    )


class SearchResult(BaseModel):
    """Single search result."""

    chunk_id: UUID
    document_id: UUID
    text: str
    score: float
    metadata: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    """Response model for search results."""

    results: List[SearchResult]
    total: int
    query_time_ms: float

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Batch Operation Models
# ============================================================================


class BatchChunkCreate(BaseModel):
    """Request model for batch chunk creation."""

    document_id: UUID = Field(..., description="ID of the parent document")
    chunks: List[ChunkCreate] = Field(
        ..., min_length=1, description="List of chunks to create"
    )


class BatchDocumentCreate(BaseModel):
    """Request model for batch document creation."""

    library_id: UUID = Field(..., description="ID of the parent library")
    documents: List[DocumentCreate] = Field(
        ..., min_length=1, description="List of documents to create"
    )
