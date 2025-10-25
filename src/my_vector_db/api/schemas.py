"""
API request and response schemas (DTOs).

These schemas define the structure of data sent to and from the API endpoints.
They are separate from domain models to allow for API versioning and flexibility.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from my_vector_db.domain.models import IndexType


# ============================================================================
# Library Schemas
# ============================================================================


class CreateLibraryRequest(BaseModel):
    """Request schema for creating a new library."""

    name: str = Field(..., min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)
    index_type: IndexType = Field(default=IndexType.FLAT)
    index_config: dict[str, Any] = Field(default_factory=dict)


class UpdateLibraryRequest(BaseModel):
    """Request schema for updating an existing library."""

    name: str | None = Field(None, min_length=1, max_length=255)
    metadata: dict[str, Any] | None = None
    index_type: IndexType | None = None
    index_config: dict[str, Any] | None = None


class LibraryResponse(BaseModel):
    """Response schema for library data."""

    id: UUID
    name: str
    document_ids: list[UUID]
    metadata: dict[str, Any]
    index_type: str
    index_config: dict[str, Any]
    created_at: datetime


# ============================================================================
# Document Schemas
# ============================================================================


class CreateDocumentRequest(BaseModel):
    """Request schema for creating a new document."""

    name: str = Field(..., min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateDocumentRequest(BaseModel):
    """Request schema for updating an existing document."""

    name: str | None = Field(None, min_length=1, max_length=255)
    metadata: dict[str, Any] | None = None


class DocumentResponse(BaseModel):
    """Response schema for document data."""

    id: UUID
    name: str
    chunk_ids: list[UUID]
    metadata: dict[str, Any]
    library_id: UUID
    created_at: datetime


# ============================================================================
# Chunk Schemas
# ============================================================================


class CreateChunkRequest(BaseModel):
    """Request schema for creating a new chunk."""

    text: str = Field(..., min_length=1)
    embedding: list[float] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateChunkRequest(BaseModel):
    """Request schema for updating an existing chunk."""

    text: str | None = Field(None, min_length=1)
    embedding: list[float] | None = Field(None, min_length=1)
    metadata: dict[str, Any] | None = None


class ChunkResponse(BaseModel):
    """Response schema for chunk data."""

    id: UUID
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    document_id: UUID
    created_at: datetime


# ============================================================================
# Query Schemas
# ============================================================================


class QueryRequest(BaseModel):
    """
    Request schema for k-nearest neighbor search.

    Attributes:
        embedding: Query vector to search for
        k: Number of nearest neighbors to return
        filters: Optional metadata filters (e.g., {"date_gte": "2024-01-01"})
    """

    embedding: list[float] = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=1000)
    filters: dict[str, Any] | None = None


class QueryResult(BaseModel):
    """A single result from a kNN query."""

    chunk_id: UUID
    document_id: UUID
    text: str
    score: float  # Similarity score (e.g., cosine similarity)
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """Response schema for kNN query results."""

    results: list[QueryResult]
    total: int
    query_time_ms: float
