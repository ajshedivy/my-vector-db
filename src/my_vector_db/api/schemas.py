"""
API request and response schemas (DTOs).

These schemas define the structure of data sent to and from the API endpoints.
They are separate from domain models to allow for API versioning and flexibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from my_vector_db.domain.models import IndexType, SearchFilters


# ============================================================================
# Library Schemas
# ============================================================================


class CreateLibraryRequest(BaseModel):
    """Request schema for creating a new library."""

    name: str = Field(..., min_length=1, max_length=255)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    index_type: IndexType = Field(default=IndexType.FLAT)
    index_config: Dict[str, Any] = Field(default_factory=dict)


class UpdateLibraryRequest(BaseModel):
    """Request schema for updating an existing library."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = None
    index_type: Optional[IndexType] = None
    index_config: Optional[Dict[str, Any]] = None


class LibraryResponse(BaseModel):
    """Response schema for library data."""

    id: UUID
    name: str
    document_ids: List[UUID]
    metadata: Dict[str, Any]
    index_type: str
    index_config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Document Schemas
# ============================================================================


class CreateDocumentRequest(BaseModel):
    """Request schema for creating a new document."""

    name: str = Field(..., min_length=1, max_length=255)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateDocumentRequest(BaseModel):
    """Request schema for updating an existing document."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Response schema for document data."""

    id: UUID
    name: str
    chunk_ids: List[UUID]
    metadata: Dict[str, Any]
    library_id: UUID
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Chunk Schemas
# ============================================================================


class CreateChunkRequest(BaseModel):
    """Request schema for creating a new chunk."""

    text: str = Field(..., min_length=1)
    embedding: List[float] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateChunkRequest(BaseModel):
    """Request schema for updating an existing chunk."""

    text: Optional[str] = Field(None, min_length=1)
    embedding: Optional[List[float]] = Field(None, min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class ChunkResponse(BaseModel):
    """Response schema for chunk data."""

    id: UUID
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    document_id: UUID
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Query Schemas
# ============================================================================


class QueryRequest(BaseModel):
    """
    Request schema for k-nearest neighbor search.

    Attributes:
        embedding: Query vector to search for
        k: Number of nearest neighbors to return
        filters: Optional SearchFilters for filtering results

    Note:
        Custom filter functions (custom_filter in SearchFilters) are not supported
        via REST API as functions cannot be serialized. Use declarative filters only.

    Examples:
        # Simple metadata filter
        {
            "embedding": [0.1, 0.2, ...],
            "k": 10,
            "filters": {
                "metadata": {
                    "operator": "and",
                    "filters": [
                        {"field": "category", "operator": "eq", "value": "tech"}
                    ]
                }
            }
        }

        # Time-based filter
        {
            "embedding": [0.1, 0.2, ...],
            "k": 5,
            "filters": {
                "created_after": "2024-01-01T00:00:00Z",
                "created_before": "2024-12-31T23:59:59Z"
            }
        }
    """

    embedding: List[float] = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=1000)
    filters: Optional[SearchFilters] = Field(
        default=None,
        description="Search filters (declarative only - custom functions not supported via API)",
    )


class QueryResult(BaseModel):
    """A single result from a kNN query."""

    chunk_id: UUID
    document_id: UUID
    text: str
    score: float  # Similarity score (e.g., cosine similarity)
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response schema for kNN query results."""

    results: List[QueryResult]
    total: int
    query_time_ms: float


# ============================================================================
# Batch Operation Schemas
# ============================================================================


class BatchChunkCreateRequest(BaseModel):
    """Request schema for batch chunk creation."""

    chunks: List[CreateChunkRequest] = Field(
        ..., min_length=1, description="List of chunks to create"
    )


class BatchChunkResponse(BaseModel):
    """Response schema for batch chunk creation."""

    chunks: List[ChunkResponse] = Field(..., description="Created chunks")
    total: int = Field(..., description="Total number of chunks created")


class BatchDocumentCreateRequest(BaseModel):
    """Request schema for batch document creation."""

    documents: List[CreateDocumentRequest] = Field(
        ..., min_length=1, description="List of documents to create"
    )


class BatchDocumentResponse(BaseModel):
    """Response schema for batch document creation."""

    documents: List[DocumentResponse] = Field(..., description="Created documents")
    total: int = Field(..., description="Total number of documents created")


class IndexBuildResponse(BaseModel):
    """Response schema for index build operation."""

    library_id: UUID = Field(..., description="Library ID")
    total_vectors: int = Field(..., description="Number of vectors indexed")
    dimension: int = Field(..., description="Vector dimension")
    index_type: str = Field(..., description="Index type (flat, hnsw)")
    index_config: Dict[str, Any] = Field(
        default_factory=dict, description="Index configuration parameters"
    )
    status: str = Field(default="success", description="Build status")
