"""
Pydantic models for SDK request and response types.

These models provide type safety and validation for all API operations.
They mirror the API schemas but are independent of the server implementation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IndexType(str, Enum):
    """
    Supported vector index types.

    Each index type offers different tradeoffs between accuracy, speed, and memory:
    - FLAT: Exact search, O(n) time, baseline for comparison
    - HNSW: Approximate search, graph-based, excellent for high-dimensional data
    """

    FLAT = "flat"
    HNSW = "hnsw"


# ============================================================================
# Library Models
# ============================================================================


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


class Library(BaseModel):
    """Response model for library data."""

    id: UUID
    name: str
    document_ids: List[UUID]
    metadata: Dict[str, Any]
    index_type: str
    index_config: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Document Models
# ============================================================================


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


class Document(BaseModel):
    """Response model for document data."""

    id: UUID
    name: str
    chunk_ids: List[UUID]
    metadata: Dict[str, Any]
    library_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Chunk Models
# ============================================================================


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


class Chunk(BaseModel):
    """Response model for chunk data."""

    id: UUID
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    document_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Search Models
# ============================================================================


class SearchQuery(BaseModel):
    """Request model for vector search."""

    embedding: List[float] = Field(
        ..., min_length=1, description="Query vector embedding"
    )
    k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters"
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
