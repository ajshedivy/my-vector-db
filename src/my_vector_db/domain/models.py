"""
Core domain models for the Vector Database.

These Pydantic models represent the core entities:
- Chunk: A piece of text with its embedding and metadata
- Document: A collection of chunks with metadata
- Library: A collection of documents with an associated index
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

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


class Chunk(BaseModel):
    """
    A chunk represents a piece of text with its vector embedding.

    Attributes:
        id: Unique identifier for the chunk
        text: The actual text content
        embedding: Vector representation of the text (list of floats)
        metadata: Additional metadata (e.g., source, page_number, etc.)
        document_id: ID of the parent document
        created_at: Timestamp when chunk was created
    """

    id: UUID = Field(default_factory=uuid4)
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: UUID
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class Document(BaseModel):
    """
    A document is a collection of chunks.

    Attributes:
        id: Unique identifier for the document
        name: Human-readable name for the document
        chunk_ids: List of chunk IDs belonging to this document
        metadata: Additional metadata (e.g., author, category, etc.)
        library_id: ID of the parent library
        created_at: Timestamp when document was created
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    chunk_ids: list[UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    library_id: UUID
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class Library(BaseModel):
    """
    A library is a collection of documents with a vector index.

    Attributes:
        id: Unique identifier for the library
        name: Human-readable name for the library
        document_ids: List of document IDs in this library
        metadata: Additional metadata
        index_type: Type of vector index to use
        index_config: Configuration parameters for the index
        created_at: Timestamp when library was created
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    document_ids: list[UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    index_type: IndexType = IndexType.FLAT
    index_config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)
