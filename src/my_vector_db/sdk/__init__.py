"""
Vector Database Python SDK

A simple, type-safe Python SDK for interacting with the Vector Database API.

Quick Start:
    >>> from my_vector_db.sdk import VectorDBClient
    >>>
    >>> # Create client
    >>> client = VectorDBClient(base_url="http://localhost:8000")
    >>>
    >>> # Create library
    >>> library = client.libraries.create(name="my_lib", index_type="hnsw")
    >>>
    >>> # Create document and chunk
    >>> doc = client.documents.create(library_id=library.id, name="doc1")
    >>> chunk = client.chunks.create(
    ...     document_id=doc.id,
    ...     text="Hello world",
    ...     embedding=[0.1, 0.2, 0.3, ...]
    ... )
    >>>
    >>> # Search
    >>> results = client.search.query(
    ...     library_id=library.id,
    ...     embedding=[0.1, 0.2, 0.3, ...],
    ...     top_k=10
    ... )

Environment Configuration:
    >>> # Set environment variables
    >>> import os
    >>> os.environ["VECTOR_DB_BASE_URL"] = "http://localhost:8000"
    >>>
    >>> # Create client from environment
    >>> client = VectorDBClient.from_env()

Context Manager:
    >>> with VectorDBClient(base_url="...") as client:
    ...     library = client.libraries.create(name="temp")
    ...     # Client auto-closes on exit
"""

__version__ = "0.1.0"

# Main client
from my_vector_db.sdk.client import VectorDBClient

# Exceptions
from my_vector_db.sdk.exceptions import (
    ServerConnectionError,
    NotFoundError,
    ServerError,
    TimeoutError,
    ValidationError,
    VectorDBError,
)

# Domain models (entities and filters)
from my_vector_db.domain.models import (
    BuildIndexResult,
    Chunk,
    Document,
    FilterGroup,
    FilterOperator,
    IndexType,
    Library,
    LogicalOperator,
    MetadataFilter,
    SearchFilters,
    SearchFiltersWithCallable,
)

# SDK models (request/response DTOs)
from my_vector_db.sdk.models import (
    BatchChunkCreate,
    BatchDocumentCreate,
    ChunkCreate,
    ChunkUpdate,
    DocumentCreate,
    DocumentUpdate,
    LibraryCreate,
    LibraryUpdate,
    SearchQuery,
    SearchResponse,
    SearchResult,
)

__all__ = [
    # Version
    "__version__",
    # Client
    "VectorDBClient",
    # Exceptions
    "VectorDBError",
    "ValidationError",
    "NotFoundError",
    "ServerError",
    "ServerConnectionError",
    "TimeoutError",
    # Domain Models (Entities)
    "Library",
    "Document",
    "Chunk",
    "BuildIndexResult",
    # SDK Models (DTOs - for advanced users)
    "LibraryCreate",
    "LibraryUpdate",
    "DocumentCreate",
    "DocumentUpdate",
    "ChunkCreate",
    "ChunkUpdate",
    "BatchChunkCreate",
    "BatchDocumentCreate",
    "SearchQuery",
    "SearchResponse",
    "SearchResult",
    # Filter Models
    "SearchFilters",
    "SearchFiltersWithCallable",
    "FilterGroup",
    "MetadataFilter",
    "FilterOperator",
    "LogicalOperator",
    "IndexType",
]
