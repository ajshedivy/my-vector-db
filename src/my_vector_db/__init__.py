"""
My Vector DB - A lightweight, in-memory vector database with multiple index types.

This package provides a complete vector database solution with:
- Multiple index algorithms (Flat, HNSW)
- RESTful API via FastAPI
- Python SDK for easy integration
- Thread-safe in-memory storage
- Rich domain models with Pydantic validation

Quick Start (Recommended):
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
"""

# Version
__version__ = "0.2.1"

# ============================================================================
# Public API - SDK (Primary Interface)
# ============================================================================
# Import everything through SDK layer for consistency

from my_vector_db.sdk import (
    # Client
    VectorDBClient,
    # Exceptions
    VectorDBError,
    ServerConnectionError,
    NotFoundError,
    ValidationError,
    ServerError,
    TimeoutError,
    # Domain Models (Entities)
    Library,
    Document,
    Chunk,
    BuildIndexResult,
    # Filter Models
    SearchFilters,
    SearchFiltersWithCallable,
    FilterGroup,
    MetadataFilter,
    FilterOperator,
    LogicalOperator,
    IndexType,
    # Response Models (what users receive)
    SearchResponse,
    SearchResult,
)

# ============================================================================
# Internal APIs (importable but not in __all__)
# ============================================================================


# ============================================================================
# Public Exports
# ============================================================================

__all__ = [
    # Version
    "__version__",
    # Client
    "VectorDBClient",
    # Exceptions
    "VectorDBError",
    "ServerConnectionError",
    "NotFoundError",
    "ValidationError",
    "ServerError",
    "TimeoutError",
    # Domain Models (Entities)
    "Library",
    "Document",
    "Chunk",
    "BuildIndexResult",
    # Filter Models
    "SearchFilters",
    "SearchFiltersWithCallable",
    "FilterGroup",
    "MetadataFilter",
    "FilterOperator",
    "LogicalOperator",
    "IndexType",
    # Response Models
    "SearchResponse",
    "SearchResult",
]
