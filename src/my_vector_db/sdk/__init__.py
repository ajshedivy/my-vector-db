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

# Models (for type hints and construction)
from my_vector_db.sdk.models import (
    Chunk,
    ChunkCreate,
    ChunkUpdate,
    Document,
    DocumentCreate,
    DocumentUpdate,
    Library,
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
    # Configuration
    "SDKConfig",
    # Exceptions
    "VectorDBError",
    "ValidationError",
    "NotFoundError",
    "ServerError",
    "ServerConnectionError",
    "TimeoutError",
    # Models
    "Library",
    "LibraryCreate",
    "LibraryUpdate",
    "Document",
    "DocumentCreate",
    "DocumentUpdate",
    "Chunk",
    "ChunkCreate",
    "ChunkUpdate",
    "SearchQuery",
    "SearchResponse",
    "SearchResult",
]
