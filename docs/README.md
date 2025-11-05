# My Vector Database Python SDK

The Vector Database Python SDK provides a type-safe, easy-to-use interface for interacting with the Vector Database API.

## Table of Contents

- [My Vector Database Python SDK](#my-vector-database-python-sdk)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Client SDK Reference](#client-sdk-reference)
    - [Initialization](#initialization)
      - [VectorDBClient](#vectordbclient)
    - [Library Operations](#library-operations)
      - [create\_library](#create_library)
      - [get\_library](#get_library)
      - [list\_libraries](#list_libraries)
      - [update\_library](#update_library)
      - [delete\_library](#delete_library)
      - [build\_index](#build_index)
    - [Document Operations](#document-operations)
      - [create\_document](#create_document)
      - [get\_document](#get_document)
      - [list\_documents](#list_documents)
      - [update\_document](#update_document)
      - [delete\_document](#delete_document)
    - [Chunk Operations](#chunk-operations)
      - [add\_chunk](#add_chunk)
      - [add\_chunks](#add_chunks)
      - [create\_chunk](#create_chunk)
      - [get\_chunk](#get_chunk)
      - [list\_chunks](#list_chunks)
      - [list\_all\_chunks](#list_all_chunks)
      - [update\_chunk](#update_chunk)
      - [delete\_chunk](#delete_chunk)
    - [Search Operations](#search-operations)
      - [search](#search)
  - [Persistence Management](#persistence-management)
    - [Overview](#overview)
    - [Container Setup](#container-setup)
    - [Client Operations](#client-operations)
      - [save\_snapshot](#save_snapshot)
      - [restore\_snapshot](#restore_snapshot)
      - [get\_persistence\_status](#get_persistence_status)
    - [Persistence Workflow](#persistence-workflow)
  - [Filtering Guide](#filtering-guide)
    - [Declarative Filters](#declarative-filters)
      - [Metadata Filters](#metadata-filters)
      - [Time-Based Filters](#time-based-filters)
      - [Document ID Filters](#document-id-filters)
    - [Custom Filter Functions](#custom-filter-functions)
    - [Filter Composition](#filter-composition)
  - [Vector Indexes](#vector-indexes)
    - [Index Types](#index-types)
      - [FLAT Index (Exact Search) ✅ Implemented](#flat-index-exact-search--implemented)
      - [HNSW Index (Approximate Search) ⚠️ Not Yet Implemented](#hnsw-index-approximate-search-️-not-yet-implemented)
  - [Best Practices](#best-practices)
    - [Index Selection](#index-selection)
    - [Metadata Design](#metadata-design)
    - [Filter Strategy](#filter-strategy)
    - [Connection Management](#connection-management)
    - [Performance Considerations](#performance-considerations)
    - [Limitations](#limitations)
  - [MCP Server](#mcp-server)
    - [Overview](#overview-1)
    - [Installation](#installation-1)
    - [Configuration](#configuration)
      - [Prerequisites](#prerequisites)
      - [MCP Client Configuration](#mcp-client-configuration)
      - [Verifying Configuration](#verifying-configuration)
    - [Available Tools](#available-tools)
      - [TOOL: search](#tool-search)
      - [TOOL: list\_libraries](#tool-list_libraries)
      - [TOOL: list\_documents](#tool-list_documents)
      - [TOOL: list\_chunks](#tool-list_chunks)
      - [TOOL: get\_library](#tool-get_library)
      - [TOOL: get\_document](#tool-get_document)
      - [TOOL: get\_chunk](#tool-get_chunk)
    - [Embedding Generation](#embedding-generation)
  - [Agno Integration](#agno-integration)
    - [Overview](#overview-2)
    - [Quick Start](#quick-start-1)
    - [Configuration](#configuration-1)
      - [MyVectorDB Parameters](#myvectordb-parameters)
    - [Document Management](#document-management)
    - [Library Management](#library-management)
    - [Search](#search-1)
    - [Example: Full Workflow](#example-full-workflow)
    - [Supported Operations](#supported-operations)
    - [Best Practices](#best-practices-1)
  - [Type Reference](#type-reference)
    - [Library](#library)
    - [Document](#document)
    - [Chunk](#chunk)
    - [SearchResponse](#searchresponse)
    - [SearchResult](#searchresult)
    - [BuildIndexResult](#buildindexresult)
    - [SearchFilters](#searchfilters)
    - [SearchFiltersWithCallable](#searchfilterswithcallable)
    - [FilterGroup](#filtergroup)
    - [MetadataFilter](#metadatafilter)
    - [IndexType](#indextype)
    - [FilterOperator](#filteroperator)
    - [LogicalOperator](#logicaloperator)
  - [Error Handling](#error-handling)


## Installation

```bash
pip install my-vector-db
```

## Quick Start

```python
from my_vector_db import VectorDBClient

# Initialize client
client = VectorDBClient(base_url="http://localhost:8000")

# Create a library
library = client.create_library(name="documents", index_type="hnsw")

# Create a document
doc = client.create_document(library_id=library.id, name="sample")

# Add a chunk with embedding
chunk = client.add_chunk(
    document_id=doc.id,
    text="Machine learning enables computers to learn from data",
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    metadata={"category": "ai", "topic": "ml"}
)

# Perform similarity search
results = client.search(
    library_id=library.id,
    embedding=[0.15, 0.25, 0.35, 0.45, 0.55],
    k=10
)

# Clean up
client.close()
```

Using context manager (recommended):

```python
with VectorDBClient(base_url="http://localhost:8000") as client:
    library = client.create_library(name="documents")
    # Client automatically closes on exit
```

## Client SDK Reference

### Initialization

#### VectorDBClient

```python
VectorDBClient(
    base_url: str = "http://localhost:8000",
    timeout: float = 30.0,
    api_key: Optional[str] = None
)
```

Initialize the Vector Database client.

**Parameters:**
- `base_url` (str): Base URL of the Vector Database API. Default: `"http://localhost:8000"`
- `timeout` (float): Request timeout in seconds. Default: `30.0`
- `api_key` (Optional[str]): Optional API key for authentication. Default: `None`

**Example:**

```python
# Basic initialization
client = VectorDBClient()

# Custom configuration
client = VectorDBClient(
    base_url="http://api.example.com",
    timeout=60.0,
    api_key="your-api-key"
)
```

### Library Operations

A library is a collection of documents with an associated vector index.

#### create_library

```python
create_library(
    name: str,
    index_type: Union[IndexType, str] = IndexType.FLAT,
    metadata: Optional[Dict[str, Any]] = None,
    index_config: Optional[Dict[str, Any]] = None
) -> Library
```

Create a new library.

**Parameters:**
- `name` (str): Library name (1-255 characters)
- `metadata` (Optional[Dict[str, Any]]): Optional metadata dictionary
- `index_type` (Union[IndexType, str]): Index type. Options: `"flat"`, `"hnsw"`. Default: `"flat"`
- `index_config` (Optional[Dict[str, Any]]): Index-specific configuration

**Returns:**
- `Library`: The created library

**Raises:**
- `ValidationError`: If validation fails
- `VectorDBError`: For other errors

**Example:**

```python
# Flat index (exact search)
library = client.create_library(
    name="my_library",
    index_type="flat",
    metadata={"description": "Document embeddings"}
)

# HNSW index (approximate search)
library = client.create_library(
    name="fast_search",
    index_type="hnsw",
    index_config={
        "m": 16,
        "ef_construction": 200
    }
)
```

#### get_library

```python
get_library(library_id: Union[UUID, str]) -> Library
```

Get a library by ID.

**Parameters:**
- `library_id` (Union[UUID, str]): Library identifier

**Returns:**
- `Library`: The library

**Raises:**
- `NotFoundError`: If library not found

#### list_libraries

```python
list_libraries() -> List[Library]
```

List all libraries.

**Returns:**
- `List[Library]`: List of all libraries

#### update_library

```python
update_library(
    library: Union[Library, UUID, str],
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    index_type: Optional[IndexType] = None,
    index_config: Optional[Dict[str, Any]] = None,
) -> Library
```

Update an existing library.

Pass either a Library object (fetch-modify-update) or a library ID with fields to update.
When passing a Library object, fields can be overridden via keyword arguments.

**Parameters:**
- `library` (Union[Library, UUID, str]): Library object or identifier
- `name` (Optional[str]): New name
- `metadata` (Optional[Dict[str, Any]]): New metadata
- `index_type` (Optional[IndexType]): New index type
- `index_config` (Optional[Dict[str, Any]]): New index configuration

**Returns:**
- `Library`: The updated library

**Raises:**
- `NotFoundError`: If library not found

#### delete_library

```python
delete_library(library_id: Union[UUID, str]) -> None
```

Delete a library and all its documents and chunks.

**Parameters:**
- `library_id` (Union[UUID, str]): Library identifier

**Raises:**
- `NotFoundError`: If library not found

#### build_index

```python
build_index(library_id: Union[UUID, str]) -> BuildIndexResult
```

Build or rebuild the vector index for a library.

**Parameters:**
- `library_id` (Union[UUID, str]): Library identifier

**Returns:**
- `Dict[str, Any]`: Index build status

**Raises:**
- `NotFoundError`: If library not found

### Document Operations

A document is a logical grouping of chunks within a library.

#### create_document

```python
create_document(
    library_id: Union[UUID, str],
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Document
```

Create a new document in a library.

**Parameters:**
- `library_id` (Union[UUID, str]): Parent library identifier
- `name` (str): Document name (1-255 characters)
- `metadata` (Optional[Dict[str, Any]]): Optional metadata

**Returns:**
- `Document`: The created document

**Raises:**
- `NotFoundError`: If library not found
- `ValidationError`: If validation fails

**Example:**

```python
doc = client.create_document(
    library_id=library.id,
    name="research_paper.pdf",
    metadata={"author": "John Doe", "year": 2024}
)
```

#### get_document

```python
get_document(document_id: Union[UUID, str]) -> Document
```

Get a document by ID.

**Parameters:**
- `document_id` (Union[UUID, str]): Document identifier

**Returns:**
- `Document`: The document

**Raises:**
- `NotFoundError`: If document not found

#### list_documents

```python
list_documents(library_id: Union[UUID, str]) -> List[Document]
```

List all documents in a library.

**Parameters:**
- `library_id` (Union[UUID, str]): Library identifier

**Returns:**
- `List[Document]`: List of documents

**Raises:**
- `NotFoundError`: If library not found

#### update_document

```python
update_document(
    document: Union[Document, UUID, str],
    *,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Document
```

Update a document. Pass either a Document object (fetch-modify-update) or a document ID with fields to update. When passing a Document object, fields can be overridden via keyword arguments.

**Parameters:**
- `document` (Union[Document, UUID, str]): Document object or identifier
- `name` (Optional[str]): New name
- `metadata` (Optional[Dict[str, Any]]): New metadata

**Returns:**
- `Document`: The updated document

**Raises:**
- `NotFoundError`: If document not found

#### delete_document

```python
delete_document(document_id: Union[UUID, str]) -> None
```

Delete a document and all its chunks.

**Parameters:**
- `document_id` (Union[UUID, str]): Document identifier

**Raises:**
- `NotFoundError`: If document not found

### Chunk Operations

A chunk represents a piece of text with its vector embedding and metadata.

**Developer Experience Note**: The SDK provides `add_chunk()` and `add_chunks()` methods as the primary interface for working with chunks. These methods offer a more intuitive API that accepts both Chunk objects (for working with pre-computed embeddings) and primitives (text, embedding, metadata), making it easier to integrate the vector database into your workflows.

#### add_chunk

```python
def add_chunk(
    self,
    *,
    chunk: Optional[Chunk] = None,
    document_id: Optional[Union[UUID, str]] = None,
    text: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Chunk:
```

Add a chunk to a document. Supports both object-oriented and primitive parameter styles for maximum flexibility.

**Parameters:**
- `chunk` (Optional[Chunk]): Pre-constructed Chunk object with text, embedding, and metadata
- `document_id` (Union[UUID, str]): Parent document identifier
- `text` (Optional[str]): Text content (required if chunk not provided)
- `embedding` (Optional[List[float]]): Vector embedding (required if chunk not provided)
- `metadata` (Optional[Dict[str, Any]]): Optional metadata

**Note**: You must provide either a `chunk` object OR both `text`,  `embedding` and `document_id` parameters.

**Returns:**
- `Chunk`: The created chunk

**Raises:**
- `NotFoundError`: If document not found
- `ValidationError`: If validation fails
- `ValueError`: If neither chunk nor text+embedding provided

**Example - Using Chunk Object:**

```python
from my_vector_db import Chunk

# Create chunk object with pre-computed embedding
chunk_obj = Chunk(
    document_id=doc.id,
    text="Vector databases enable semantic search",
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    metadata={
        "section": "introduction",
        "page": 1,
        "confidence": 0.95
    }
)

chunk = client.add_chunk(
    chunk=chunk_obj
)
```

**Example - Using Primitives:**

```python
# Add chunk using primitive parameters
chunk = client.add_chunk(
    document_id=doc.id,
    text="Vector databases enable semantic search",
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    metadata={
        "section": "introduction",
        "page": 1,
        "confidence": 0.95
    }
)
```

#### add_chunks

```python
def add_chunks(
    self,
    *,
    chunks: List[Union[Chunk, Dict[str, Any]]],
    document_id: Optional[Union[UUID, str]] = None,
) -> List[Chunk]:
```

Add multiple chunks to a document in a single batch operation. Supports both Chunk objects and dictionaries for maximum flexibility.

**Parameters:**
- `chunks` (List[Union[Chunk, Dict[str, Any]]]): List of chunks to add. Each item can be:
  - A `Chunk` object with text, embedding, and metadata
  - A `Dict` with "text" and "embedding" keys (metadata optional)
- `document_id` (Union[UUID, str]): Parent document identifier


**Returns:**
- `List[Chunk]`: List of created chunks

**Raises:**
- `NotFoundError`: If document not found
- `ValidationError`: If validation fails
- `ValueError`: If any chunk dict is missing required fields

**Example - Using Chunk Objects:**

```python
from my_vector_db import Chunk

chunks = [
    Chunk(
        document_id=doc.id,
        text="The quick brown fox jumps over the lazy dog",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        metadata={"source": "example", "position": 1}
    ),
    Chunk(
        document_id=doc.id,
        text="A fast red bird flies through the clear blue sky",
        embedding=[0.9, 0.1, 0.5, 0.3, 0.7],
        metadata={"source": "example", "position": 2}
    )
]

created = client.add_chunks(
    chunks=chunks
)
print(f"Added {len(created)} chunks")
```

**Example - Using Dictionaries:**

```python
chunk_dicts = [
    {
        "text": "Python is a high-level programming language",
        "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
        "metadata": {"topic": "programming"}
    },
    {
        "text": "FastAPI is a modern web framework for Python",
        "embedding": [0.3, 0.4, 0.5, 0.6, 0.7],
        "metadata": {"topic": "web"}
    }
]

created = client.add_chunks(
    document_id=doc.id,
    chunks=chunk_dicts
)
```

**Performance Note**: Batch operations are more efficient than adding chunks one at a time, as they use a single API request and perform atomic database operations.

#### create_chunk

**⚠️ DEPRECATED**: This method is deprecated and will be removed in a future version. Use `add_chunk()` instead for better developer experience.

```python
create_chunk(
    document_id: Union[UUID, str],
    text: str,
    embedding: List[float],
    metadata: Optional[Dict[str, Any]] = None
) -> Chunk
```

Create a new chunk in a document using primitive parameters.

**Deprecation Notice**: This method only supports primitive parameters. The new `add_chunk()` method supports both Chunk objects and primitives, providing a more flexible and intuitive interface for working with pre-computed embeddings.

**Migration Example:**

```python
# Old (deprecated):
chunk = client.create_chunk(
    document_id=doc.id,
    text="Sample text",
    embedding=[0.1, 0.2, 0.3],
    metadata={"key": "value"}
)

# New (recommended):
chunk = client.add_chunk(
    document_id=doc.id,
    text="Sample text",
    embedding=[0.1, 0.2, 0.3],
    metadata={"key": "value"}
)
```

#### get_chunk

```python
get_chunk(chunk_id: Union[UUID, str]) -> Chunk
```

Get a chunk by ID.

**Parameters:**
- `chunk_id` (Union[UUID, str]): Chunk identifier

**Returns:**
- `Chunk`: The chunk

**Raises:**
- `NotFoundError`: If chunk not found

#### list_chunks

```python
list_chunks(document_id: Union[UUID, str]) -> List[Chunk]
```

List all chunks in a document.

**Parameters:**
- `document_id` (Union[UUID, str]): Document identifier

**Returns:**
- `List[Chunk]`: List of chunks

**Raises:**
- `NotFoundError`: If document not found

#### list_all_chunks
```python
list_all_chunks(self, library_id: Union[UUID, str]) -> List[Chunk]
```
List all chunks in a library across all documents.

> Note: This method may be resource-intensive for libraries with a large number of chunks. It is meant for convieience in scenarios such as data auditing or analysis. For production use cases, consider using document-level chunk listing or search operations with filters.

**Parameters:**
- `library_id` (Union[UUID, str]): Library identifier
  
**Returns:**
- `List[Chunk]`: List of all chunks in the library
  
**Raises:**
- `NotFoundError`: If library not found

#### update_chunk

```python
update_chunk(
    chunk: Union[Chunk, UUID, str],
    *,
    text: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Chunk
```

Update a chunk. Pass either a Chunk object (fetch-modify-update) or a chunk ID with fields to update. Only provided fields are updated.
**Parameters:**
- `chunk` (Union[Chunk, UUID, str]): Chunk object or identifier
- `text` (Optional[str]): New text
- `embedding` (Optional[List[float]]): New embedding
- `metadata` (Optional[Dict[str, Any]]): New metadata

**Returns:**
- `Chunk`: The updated chunk

**Raises:**
- `NotFoundError`: If chunk not found

#### delete_chunk

```python
delete_chunk(chunk_id: Union[UUID, str]) -> None
```

Delete a chunk.

**Parameters:**
- `chunk_id` (Union[UUID, str]): Chunk identifier

**Raises:**
- `NotFoundError`: If chunk not found

### Search Operations

#### search

```python
search(
    self,
    library_id: Union[UUID, str],
    embedding: List[float],
    k: int = 10,
    filters: Optional[Union[SearchFilters, Dict[str, Any]]] = None,
    filter_function: Optional[Callable[[SearchResult], bool]] = None,
    combined_filters: Optional[SearchFiltersWithCallable] = None,
) -> SearchResponse:
```

Perform k-nearest neighbor vector search in a library.

**Parameters:**
- `library_id` (Union[UUID, str]): Library to search in
- `embedding` (List[float]): Query vector embedding
- `k` (int): Number of nearest neighbors to return (1-1000). Default: `10`
- `filters` (Optional[Union[SearchFilters, Dict[str, Any]]]): Declarative search filters applied server-side. Can be:
  - `SearchFilters` object (structured filters for metadata, time, document IDs)
  - `Dict` (converted to SearchFilters with validation)
- `filter_function` (Optional[Callable[[SearchResult], bool]]): Custom filter function applied client-side
- `combined_filters` (Optional[SearchFiltersWithCallable]): Combined declarative and custom filters (includes both metadata filters and custom_filter function)

**Returns:**
- `SearchResponse`: Search results with matching chunks and query time

**Raises:**
- `ValidationError`: If request validation fails or multiple filter parameters provided
- `NotFoundError`: If library not found
- `VectorDBError`: For other errors

**Note:**

- Only ONE of `filters`, `filter_function`, or `combined_filters` can be specified
- Declarative filters (filters param) are applied SERVER-SIDE for optimal performance
- Custom filter functions (filter_function param) are applied CLIENT-SIDE after fetching
- Combined filters (combined_filters param) apply declarative server-side then custom client-side
- When using client-side filtering, the SDK over-fetches (k*3) results and filters them locally
- Filter functions receive SearchResult objects with: chunk_id, document_id, text, score, metadata

**Example:**

```python
# Basic search
results = client.search(
    library_id=library.id,
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
    k=10
)

for result in results.results:
    print(f"Score: {result.score:.4f} - {result.text}")
```

## Persistence Management

The Vector Database supports optional data persistence, allowing you to save and restore database state to/from disk. This enables durable storage across container restarts, backups, and disaster recovery scenarios.

### Overview

When persistence is enabled:
- Database state is automatically saved to disk based on the configured threshold
- Manual snapshots can be triggered via the SDK
- The database can be restored from the latest snapshot on startup or on-demand
- All libraries, documents, chunks, embeddings, and metadata are preserved

**What is Persisted:**
- Libraries (including index configuration)
- Documents
- Chunks (text, embeddings, metadata)
- Vector indexes (FLAT indexes only - HNSW requires rebuild)

**Not Persisted:**
- In-flight API requests
- Connection state
- Server logs

### Container Setup

Persistence is configured via environment variables when running the Vector Database server. The most common deployment method is Docker/Docker Compose.

**Environment Variables:**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_STORAGE_PERSISTENCE` | boolean | `false` | Enable or disable persistence. Set to `"true"` to enable. |
| `STORAGE_DIR` | string | `"./data"` | Directory path where snapshot files are stored. Should be mounted as a volume. |
| `STORAGE_SAVE_EVERY` | integer | `-1` | Automatic save threshold (number of write operations). `-1` disables automatic saves. |

**Docker Compose Example:**

```yaml
services:
  vector-db:
    image: my-vector-db:latest
    container_name: vector-db-api
    ports:
      - "8000:8000"
    environment:
      # Enable persistence
      - ENABLE_STORAGE_PERSISTENCE=true

      # Storage directory inside container
      - STORAGE_DIR=/app/data

      # Auto-save every 100 write operations (optional)
      - STORAGE_SAVE_EVERY=100

    volumes:
      # Mount data directory for persistent storage
      # This ensures snapshots survive container restarts
      - ./data:/app/data

    restart: unless-stopped
```

**Docker Run Example:**

```bash
docker run -d \
  --name vector-db-api \
  -p 8000:8000 \
  -e ENABLE_STORAGE_PERSISTENCE=true \
  -e STORAGE_DIR=/app/data \
  -e STORAGE_SAVE_EVERY=100 \
  -v $(pwd)/data:/app/data \
  my-vector-db:latest
```

**Volume Mounting Best Practices:**

1. **Always use volumes** when persistence is enabled to ensure data survives container restarts
2. **Use absolute paths** or named volumes for production deployments
3. **Ensure write permissions** - the container process must have write access to the mounted directory
4. **Backup the data directory** regularly for disaster recovery

**Automatic Save Threshold:**

The `STORAGE_SAVE_EVERY` parameter controls automatic snapshot saves:

```yaml
# Save after every 10 write operations (creates/updates/deletes)
- STORAGE_SAVE_EVERY=10

# Save after every 100 operations (good for moderate workloads)
- STORAGE_SAVE_EVERY=100

# Save after every 1000 operations (good for high-throughput workloads)
- STORAGE_SAVE_EVERY=1000

# Disable automatic saves (manual saves only)
- STORAGE_SAVE_EVERY=-1
```

**Considerations:**
- Lower values (10-50): More durable, higher disk I/O overhead
- Higher values (500-1000): Less frequent saves, risk losing more data on crash
- `-1` (manual only): Maximum performance, requires explicit save calls

**Startup Behavior:**

When persistence is enabled and a snapshot file exists, the database automatically restores from the latest snapshot on startup:

```
INFO: Persistence: Enabled
INFO: Snapshot file found - restoring from snapshot
INFO: Restored 5 libraries, 42 documents, 1247 chunks
INFO: Database ready
```

### Client Operations

The SDK provides three methods for managing persistence programmatically.

#### save_snapshot

```python
save_snapshot() -> Dict[str, Any]
```

Manually trigger a database snapshot save. This saves the current database state to disk immediately, regardless of the automatic save threshold configured on the server.

**Returns:**
- `Dict[str, Any]`: Dictionary with save status and statistics

**Raises:**
- `ServiceUnavailableError`: If persistence is not enabled on the server
- `VectorDBError`: For other server errors

**Example:**

```python
# Trigger manual save
result = client.save_snapshot()

print(f"Snapshot saved successfully")
print(f"Libraries: {result['stats']['libraries']}")
print(f"Documents: {result['stats']['documents']}")
print(f"Chunks: {result['stats']['chunks']}")
print(f"Snapshot file: {result['snapshot_path']}")
```

**Response Structure:**

```python
{
    "status": "success",
    "message": "Snapshot saved successfully",
    "snapshot_path": "/app/data/snapshot.json",
    "timestamp": "2024-01-15T10:30:45.123456",
    "stats": {
        "libraries": 5,
        "documents": 42,
        "chunks": 1247
    }
}
```

**Use Cases:**
- Before performing bulk deletions or updates
- After completing a batch data import
- Before shutting down the server for maintenance
- Creating manual backup points

#### restore_snapshot

```python
restore_snapshot() -> Dict[str, Any]
```

Restore database state from the latest snapshot file.

**⚠️ WARNING:** This will **replace ALL current data** with the snapshot data. Any data created or modified after the snapshot was taken will be **permanently lost**.

**Returns:**
- `Dict[str, Any]`: Dictionary with restore status and restored counts

**Raises:**
- `NotFoundError`: If no snapshot file exists
- `ServiceUnavailableError`: If persistence is not enabled on the server
- `VectorDBError`: For other errors

**Example:**

```python
# Confirm before restoring
print("WARNING: This will replace all current data!")
confirm = input("Continue? (yes/no): ")

if confirm.lower() == "yes":
    result = client.restore_snapshot()

    print(f"Restore successful")
    print(f"Restored {result['stats']['libraries']} libraries")
    print(f"Restored {result['stats']['documents']} documents")
    print(f"Restored {result['stats']['chunks']} chunks")
    print(f"Snapshot timestamp: {result['snapshot_timestamp']}")
```

**Response Structure:**

```python
{
    "status": "success",
    "message": "Snapshot restored successfully",
    "snapshot_path": "/app/data/snapshot.json",
    "snapshot_timestamp": "2024-01-15T10:30:45.123456",
    "stats": {
        "libraries": 5,
        "documents": 42,
        "chunks": 1247
    }
}
```

**Use Cases:**
- Recovering from data corruption
- Reverting to a known good state after errors
- Resetting to a baseline state for testing
- Disaster recovery

**Important Notes:**
- The restore operation is **irreversible** - current data is cleared before restoring
- HNSW indexes need to be rebuilt after restore (FLAT indexes are ready immediately)
- All in-memory state is cleared and replaced with snapshot data

#### get_persistence_status

```python
get_persistence_status() -> Dict[str, Any]
```

Get current persistence status and statistics. Returns information about whether persistence is enabled, snapshot file status, and current database metrics.

**Returns:**
- `Dict[str, Any]`: Dictionary with persistence status details

**Example:**

```python
status = client.get_persistence_status()

print(f"Persistence enabled: {status['enabled']}")

if status['enabled']:
    print(f"Storage directory: {status['storage_dir']}")
    print(f"Save threshold: {status['save_threshold']}")
    print(f"Operations since save: {status['operations_since_save']}")

    if status['snapshot_exists']:
        print(f"Last snapshot: {status['last_snapshot_time']}")
        print(f"Snapshot size: {status['snapshot_size_bytes']} bytes")
    else:
        print("No snapshot file found")

# Check current database stats
print(f"\nCurrent database:")
print(f"Libraries: {status['current_stats']['libraries']}")
print(f"Documents: {status['current_stats']['documents']}")
print(f"Chunks: {status['current_stats']['chunks']}")
```

**Response Structure:**

```python
{
    "enabled": true,
    "storage_dir": "/app/data",
    "save_threshold": 100,
    "operations_since_save": 45,
    "snapshot_exists": true,
    "snapshot_path": "/app/data/snapshot.json",
    "last_snapshot_time": "2024-01-15T10:30:45.123456",
    "snapshot_size_bytes": 524288,
    "current_stats": {
        "libraries": 5,
        "documents": 42,
        "chunks": 1247
    }
}
```

**Use Cases:**
- Monitoring when the next automatic save will occur
- Checking if a snapshot file exists before restore
- Verifying persistence configuration
- Building monitoring dashboards

### Persistence Workflow

**Basic Workflow with Manual Saves:**

```python
from my_vector_db import VectorDBClient

with VectorDBClient(base_url="http://localhost:8000") as client:
    # Check persistence status
    status = client.get_persistence_status()
    if not status['enabled']:
        print("Warning: Persistence is not enabled on the server")

    # Create some data
    library = client.create_library(name="my_library")
    doc = client.create_document(library_id=library.id, name="doc1")

    # Add chunks
    chunks = [
        {"text": f"Chunk {i}", "embedding": [0.1*i, 0.2*i, 0.3*i]}
        for i in range(100)
    ]
    client.add_chunks(document_id=doc.id, chunks=chunks)

    # Manually save after bulk operation
    save_result = client.save_snapshot()
    print(f"Saved {save_result['stats']['chunks']} chunks to disk")
```

**Backup and Restore Workflow:**

```python
# 1. Create a backup before risky operations
print("Creating backup...")
backup_result = client.save_snapshot()
print(f"Backup created at {backup_result['timestamp']}")

try:
    # 2. Perform risky operations (bulk updates, deletions)
    for doc in documents_to_delete:
        client.delete_document(doc.id)

    # 3. Verify operations succeeded
    remaining_docs = client.list_documents(library_id=library.id)
    print(f"Operation successful - {len(remaining_docs)} documents remaining")

    # 4. Save the new state
    client.save_snapshot()

except Exception as e:
    # 5. Restore from backup on error
    print(f"Error occurred: {e}")
    print("Restoring from backup...")

    restore_result = client.restore_snapshot()
    print(f"Restored to backup from {restore_result['snapshot_timestamp']}")
```

**Monitoring Automatic Saves:**

```python
import time

# Poll persistence status to monitor automatic saves
while True:
    status = client.get_persistence_status()

    if status['enabled'] and status['save_threshold'] > 0:
        ops_remaining = status['save_threshold'] - status['operations_since_save']
        print(f"Operations until next save: {ops_remaining}")

        if ops_remaining < 10:
            print("WARNING: Approaching automatic save threshold")

    time.sleep(30)  # Check every 30 seconds
```

**Production Deployment Best Practices:**

```python
# 1. Enable persistence with reasonable thresholds
# docker-compose.yml:
#   ENABLE_STORAGE_PERSISTENCE=true
#   STORAGE_SAVE_EVERY=500  # Balance between durability and performance

# 2. Monitor persistence status
def check_persistence_health(client):
    status = client.get_persistence_status()

    if not status['enabled']:
        raise RuntimeError("Persistence is disabled - data loss risk!")

    if not status['snapshot_exists']:
        print("Warning: No snapshot file exists yet")

    # Alert if too many operations since last save
    if status['operations_since_save'] > 1000:
        print(f"WARNING: {status['operations_since_save']} ops since last save")

# 3. Implement graceful shutdown with save
import atexit

def shutdown_handler():
    try:
        client.save_snapshot()
        print("Final snapshot saved before shutdown")
    except Exception as e:
        print(f"Failed to save snapshot on shutdown: {e}")

atexit.register(shutdown_handler)

# 4. Regular backups (external to container)
# Backup the mounted volume directory:
#   tar -czf backup-$(date +%Y%m%d).tar.gz ./data/
```

**Error Handling:**

```python
from my_vector_db import (
    ServiceUnavailableError,
    NotFoundError,
    VectorDBError
)

try:
    # Attempt to save snapshot
    result = client.save_snapshot()
    print("Snapshot saved successfully")

except ServiceUnavailableError:
    print("ERROR: Persistence is not enabled on the server")
    print("Configure ENABLE_STORAGE_PERSISTENCE=true in environment")

except VectorDBError as e:
    print(f"Failed to save snapshot: {e}")

try:
    # Attempt to restore
    result = client.restore_snapshot()

except NotFoundError:
    print("ERROR: No snapshot file exists to restore from")

except ServiceUnavailableError:
    print("ERROR: Persistence is not enabled on the server")

except VectorDBError as e:
    print(f"Failed to restore snapshot: {e}")
```

## Filtering Guide

The SDK supports powerful filtering capabilities to refine search results based on metadata, time ranges, document IDs, or custom logic.

### Declarative Filters

Declarative filters are defined using Pydantic models and are applied server-side for optimal performance.

#### Metadata Filters

Filter chunks based on metadata field values.

**Available Operators:**

- `eq`: Equals
- `ne`: Not equals
- `gt`: Greater than
- `gte`: Greater than or equal
- `lt`: Less than
- `lte`: Less than or equal
- `in`: Value in list
- `not_in`: Value not in list
- `contains`: String contains substring
- `not_contains`: String does not contain substring
- `starts_with`: String starts with prefix
- `ends_with`: String ends with suffix

**Example - Simple Metadata Filter:**

```python
from my_vector_db import SearchFilters, FilterGroup, MetadataFilter, FilterOperator, LogicalOperator

# Filter by category
filters = SearchFilters(
    metadata=FilterGroup(
        operator=LogicalOperator.AND,
        filters=[
            MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="technology")
        ]
    )
)

results = client.search(
    library_id=library.id,
    embedding=query_vector,
    k=10,
    filters=filters
)
```

**Example - Complex Nested Filters:**

```python
# (category == "tech" OR category == "science") AND confidence > 0.8
filters = SearchFilters(
    metadata=FilterGroup(
        operator=LogicalOperator.AND,
        filters=[
            FilterGroup(
                operator=LogicalOperator.OR,
                filters=[
                    MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="tech"),
                    MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="science")
                ]
            ),
            MetadataFilter(field="confidence", operator=FilterOperator.GREATER_THAN, value=0.8)
        ]
    )
)

results = client.search(library_id=library.id, embedding=query_vector, k=10, filters=filters)
```

**Example - Using Dict (Convenience):**

```python
# Same filter as above, using dict syntax
filters = {
    "metadata": {
        "operator": "and",
        "filters": [
            {
                "operator": "or",
                "filters": [
                    {"field": "category", "operator": "eq", "value": "tech"},
                    {"field": "category", "operator": "eq", "value": "science"}
                ]
            },
            {"field": "confidence", "operator": "gt", "value": 0.8}
        ]
    }
}

results = client.search(library_id=library.id, embedding=query_vector, k=10, filters=filters)
```

#### Time-Based Filters

Filter chunks by creation time.

```python
from datetime import datetime

filters = SearchFilters(
    created_after=datetime(2024, 1, 1),
    created_before=datetime(2024, 12, 31)
)

results = client.search(library_id=library.id, embedding=query_vector, k=10, filters=filters)
```

#### Document ID Filters

Filter chunks by specific document IDs.

```python
filters = SearchFilters(
    document_ids=[str(doc1.id), str(doc2.id)]
)

results = client.search(library_id=library.id, embedding=query_vector, k=10, filters=filters)
```

### Custom Filter Functions

Custom filter functions allow you to implement arbitrary filtering logic in Python. These are applied client-side after fetching results from the API.

**Example - Simple Lambda:**

```python
results = client.search(
    library_id=library.id,
    embedding=query_vector,
    k=10,
    filter_function=lambda result: "machine learning" in result.text.lower()
)
```

**Example - Complex Function:**

```python
from my_vector_db.sdk.models import SearchResult

def custom_filter(result: SearchResult) -> bool:
    if result.metadata.get("category") != "research":
        return False

    keywords = ["neural", "network", "deep learning"]
    text_lower = result.text.lower()

    return any(keyword in text_lower for keyword in keywords)

results = client.search(
    library_id=library.id,
    embedding=query_vector,
    k=10,
    filter_function=custom_filter
)
```

### Filter Composition

Combine declarative (server-side) and custom (client-side) filters for maximum flexibility. Declarative filters narrow candidates by metadata, then custom filters refine by text content.

**How Combined Filtering Works:**

1. **Server-side**: Declarative filters (metadata, dates) are applied first, fetching k*3 candidates
2. **Client-side**: Custom filter function refines those candidates based on text or complex logic
3. **Result**: Top k results that pass both filter stages

**Example - Combined Server + Client Filtering:**

```python
from my_vector_db import SearchFiltersWithCallable, FilterGroup, MetadataFilter, FilterOperator, LogicalOperator

results = client.search(
    library_id=library.id,
    embedding=query_vector,
    k=10,
    combined_filters=SearchFiltersWithCallable(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="research"),
                MetadataFilter(field="confidence", operator=FilterOperator.GREATER_THAN, value=0.8)
            ]
        ),
        custom_filter=lambda result: "machine learning" in result.text.lower()
    )
)

# Workflow:
# 1. Server returns 30 results (k*3) where category="research" AND confidence>0.8
# 2. SDK applies custom filter: keep only results containing "machine learning"
# 3. Returns up to 10 results that pass BOTH filters
```

**Comparison:**

```python
# Server-side only (declarative filters)
results = client.search(
    library_id=lib.id,
    embedding=vec,
    k=10,
    filters=SearchFilters(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="research")
            ]
        )
    )
)
# → Fast, efficient, metadata-based filtering

# Client-side only (custom filter function)
results = client.search(
    library_id=lib.id,
    embedding=vec,
    k=10,
    filter_function=lambda result: "machine learning" in result.text.lower()
)
# → Flexible, text-based filtering, over-fetches k*3

# Combined (best of both)
results = client.search(
    library_id=lib.id,
    embedding=vec,
    k=10,
    combined_filters=SearchFiltersWithCallable(
        metadata=FilterGroup(...),
        custom_filter=lambda result: "machine learning" in result.text.lower()
    )
)
# → Efficient metadata filtering + flexible text filtering
```

## Vector Indexes

The Vector Database supports multiple index types, each optimized for different use cases and performance characteristics. Understanding the available indexes and their configuration options is crucial for optimal performance.

### Index Types

#### FLAT Index (Exact Search) ✅ Implemented

The FLAT index performs exhaustive brute-force search by comparing the query vector against every vector in the index. This guarantees exact nearest neighbor results with 100% recall.

**Characteristics:**
- **Search Complexity:** O(n * d) where n = number of vectors, d = dimension
- **Space Complexity:** O(n * d)
- **Recall:** 100% (exact search, guaranteed to find true nearest neighbors)
- **Build Time:** None (no index build required)
- **Best For:** Small to medium datasets (< 10,000 vectors), when accuracy is critical

**Pros:**
- Exact results - always finds true nearest neighbors
- Simple and predictable performance
- No index build overhead
- Ideal for development and testing
- Fast for small datasets

**Cons:**
- Scales linearly with dataset size O(n)
- Slow for large datasets
- Not optimized for high-dimensional spaces

**Configuration:**

```python
library = client.create_library(
    name="my_library",
    index_type="flat",
    index_config={
        "metric": "cosine"  # Distance metric (default: "cosine")
    }
)
```

**Supported Metrics:**

The FLAT index supports three distance/similarity metrics via the `metric` parameter in `index_config`:

| Metric | Description | Score Range | Best For |
|--------|-------------|-------------|----------|
| `cosine` | Cosine similarity - measures angle between vectors | -1 to 1 (higher = more similar) | Text embeddings, normalized vectors, semantic similarity |
| `euclidean` | Euclidean distance - straight-line distance in vector space | 0 to ∞ (lower = more similar, negated in results) | Image embeddings, spatial data, when magnitude matters |
| `dot_product` | Dot product / inner product - sum of element-wise products | -∞ to ∞ (higher = more similar) | Normalized vectors, when both direction and magnitude matter |

**Metric Formulas:**

- **Cosine Similarity:** `similarity = (A · B) / (||A|| * ||B||)`
  - Measures the angle between vectors (direction), ignoring magnitude
  - Returns 1 for identical direction, 0 for orthogonal, -1 for opposite direction

- **Euclidean Distance:** `distance = sqrt(Σ(A_i - B_i)²)`
  - L2 norm - measures straight-line distance in Euclidean space
  - Considers both direction and magnitude
  - Note: SDK negates distance for consistent "higher is better" scoring

- **Dot Product:** `dot_product = Σ(A_i * B_i)`
  - Inner product of vectors
  - For normalized vectors, equivalent to cosine similarity
  - Sensitive to vector magnitude

**Examples:**

```python
# Cosine similarity (default) - best for text embeddings
library = client.create_library(
    name="text_embeddings",
    index_type="flat",
    index_config={"metric": "cosine"}
)

# Euclidean distance - best for image embeddings
library = client.create_library(
    name="image_embeddings",
    index_type="flat",
    index_config={"metric": "euclidean"}
)

# Dot product - for specialized use cases
library = client.create_library(
    name="custom_vectors",
    index_type="flat",
    index_config={"metric": "dot_product"}
)
```

#### HNSW Index (Approximate Search) ⚠️ Not Yet Implemented

The Hierarchical Navigable Small World (HNSW) index is a graph-based approximate nearest neighbor algorithm that provides fast search for large datasets by trading some accuracy for speed.

**Status:** The HNSW index type is defined in the API but **not yet implemented**. Attempting to create a library with `index_type="hnsw"` will succeed, but the index will not be built and search operations will fail.

**Planned Characteristics:**
- **Search Complexity:** O(log n) approximate (not guaranteed exact)
- **Space Complexity:** O(n * M * log n) where M = max connections per node
- **Recall:** 95-99% (configurable, depends on ef_search parameter)
- **Build Time:** O(n * log n * d) - requires explicit index build
- **Best For:** Large datasets (> 10,000 vectors), when speed > perfect accuracy

**Planned Pros:**
- Fast approximate search O(log n)
- Excellent for high-dimensional data
- Scalable to millions of vectors
- Configurable accuracy/speed tradeoff

**Planned Cons:**
- Approximate results (may miss true nearest neighbors)
- Higher memory usage than FLAT
- Requires index build/rebuild after updates
- More complex configuration

**Planned Configuration:**

When implemented, HNSW will support the following parameters:

```python
library = client.create_library(
    name="large_dataset",
    index_type="hnsw",
    index_config={
        "m": 16,              # Max connections per node (default: 16)
                              # Higher = better recall, more memory
                              # Typical range: 8-64

        "ef_construction": 200,  # Candidate list size during build (default: 200)
                                # Higher = better index quality, slower build
                                # Typical range: 100-500

        "ef_search": 50,      # Candidate list size during search (default: 50)
                              # Higher = better recall, slower search
                              # Typical range: k to 500

        "metric": "cosine"    # Distance metric (same as FLAT)
    }
)

# After adding all chunks, build the index
client.build_index(library_id=library.id)
```

**HNSW Parameters Explained:**

- **m (max connections):** Number of bidirectional links per node in the graph
  - Lower values (8-16): Less memory, faster build, lower recall
  - Higher values (32-64): More memory, slower build, higher recall
  - Default: 16 (good balance)

- **ef_construction:** Size of the dynamic candidate list during index construction
  - Controls index build quality
  - Higher values create better quality index but take longer to build
  - Should be >= m and typically 10-40x larger
  - Default: 200

- **ef_search:** Size of the dynamic candidate list during search
  - Controls search accuracy vs speed tradeoff
  - Must be >= k (number of results requested)
  - Higher values increase recall but slow down search
  - Can be adjusted at search time (not just during library creation)
  - Default: 50

## Best Practices

### Index Selection

Choose the appropriate index type based on your use case:

**FLAT Index (Exact Search)**

Use when:
- Dataset is small (< 10,000 vectors)
- Accuracy is critical (need exact nearest neighbors)
- Build time doesn't matter

Characteristics:
- O(n) search time (linear scan)
- 100% recall (exact results)
- No index build required
- Best for development and testing

```python
library = client.create_library(
    name="small_dataset",
    index_type="flat",
    index_config={"metric": "euclidean"}
)
```

**HNSW Index (Approximate Search)**

Use when:
- Dataset is large (> 10,000 vectors)
- Speed is more important than perfect accuracy
- Can tolerate 95-99% recall

Characteristics:
- O(log n) search time (sublinear)
- High recall with proper configuration
- Requires index build after adding chunks
- Best for production at scale

```python
library = client.create_library(
    name="large_dataset",
    index_type="hnsw",
    index_config={
        "m": 16,              # Connections per layer (higher = better recall, more memory)
        "ef_construction": 200  # Build quality (higher = better recall, slower build)
    }
)

# Add all chunks first
for chunk_data in chunks:
    client.add_chunk(document_id=doc.id, **chunk_data)

# Build index once after all chunks are added
client.build_index(library_id=library.id)
```

### Metadata Design

Design metadata fields strategically for efficient filtering:

**Do:**

```python
# Use consistent, queryable field names
metadata = {
    "category": "technology",      # Categorical data
    "confidence": 0.95,            # Numeric scores
    "source": "arxiv",             # String identifiers
    "tags": ["ai", "ml", "nlp"],   # List for "in" queries
    "published_year": 2024         # Temporal data
}
```

**Don't:**

```python
# Avoid unstructured or inconsistent metadata
metadata = {
    "Category": "Technology",         # Inconsistent casing
    "conf": "95%",                    # String instead of number
    "metadata": {"nested": "data"},   # Deep nesting (hard to filter)
    "description": "long text..."     # Large text (use chunk.text instead)
}
```

**Indexing Strategy:**

Fields you plan to filter on frequently should be in metadata. Large text content should be in `chunk.text` for full-text custom filtering.

### Filter Strategy

Choose the right filtering approach based on your requirements:

**Use Declarative Filters When:**

- Filtering by exact metadata values
- Filtering by numeric ranges
- Filtering by time ranges or document IDs
- Performance is critical (server-side filtering is faster)

```python
# Server-side filtering (efficient)
filters = SearchFilters(
    metadata=FilterGroup(
        operator=LogicalOperator.AND,
        filters=[
            MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="tech"),
            MetadataFilter(field="confidence", operator=FilterOperator.GREATER_THAN, value=0.8)
        ]
    )
)
```

**Use Custom Filters When:**

- Need to filter on chunk text content
- Complex logic that can't be expressed declaratively
- Combining multiple conditions with custom rules
- Prototyping or development

```python
# Client-side filtering (flexible)
def is_relevant(result: SearchResult) -> bool:
    text = result.text.lower()
    has_keywords = any(kw in text for kw in ["neural", "network", "deep learning"])
    has_confidence = result.metadata.get("confidence", 0) > 0.7
    return has_keywords and has_confidence

results = client.search(library_id=library.id, embedding=vec, k=10, filter_function=is_relevant)
```

**Combining Declarative and Custom Filtering:**

Use `combined_filters` to apply both server-side metadata filtering and client-side custom logic:

```python
from my_vector_db import SearchFiltersWithCallable, FilterGroup, MetadataFilter

results = client.search(
    library_id=library.id,
    embedding=vec,
    k=10,
    combined_filters=SearchFiltersWithCallable(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="research"),
                MetadataFilter(field="confidence", operator=FilterOperator.GREATER_THAN, value=0.8)
            ]
        ),
        custom_filter=lambda result: "transformer architecture" in result.text.lower()
    )
)
```

### Connection Management

Always manage client lifecycle properly:

**Recommended: Context Manager**

```python
with VectorDBClient(base_url="http://localhost:8000") as client:
    results = client.search(library_id=lib_id, embedding=vec, k=10)
# Connection automatically closed
```

**Alternative: Explicit Close**

```python
client = VectorDBClient()
try:
    results = client.search(library_id=lib_id, embedding=vec, k=10)
finally:
    client.close()
```

**Connection Pooling:**

The SDK uses httpx which maintains a connection pool. For high-throughput applications, consider:

```python
# Single client instance for multiple requests
client = VectorDBClient(timeout=60.0)

for query in queries:
    results = client.search(library_id=lib_id, embedding=query, k=10)

client.close()
```

### Performance Considerations

**Over-Fetching Behavior**

The system automatically over-fetches when filters are present to ensure enough results:

- **Declarative filters**: Server fetches k*3, filters, returns k
- **Custom filters**: SDK fetches k*3, filters client-side, returns k
- **Both**: Server fetches k*9 (intentional for dual filtering)

**Implications:**

```python
# Highly selective filters may return fewer than k results
filters = SearchFilters(
    metadata=FilterGroup(
        operator=LogicalOperator.AND,
        filters=[
            MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="rare_category")
        ]
    )
)

results = client.search(library_id=lib_id, embedding=vec, k=10, filters=filters)
# May return < 10 results if fewer than 10 match in top 30 candidates
```

**Solutions:**

1. Request more results: `k=30` to get 10 after filtering
2. Use less selective filters
3. Increase over-fetch factor (requires server configuration)

**Network Transfer Optimization**

Custom filters transfer more data:

```python
# Efficient: No filters, transfers k results
results = client.search(library_id=lib_id, embedding=vec, k=10)

# Moderate: Declarative filters, transfers k results (filtered server-side)
results = client.search(library_id=lib_id, embedding=vec, k=10, filters={"metadata": {...}})

# Higher transfer: Custom filter, transfers k*3 results (filtered client-side)
results = client.search(library_id=lib_id, embedding=vec, k=10, filter_function=lambda r: ...)
```

For production systems, prefer declarative filters when possible.

**Batch Operations**

The SDK provides efficient batch operations via `add_chunks()` for inserting multiple chunks in a single request:

```python
# Efficient: Single API request for multiple chunks
chunks = [
    {"text": "Chunk 1", "embedding": [0.1, 0.2], "metadata": {"index": 1}},
    {"text": "Chunk 2", "embedding": [0.3, 0.4], "metadata": {"index": 2}},
    # ... more chunks
]
client.add_chunks(document_id=doc.id, chunks=chunks)

# For HNSW: Build index once after all chunks
client.build_index(library_id=library.id)
```

**Performance Comparison:**

```python
# Less efficient: N API requests
for chunk_data in chunks:
    client.add_chunk(document_id=doc.id, **chunk_data)

# More efficient: 1 API request (atomic operation)
client.add_chunks(document_id=doc.id, chunks=chunks)
```

Batch operations are atomic - either all chunks are created or none are, ensuring data consistency.

### Limitations

**Client-Side Filter Constraints**

Custom filter functions have limited chunk information:

```python
def my_filter(result: SearchResult) -> bool:
    # Available: result.chunk_id, result.text, result.metadata, result.document_id, result.score
    # NOT available: embedding, created_at (not included in SearchResult)
    return "keyword" in result.text
```

This is because `SearchResult` objects don't include embeddings or timestamps to reduce network transfer.

**Over-Fetch Factor**

The fixed over-fetch factor (k*3) may be insufficient for highly selective filters:

```python
# Problem: Filter matches only 1% of chunks
# Over-fetch: 10 * 3 = 30 candidates
# Matched: ~0.3 chunks (likely returns 0)

results = client.search(
    library_id=lib_id,
    embedding=vec,
    k=10,
    filter_function=lambda result: result.metadata.get("ultra_rare_tag") == "specific_value"
)
```

**Workaround:** Request more results or use declarative filters.

**Batch Operations**

For creating multiple chunks, use `add_chunks()` for efficient bulk operations:

```python
# Efficient: Use batch operations for multiple chunks (1 request)
chunks = [chunk_data[i] for i in range(1000)]
client.add_chunks(document_id=doc_id, chunks=chunks)

# Less efficient: Individual operations (1000 requests)
for i in range(1000):
    client.add_chunk(document_id=doc_id, **chunk_data[i])
```

Note: Batch operations for documents are also available via `add_documents()` (not yet implemented in SDK).

**Thread Safety**

The client is not thread-safe. Each thread should have its own client instance:

```python
from concurrent.futures import ThreadPoolExecutor

def search_thread(query_vec):
    # Each thread creates its own client
    with VectorDBClient() as client:
        return client.search(library_id=lib_id, embedding=query_vec, k=10)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(search_thread, query_vectors))
```

**Index Rebuild Required**

HNSW indexes must be rebuilt after adding/updating/deleting chunks:

```python
# Add new chunks
client.add_chunk(document_id=doc_id, **chunk_data)

# Rebuild index to include new chunks in search
client.build_index(library_id=lib_id)
```

FLAT indexes don't require rebuilding (automatic).

## MCP Server

The Vector Database includes a Model Context Protocol (MCP) server that enables seamless integration with Claude Desktop and other MCP-compatible AI assistants. The MCP server provides natural language access to your vector database through conversational tools.

### Overview

The MCP server exposes the Vector Database functionality through standardized MCP tools that can be called by AI assistants. It handles:

- **Automatic embedding generation** using Cohere's embedding API
- **Name-based resource lookup** (e.g., use library names instead of UUIDs)
- **Semantic search** with natural language queries
- **Database exploration** (listing libraries, documents, chunks)
- **Resource inspection** (detailed information about specific entities)

**Key Features:**

- **Zero embedding management**: Just provide text queries - embeddings are generated automatically
- **Human-friendly interface**: Use library/document names instead of UUIDs

### Installation

The MCP server is included with the Vector Database Python package.

### Configuration

#### Prerequisites

1. **Running Vector Database API**: The MCP server connects to a Vector Database API instance
   ```bash
   # Start the API server (default: http://localhost:8000)
   cd /path/to/my-vector-db
   docker compose up -d
   ```

2. **Cohere API Key**: Required for automatic embedding generation
   - Sign up at [https://cohere.com](https://cohere.com)
   - Get your API key from the dashboard
   - The server uses the `embed-english-light-v3.0` model (384 dimensions)

#### MCP Client Configuration

To connect the MCP server to Claude Desktop or other MCP clients, add a configuration entry to your MCP settings file.

**Claude Desktop Configuration:**

1. Locate your Claude Desktop MCP configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add the Vector Database MCP server configuration:

```json
{
  "mcpServers": {
    "my-vector-db": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/my-vector-db/src/my_vector_db/mcp",
        "run",
        "server.py",
        "--stdio"
      ],
      "env": {
        "VECTORDB_BASE_URL": "http://localhost:8000",
        "COHERE_API_KEY": "your-cohere-api-key-here"
      }
    }
  }
}
```

**Configuration Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `VECTORDB_BASE_URL` | URL of the Vector Database API | `http://localhost:8000` |
| `COHERE_API_KEY` | Cohere API key for embedding generation | Required |

#### Verifying Configuration

After configuring Claude Desktop:

1. **Restart Claude Desktop** to load the new MCP server
2. **Check for the tools icon** (🔨) in the input area - this indicates MCP tools are available
3. **Test the connection** by asking Claude to list libraries:
   ```
   Can you list all libraries in my vector database?
   ```

If configured correctly, Claude will use the `list_libraries` tool to fetch and display your libraries.

### Available Tools

The MCP server provides the following tools for interacting with your Vector Database:
- `search`
- `list_libraries`
- `list_documents`
- `list_chunks`
- `get_library`
- `get_document`
- `get_chunk`

#### TOOL: search

Perform semantic vector search using natural language queries.

**Signature:**
```python
async def search(library_name: str, query_text: str, k: int = 5) -> str
```

**Parameters:**
- `library_name` (str): Name or UUID of the library to search in
- `query_text` (str): Natural language query to search for
- `k` (int): Number of results to return (default: 5)

**Returns:**
- Formatted string with search results including scores, text, metadata, document IDs, and chunk IDs

**Example Usage:**
```
Find the top 5 chunks about machine learning in my research library
```

#### TOOL: list_libraries

List all libraries in the vector database.

**Signature:**
```python
async def list_libraries() -> str
```

**Returns:**
- Formatted list of all libraries with name, ID, index type, creation date, and metadata

**Example Usage:**
```
Show me all the libraries in my vector database
```

#### TOOL: list_documents

List all documents in a specific library.

**Signature:**
```python
async def list_documents(library_name: str) -> str
```

**Parameters:**
- `library_name` (str): Name or UUID of the library

**Returns:**
- Formatted list of documents with name, ID, chunk count, creation date, and metadata

**Example Usage:**
```
List all documents in the Research Papers library
```

#### TOOL: list_chunks

List all chunks in a specific document.

**Signature:**
```python
async def list_chunks(document_name: str) -> str
```

**Parameters:**
- `document_name` (str): Name or UUID of the document

**Returns:**
- Formatted list of chunks with ID, text preview, metadata, and creation date

**Example Usage:**
```
Show me the chunks in the "Attention Is All You Need" document
```

#### TOOL: get_library

Get detailed information about a specific library.

**Signature:**
```python
async def get_library(library_name: str) -> str
```

**Parameters:**
- `library_name` (str): Name or UUID of the library

**Returns:**
- Detailed library information including ID, index configuration, document count, timestamps, and metadata

**Example Usage:**
```
Get details about the Research Papers library
```

#### TOOL: get_document

Get detailed information about a specific document.

**Signature:**
```python
async def get_document(document_name: str) -> str
```

**Parameters:**
- `document_name` (str): Name or UUID of the document

**Returns:**
- Detailed document information including ID, library ID, chunk count, timestamps, and metadata

**Example Usage:**
```
Tell me about the "Attention Is All You Need" document
```

#### TOOL: get_chunk

Get detailed information about a specific chunk by its UUID.

**Signature:**
```python
async def get_chunk(chunk_id: str) -> str
```

**Parameters:**
- `chunk_id` (str): UUID of the chunk

**Returns:**
- Detailed chunk information including full text, document ID, metadata, and creation date

**Example Usage:**
```
Show me the details for chunk 990e8400-e29b-41d4-a716-446655440004
```

### Embedding Generation

The MCP server automatically generates embeddings for search queries using Cohere's API:

**Model:** `embed-english-light-v3.0`
- **Dimensions:** 384
- **Input Type:** `search_query` (optimized for retrieval)
- **Best For:** English text, semantic search, Q&A

**Usage:**
When you call the `search` tool, the server:
1. Takes your natural language query text
2. Calls Cohere's embedding API
3. Generates a 384-dimensional vector
4. Passes the vector to the Vector Database search endpoint

## Agno Integration

The Vector Database provides a native integration with the [Agno](https://github.com/agno-agi/agno) agentic framework through the `MyVectorDB` class, which implements Agno's `VectorDb` interface.

> Here is Agno's Knowledge Base documentation for reference: [Agno Knowledge Base](https://docs.agno.com/concepts/knowledge/overview)

### Overview

The `MyVectorDB` adapter enables Agno agents to use the Vector Database for knowledge storage and retrieval with automatic embedding generation and document management.

**Key Features:**
- Full Agno `VectorDb` interface implementation
- Automatic embedding generation via Cohere
- Document-based organization (each insert creates a new document container)
- Content hash-based deduplication
- Support for upsert operations (update by name)

### Quick Start

```python
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.anthropic import Claude
from agno.knowledge.embedder.cohere import CohereEmbedder
from my_vector_db.db import MyVectorDB

# Configure embedder (384 dimensions)
embedder = CohereEmbedder(
    id="embed-english-light-v3.0",
    input_type="search_document",
)

# Initialize vector database
vector_db = MyVectorDB(
    api_base_url="http://localhost:8000",
    library_name="Python Programming Guide",
    embedder=embedder,
)

# Create knowledge base
knowledge_base = Knowledge(
    name="My Knowledge Base",
    vector_db=vector_db,
    max_results=5,
)

# Add content
knowledge_base.add_content(
    name="python-intro",
    text_content="Python is a high-level programming language...",
    skip_if_exists=True,
)

# Create agent with knowledge
agent = Agent(
    name="Assistant",
    knowledge=knowledge_base,
    model=Claude(id="claude-sonnet-4-5"),
    search_knowledge=True,
)

# Use the agent
agent.print_response("What is Python?", stream=True)
```

### Configuration

#### MyVectorDB Parameters

```python
MyVectorDB(
    api_base_url: str = "http://localhost:8000",
    library_name: Optional[str] = None,
    index_type: str = "flat",
    embedder: Optional[Embedder] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    id: Optional[str] = None,
)
```

**Parameters:**
- `api_base_url` (str): Vector Database API endpoint. Default: `"http://localhost:8000"`
- `library_name` (Optional[str]): Library name to create/use. Default: `"agno_knowledge_base"`
- `index_type` (str): Index type (`"flat"` or `"hnsw"`). Default: `"flat"`
- `embedder` (Optional[Embedder]): Agno embedder instance. Default: `CohereEmbedder("embed-english-light-v3.0")`
- `name` (Optional[str]): Instance name for Agno
- `description` (Optional[str]): Instance description
- `id` (Optional[str]): Custom instance ID (auto-generated if not provided)

### Document Management

The adapter automatically manages documents and chunks:

**Insert Behavior:**
- Each `knowledge_base.add_content()` call creates a new document container
- Documents are chunked and embedded automatically by Agno
- Chunks are stored with metadata including content hash, name, and custom metadata

**Upsert Behavior:**
- Checks if content hash already exists (skips if duplicate)
- Deletes existing documents with the same name (enables updates)
- Inserts new content

**Content Hash Deduplication:**
- Content is hashed (MD5) to detect duplicates
- `skip_if_exists=True` prevents re-inserting identical content
- Enables efficient incremental knowledge base updates

### Library Management

The adapter can automatically connect to an existing library or create a new one. Just pass the human-friendly library name when initializing:

```python
vector_db = MyVectorDB(
    api_base_url="http://localhost:8000",
    library_name="Python Programming Guide",
    embedder=embedder,
)
```

### Search

The adapter handles semantic search with automatic embedding generation:

```python
# Search is performed automatically by Agno agent
agent.print_response("Find information about Python syntax")

# Or search directly via knowledge base
results = knowledge_base.search(query="Python syntax", max_results=5)
```

**Search Features:**
- Automatic query embedding with `input_type="search_query"` (Cohere)
- Server-side filtering support (metadata, time-based, document IDs)
- Returns Agno `Document` objects with metadata

### Example: Full Workflow

```python
from agno.knowledge.knowledge import Knowledge
from agno.agent import Agent
from agno.models.anthropic import Claude
from my_vector_db.db import MyVectorDB

# 1. Setup vector database
vector_db = MyVectorDB(
    api_base_url="http://localhost:8000",
    library_name="Documentation",
)

# 2. Create knowledge base
kb = Knowledge(
    name="Product Docs",
    vector_db=vector_db,
    max_results=10,
)

# 3. Load documentation
kb.add_content(
    name="installation-guide",
    text_content="To install the product, run: pip install my-product",
)

kb.add_content(
    name="quickstart-guide",
    text_content="Quick start: import my_product; my_product.run()",
)

# 4. Create agent
agent = Agent(
    name="DocBot",
    knowledge=kb,
    model=Claude(id="claude-sonnet-4-5"),
    search_knowledge=True,
    instructions=["You are a helpful documentation assistant"],
)

# 5. Query with automatic knowledge retrieval
agent.print_response("How do I install the product?", stream=True)
```

### Supported Operations

The `MyVectorDB` adapter implements the following Agno `VectorDb` interface methods:

| Method | Description | Status |
|--------|-------------|--------|
| `create()` | Create library | ✅ Implemented |
| `insert()` | Insert documents | ✅ Implemented |
| `upsert()` | Upsert documents | ✅ Implemented |
| `search()` | Semantic search | ✅ Implemented |
| `exists()` | Check library exists | ✅ Implemented |
| `get_count()` | Get chunk count | ✅ Implemented |
| `drop()` | Delete library | ✅ Implemented |
| `doc_exists()` | Check document exists | ✅ Implemented |
| `id_exists()` | Check chunk ID exists | ✅ Implemented |
| `delete_by_id()` | Delete chunk by ID | ✅ Implemented |
| `delete_by_name()` | Delete by name | ❌ Not implemented |
| `delete_by_metadata()` | Delete by metadata | ❌ Not implemented |

### Best Practices

**Embedder Configuration:**
- Use `CohereEmbedder` with `embed-english-light-v3.0` (384 dimensions)
- Set `input_type="search_document"` for document embedding
- Query embeddings automatically use `input_type="search_query"`

**Library Organization:**
- Use separate libraries for different knowledge domains
- Use descriptive library names (e.g., "Product Documentation", "Research Papers")

**Content Deduplication:**
- Enable `skip_if_exists=True` when adding content to avoid duplicates
- Use upsert when updating existing content by name

**Performance:**
- Use FLAT index for small to medium knowledge bases (< 10,000 chunks)
- Consider HNSW index for large knowledge bases (requires index build after updates)


## Type Reference

### Library

```python
class Library:
    id: UUID
    name: str
    document_ids: List[UUID]
    metadata: Dict[str, Any]
    index_type: str
    index_config: Dict[str, Any]
    created_at: datetime
```

### Document

```python
class Document:
    id: UUID
    name: str
    chunk_ids: List[UUID]
    metadata: Dict[str, Any]
    library_id: UUID
    created_at: datetime
```

### Chunk

```python
class Chunk:
    id: UUID
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    document_id: UUID
    created_at: datetime
```

### SearchResponse

```python
class SearchResponse:
    results: List[SearchResult]
    total: int
    query_time_ms: float
```

### SearchResult

```python
class SearchResult:
    chunk_id: UUID
    document_id: UUID
    text: str
    score: float
    metadata: Dict[str, Any]
```

### BuildIndexResult

Result of a build index operation.

```python
class BuildIndexResult:
    library_id: UUID
    total_vectors: int
    dimension: int
    index_type: str
    index_config: Dict[str, Any]
```

**Attributes:**
- `library_id`: UUID of the library
- `total_vectors`: Number of vectors in the index
- `dimension`: Dimensionality of the vectors
- `index_type`: Type of index ("flat", "hnsw")
- `index_config`: Configuration parameters for the index

**Example:**
```python
result = client.build_index(library_id=library.id)
print(f"Built {result.index_type} index with {result.total_vectors} vectors")
print(f"Vector dimension: {result.dimension}")
print(f"Config: {result.index_config}")
```

### SearchFilters

Base model for declarative search filters (API-safe, no custom functions).

```python
class SearchFilters:
    metadata: Optional[FilterGroup] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    document_ids: Optional[List[str]] = None
```

### SearchFiltersWithCallable

Extended model that adds custom filter function support (SDK/Python only).

```python
class SearchFiltersWithCallable(SearchFilters):
    custom_filter: Optional[Callable[[Chunk], bool]] = None
```

**Note:** When `custom_filter` is provided, it is applied **client-side in addition to** any declarative filters. The search workflow is:
1. Declarative filters (metadata, created_after, created_before, document_ids) are applied server-side first to narrow down candidates (over-fetching k*3 results)
2. The custom filter function is then applied client-side to those candidates
3. The top k results that pass both filters are returned

Use `SearchFilters` for pure declarative filtering (server-side only), or `SearchFiltersWithCallable` when you need custom Python logic (combined server-side + client-side filtering).

### FilterGroup

```python
class FilterGroup:
    operator: LogicalOperator  # "and" or "or"
    filters: List[Union[MetadataFilter, FilterGroup]]
```

### MetadataFilter

```python
class MetadataFilter:
    field: str
    operator: FilterOperator
    value: Union[str, int, float, bool, datetime, List[Any]]
```

### IndexType

```python
class IndexType(str, Enum):
    FLAT = "flat"  # Exact search
    HNSW = "hnsw"  # Approximate search
```

### FilterOperator

```python
class FilterOperator(str, Enum):
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
```

### LogicalOperator

```python
class LogicalOperator(str, Enum):
    AND = "and"
    OR = "or"
```

## Error Handling

The SDK raises specific exceptions for different error conditions:

```python
from my_vector_db import (
    VectorDBError,           # Base exception
    ValidationError,         # Invalid request data
    NotFoundError,          # Resource not found
    ServerError,            # Server-side error
    ServerConnectionError,  # Connection failed
    TimeoutError           # Request timeout
)

try:
    library = client.create_library(name="test")
    results = client.search(library_id=library.id, embedding=[0.1, 0.2], k=10)
except ValidationError as e:
    print(f"Invalid request: {e}")
except NotFoundError as e:
    print(f"Resource not found: {e}")
except ServerConnectionError as e:
    print(f"Connection failed: {e}")
except VectorDBError as e:
    print(f"Error: {e}")
```

Always use a context manager or explicitly call `close()` to ensure proper cleanup:

```python
# Recommended: context manager
with VectorDBClient() as client:
    results = client.search(library_id=lib_id, embedding=vec, k=10)

# Alternative: explicit close
client = VectorDBClient()
try:
    results = client.search(library_id=lib_id, embedding=vec, k=10)
finally:
    client.close()
```