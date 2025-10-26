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
      - [update\_chunk](#update_chunk)
      - [delete\_chunk](#delete_chunk)
    - [Search Operations](#search-operations)
      - [search](#search)
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
    library_id: Union[UUID, str],
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    index_type: Optional[str] = None,
    index_config: Optional[Dict[str, Any]] = None
) -> Library
```

Update a library. Only provided fields are updated.

**Parameters:**
- `library_id` (Union[UUID, str]): Library identifier
- `name` (Optional[str]): New name
- `metadata` (Optional[Dict[str, Any]]): New metadata
- `index_type` (Optional[Union[IndexType, str]]): New index type
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
    document_id: Union[UUID, str],
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Document
```

Update a document. Only provided fields are updated.

**Parameters:**
- `document_id` (Union[UUID, str]): Document identifier
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
add_chunk(
    document_id: Union[UUID, str],
    chunk: Optional[Chunk] = None,
    text: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Chunk
```

Add a chunk to a document. Supports both object-oriented and primitive parameter styles for maximum flexibility.

**Parameters:**
- `document_id` (Union[UUID, str]): Parent document identifier
- `chunk` (Optional[Chunk]): Pre-constructed Chunk object with text, embedding, and metadata
- `text` (Optional[str]): Text content (required if chunk not provided)
- `embedding` (Optional[List[float]]): Vector embedding (required if chunk not provided)
- `metadata` (Optional[Dict[str, Any]]): Optional metadata

**Note**: You must provide either a `chunk` object OR both `text` and `embedding` parameters.

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
    document_id=doc.id,
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
add_chunks(
    document_id: Union[UUID, str],
    chunks: List[Union[Chunk, Dict[str, Any]]]
) -> List[Chunk]
```

Add multiple chunks to a document in a single batch operation. Supports both Chunk objects and dictionaries for maximum flexibility.

**Parameters:**
- `document_id` (Union[UUID, str]): Parent document identifier
- `chunks` (List[Union[Chunk, Dict[str, Any]]]): List of chunks to add. Each item can be:
  - A `Chunk` object with text, embedding, and metadata
  - A `Dict` with "text" and "embedding" keys (metadata optional)

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
    document_id=doc.id,
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

#### update_chunk

```python
update_chunk(
    chunk_id: Union[UUID, str],
    text: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Chunk
```

Update a chunk. Only provided fields are updated.

**Parameters:**
- `chunk_id` (Union[UUID, str]): Chunk identifier
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
    library_id: Union[UUID, str],
    embedding: List[float],
    k: int = 10,
    filters: Optional[Union[SearchFilters, SearchFiltersWithCallable, Dict[str, Any], Callable]] = None
) -> SearchResponse
```

Perform k-nearest neighbor vector search in a library.

**Parameters:**
- `library_id` (Union[UUID, str]): Library to search in
- `embedding` (List[float]): Query vector embedding
- `k` (int): Number of nearest neighbors to return (1-1000). Default: `10`
- `filters` (Optional[Union[SearchFilters, SearchFiltersWithCallable, Dict[str, Any], Callable]]): Search filters. Can be:
  - `SearchFilters` object (declarative filters only - API-safe)
  - `SearchFiltersWithCallable` object (declarative + custom function support)
  - `Dict` (converted to SearchFilters with validation)
  - `Callable` (wrapped in SearchFiltersWithCallable as custom_filter)

**Returns:**
- `SearchResponse`: Search results with matching chunks and query time

**Raises:**
- `ValidationError`: If request validation fails
- `NotFoundError`: If library not found
- `VectorDBError`: For other errors

**Note:**

Custom filter functions are applied client-side after fetching from the API. The SDK over-fetches (k*3) results and filters them locally. This means:
- Custom filters work seamlessly with REST API
- More network transfer but enables text/custom filtering
- Filter functions receive Chunk objects with text and metadata (embedding and created_at fields are not available for client-side filters)

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

**Important:** Custom filter functions must be used with `SearchFiltersWithCallable`, not the base `SearchFilters` class. The base `SearchFilters` is designed for API-safe declarative filters only.

**Example - Simple Lambda (Direct):**

```python
# Filter chunks containing specific keywords (automatically wrapped in SearchFiltersWithCallable)
results = client.search(
    library_id=library.id,
    embedding=query_vector,
    k=10,
    filters=lambda chunk: "machine learning" in chunk.text.lower()
)
```

**Example - Complex Function:**

```python
from my_vector_db import Chunk

def custom_filter(chunk: Chunk) -> bool:
    # Complex filtering logic
    if chunk.metadata.get("category") != "research":
        return False

    keywords = ["neural", "network", "deep learning"]
    text_lower = chunk.text.lower()

    return any(keyword in text_lower for keyword in keywords)

results = client.search(
    library_id=library.id,
    embedding=query_vector,
    k=10,
    filters=custom_filter
)
```

**Example - Using SearchFiltersWithCallable (Explicit):**

```python
from my_vector_db import SearchFiltersWithCallable

filters = SearchFiltersWithCallable(
    custom_filter=lambda chunk: chunk.metadata.get("score", 0) > 50
)

results = client.search(library_id=library.id, embedding=query_vector, k=10, filters=filters)
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

# Combine server-side metadata filtering with client-side text filtering
filters = SearchFiltersWithCallable(
    # Server-side: Filter by metadata (fast, reduces network transfer)
    metadata=FilterGroup(
        operator=LogicalOperator.AND,
        filters=[
            MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="research"),
            MetadataFilter(field="confidence", operator=FilterOperator.GREATER_THAN, value=0.8)
        ]
    ),
    # Client-side: Filter by text content (flexible, applied to server results)
    custom_filter=lambda chunk: "machine learning" in chunk.text.lower()
)

results = client.search(library_id=library.id, embedding=query_vector, k=10, filters=filters)

# Workflow:
# 1. Server returns 30 results (k*3) where category="research" AND confidence>0.8
# 2. SDK applies custom filter: keep only results containing "machine learning"
# 3. Returns up to 10 results that pass BOTH filters
```

**Comparison:**

```python
# Server-side only (declarative filters)
filters = SearchFilters(
    metadata=FilterGroup(
        operator=LogicalOperator.AND,
        filters=[
            MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="research")
        ]
    )
)
results = client.search(library_id=lib.id, embedding=vec, k=10, filters=filters)
# → Fast, efficient, metadata-based filtering

# Client-side only (custom filter function)
results = client.search(
    library_id=lib.id,
    embedding=vec,
    k=10,
    filters=lambda chunk: "machine learning" in chunk.text.lower()
)
# → Flexible, text-based filtering, over-fetches k*3

# Combined (best of both)
filters = SearchFiltersWithCallable(
    metadata=FilterGroup(...),  # Server narrows by metadata
    custom_filter=lambda chunk: "machine learning" in chunk.text.lower()  # Client refines by text
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
def is_relevant(chunk: Chunk) -> bool:
    text = chunk.text.lower()
    has_keywords = any(kw in text for kw in ["neural", "network", "deep learning"])
    has_confidence = chunk.metadata.get("confidence", 0) > 0.7
    return has_keywords and has_confidence

results = client.search(library_id=library.id, embedding=vec, k=10, filters=is_relevant)
```

**Combining Declarative and Custom Filtering:**

To combine declarative (metadata, time-based) and custom (text, complex logic) filtering, implement both checks within a single custom function:

```python
from my_vector_db import Chunk

# Custom filter with embedded declarative logic
def combined_filter(chunk: Chunk) -> bool:
    # Declarative check: category metadata
    if chunk.metadata.get("category") != "research":
        return False

    # Declarative check: confidence threshold
    if chunk.metadata.get("confidence", 0) < 0.8:
        return False

    # Custom check: text content
    return "transformer architecture" in chunk.text.lower()

# Pass function directly - SDK wraps it automatically
results = client.search(library_id=library.id, embedding=vec, k=10, filters=combined_filter)
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
results = client.search(library_id=lib_id, embedding=vec, k=10, filters=lambda c: ...)
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
def my_filter(chunk: Chunk) -> bool:
    # Available: chunk.id, chunk.text, chunk.metadata, chunk.document_id
    # NOT available: chunk.embedding, chunk.created_at (set to defaults)
    return "keyword" in chunk.text
```

This is because `SearchResult` objects don't include embeddings or timestamps to reduce network transfer.

**Over-Fetch Factor**

The fixed over-fetch factor (k*3) may be insufficient for highly selective filters:

```python
# Problem: Filter matches only 1% of chunks
# Over-fetch: 10 * 3 = 30 candidates
# Matched: ~0.3 chunks (likely returns 0)

filters = lambda chunk: chunk.metadata.get("ultra_rare_tag") == "specific_value"
results = client.search(library_id=lib_id, embedding=vec, k=10, filters=filters)
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
