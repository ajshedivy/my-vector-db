# Vector Database

A production-ready vector database with REST API, Python SDK, and Agno framework integration. Built from scratch with FastAPI, featuring multiple indexing algorithms and comprehensive error handling.

## Features

### Core Functionality
- **REST API** with FastAPI for vector storage and retrieval
- **Python SDK** with type-safe interfaces and comprehensive error handling
- **Agno Framework Integration** for agent-based applications
- **Multiple Vector Indexes** implemented from scratch:
  - Flat Index
  - HNSW (Hierarchical Navigable Small World)
- **Thread-Safe Storage** with reentrant locking for concurrent operations
- **Metadata Support** for filtering and organizing vectors
- **Distance Metrics**: Cosine similarity, Euclidean, and dot product

### Extra Points
- [x] **Metadata Filtering**: Support for filtering search results based on metadata field
- [ ] **Persistence to Disk**: Save and load database state to/from disk (not yet implemented)
- [x] **Python SDK Client**: Fully featured client with documentation and examples
- [x] **Comprehensive Testing**: Unit tests for all components with >80% coverage

## Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- docker or podman
> Note: we use uv for all things Python in this project, including dependency management, testing, and running the server.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ajshedivy/my-vector-db.git
cd my-vector-db

# Install dependencies with uv (recommended)
uv sync
```

### Start the Server

```bash
# using Docker
docker compose up -d

# using podman
podman compose up -d
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Using the Python SDK

The Python SDK provides a convenient interface for interacting with the Vector Database API. Below is an example of how to use the SDK to create a library, add documents and chunks, and perform a similarity search.

```python
from my_vector_db.sdk import VectorDBClient

# Initialize client
client = VectorDBClient(base_url="http://localhost:8000")

# Create a library with FLAT index
library = client.create_library(
    name="my_library",
    index_type="flat",
    index_config={"metric": "euclidean"}
)

# Create a document
document = client.create_document(
    library_id=library.id,
    name="my_document"
)

# add chunks to the document
chunk = client.add_chunk(
    document_id=document.id,
    text="The quick brown fox jumps over the lazy dog",
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
)

chunk2 = client.add_chunk(
    document_id=document.id,
    text="The slow brown cat jumps over the lazy dog",
    embedding=[0.1, 0.7, 0.3, 0.4, 0.2],
)

# Batch add multiple chunks
chunks = [
    {"text": "Example 1", "embedding": [0.2, 0.3, 0.4, 0.5, 0.6]},
    {"text": "Example 2", "embedding": [0.3, 0.4, 0.5, 0.6, 0.7]},
]
created_chunks = client.add_chunks(document_id=document.id, chunks=chunks)

# Perform similarity search
results = client.search(
    library_id=library.id,
    embedding=[0.15, 0.25, 0.35, 0.45, 0.55],  # Query vector
    k=5
)

# Process results
for result in results.results:
    print(f"Score: {result.score:.4f} - {result.text}")

# Cleanup
client.close()
```

### Using with Agno Framework

```python
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.anthropic import Claude
from my_vector_db.db.my_vector_db import MyVectorDB

# Create vector database
vector_db = MyVectorDB(
    api_base_url="http://localhost:8000",
    library_name="knowledge_base",
    index_type="flat"
)

# Create knowledge base
knowledge = Knowledge(
    name="My Knowledge Base",
    vector_db=vector_db,
    max_results=5
)

# Add content
knowledge.add_content(
    name="example",
    text_content="Your content here"
)

# Create agent with knowledge
agent = Agent(
    name="Assistant",
    knowledge=knowledge,
    model=Claude(id="claude-sonnet-4-5"),
    search_knowledge=True
)

agent.cli_app(stream=True)
```

## API Reference

### Libraries
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries` | Create a new library |
| GET | `/libraries` | List all libraries |
| GET | `/libraries/{library_id}` | Get library by ID |
| PUT | `/libraries/{library_id}` | Update library |
| DELETE | `/libraries/{library_id}` | Delete library |
| POST | `/libraries/{library_id}/build-index` | Build or rebuild vector index |

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries/{library_id}/documents` | Create document in library |
| POST | `/libraries/{library_id}/documents/batch` | Batch create documents in library |
| GET | `/libraries/{library_id}/documents` | List documents in library |
| GET | `/documents/{document_id}` | Get document by ID |
| PUT | `/documents/{document_id}` | Update document |
| DELETE | `/documents/{document_id}` | Delete document |

### Chunks
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/{document_id}/chunks` | Create chunk in document |
| POST | `/documents/{document_id}/chunks/batch` | Batch create chunks in document |
| GET | `/documents/{document_id}/chunks` | List chunks in document |
| GET | `/chunks/{chunk_id}` | Get chunk by ID |
| PUT | `/chunks/{chunk_id}` | Update chunk |
| DELETE | `/chunks/{chunk_id}` | Delete chunk |

### Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries/{library_id}/query` | Perform k-nearest neighbor search |

**Note:** The API uses a simplified flat structure for documents and chunks. Once created, documents and chunks can be accessed directly by their globally unique IDs without specifying parent resources. This reduces redundancy since UUIDs are unique across the entire system.

## Architecture

### Project Structure

The project follows Domain-Driven Design principles:
- **Domain Layer**: Core business models (Pydantic)
- **Service Layer**: Business logic and orchestration
- **API Layer**: HTTP endpoints and DTOs
- **Infrastructure Layer**: Storage and index implementations

## Vector Index Algorithms

### Flat Index
- **Time Complexity**: O(n·d) search, O(1) insert
- **Space Complexity**: O(n·d)
- **Best For**: Small to medium datasets requiring exact results
- **Characteristics**: 100% recall, simple implementation

### HNSW (Hierarchical Navigable Small World)
- **Time Complexity**: O(log n) approximate search
- **Space Complexity**: O(n·M)
- **Best For**: Large datasets requiring fast approximate search
- **Parameters**:
  - `M`: Maximum connections per node (default: 16)
  - `ef_construction`: Construction-time search depth (default: 200)
  - `ef_search`: Query-time search depth (default: 50)

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=my_vector_db --cov-report=html

# Run specific test file
uv run pytest tests/test_sdk.py -v

# Run with verbose output
uv run pytest -vv
```

## Configuration

### Environment Variables

```bash
# API server settings
VECTOR_DB_HOST=0.0.0.0
VECTOR_DB_PORT=8000

# SDK client settings
VECTOR_DB_BASE_URL=http://localhost:8000
VECTOR_DB_TIMEOUT=30
```

### Docker Configuration

```yaml
# Production deployment
docker-compose up -d

# Development with hot reload
docker-compose --profile dev up vector-db-dev
```

## Error Handling

The SDK provides comprehensive exception handling:

```python
from my_vector_db.sdk import (
    VectorDBClient,
    VectorDBError,          # Base exception
    ValidationError,        # 400, 422 errors
    NotFoundError,          # 404 errors
    ServerError,            # 500+ errors
    ServerConnectionError,  # Connection failures
    TimeoutError           # Request timeouts
)

try:
    result = client.search(library_id=lib_id, embedding=vec, k=10)
except NotFoundError:
    print("Library not found")
except ValidationError as e:
    print(f"Invalid request: {e}")
except ServerConnectionError:
    print("Cannot connect to server")
except TimeoutError:
    print("Request timed out")
except VectorDBError as e:
    print(f"Unexpected error: {e}")
```

## Examples

See the `examples/` directory for complete usage examples:

- `sdk_example.py`: Basic SDK usage with CRUD operations and custom filters
- `batch_example.py`: Efficient batch operations for adding multiple chunks
- `agno_example.py`: Integration with Agno framework for agent applications

## Development

### Code Quality Tools

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy src/my_vector_db
```

## Author

Adam Shedivy (ajshedivyaj@gmail.com)
