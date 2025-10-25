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
- [ ] **Metadata Filtering**: Support for filtering search results based on metadata field
- [ ] **Persistence to Disk**: Save and load database state to/from disk
- [ ] **Leader-Follower Architecture**: Basic implementation for read scalability
- [x] **Python SDK Client**: Fully featured client with documentation and examples
- [x] **Comprehensive Testing**: Unit tests for all components with >80% coverage
- [ ] **Durable Execution**: `QueryWorkflow` implemented with Temporal for orchestrating query execution steps

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ajshedivy/my-vector-db.git
cd my-vector-db

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Start the Server

```bash
# Or using Docker
docker-compose up -d
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Using the Python SDK

```python
from my_vector_db.sdk import VectorDBClient

# Initialize client
client = VectorDBClient(base_url="http://localhost:8000")

# Create a library with FLAT index
library = client.create_library(
    name="my_library",
    index_type="flat",
    index_config={"metric": "cosine"}
)

# Create a document
document = client.create_document(
    library_id=library.id,
    name="my_document"
)

# Add chunks with embeddings
chunk = client.create_chunk(
    library_id=library.id,
    document_id=document.id,
    text="Sample text content",
    embedding=[0.1, 0.2, 0.3, 0.4, 0.5]  # Your embedding vector
)

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
| GET | `/libraries/{id}` | Get library by ID |
| PUT | `/libraries/{id}` | Update library |
| DELETE | `/libraries/{id}` | Delete library |

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries/{id}/documents` | Create document |
| GET | `/libraries/{id}/documents` | List documents |
| GET | `/libraries/{id}/documents/{doc_id}` | Get document |
| PUT | `/libraries/{id}/documents/{doc_id}` | Update document |
| DELETE | `/libraries/{id}/documents/{doc_id}` | Delete document |

### Chunks
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries/{id}/documents/{doc_id}/chunks` | Create chunk |
| GET | `/libraries/{id}/documents/{doc_id}/chunks` | List chunks |
| GET | `/libraries/{id}/documents/{doc_id}/chunks/{chunk_id}` | Get chunk |
| PUT | `/libraries/{id}/documents/{doc_id}/chunks/{chunk_id}` | Update chunk |
| DELETE | `/libraries/{id}/documents/{doc_id}/chunks/{chunk_id}` | Delete chunk |

### Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries/{id}/query` | Perform k-nearest neighbor search |

## Architecture

```
src/my_vector_db/
├── api/                       # REST API layer
│   ├── routes.py              # FastAPI endpoints
│   └── schemas.py             # Request/Response DTOs
├── domain/                    # Domain models
│   └── models.py              # Chunk, Document, Library
├── indexes/                   # Vector index implementations
│   ├── base.py                # Abstract VectorIndex interface
│   ├── flat.py                # Brute force exact search
│   └── hnsw.py                # Graph-based approximate search
├── sdk/                       # Python SDK
│   ├── client.py              # VectorDBClient
│   ├── models.py              # Pydantic models
│   ├── exceptions.py          # Custom exceptions
│   └── errors.py              # Error handling decorators
├── db/                        # Framework integrations
│   └── my_vector_db.py        # Agno VectorDb adapter
├── services/                  # Business logic
│   ├── library_service.py
│   ├── document_service.py
│   └── search_service.py
├── storage.py                # Thread-safe in-memory storage
└── main.py                   # Application entry point
```
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

- `sdk_example.py`: Basic SDK usage with CRUD operations
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
