# My Vector Database

A lightweight, production-ready vector database with a RESTful API and Python SDK. Built with FastAPI and Pydantic, it supports storing documents with vector embeddings and provides efficient similarity search using custom-implemented vector indexes.

## Features

### Core Features ✅

- [x] **CRUD Operations**: Full create, read, update, delete for libraries, documents, and chunks
- [x] **Vector Indexes**: FLAT index implementation with cosine, euclidean, and dot product metrics
- [x] **K-Nearest Neighbor Search**: Efficient similarity search with configurable result count
- [x] **Metadata Support**: Flexible metadata dictionaries for all entities
- [x] **Thread-Safe Operations**: Proper concurrency control for read/write operations

### Advanced Features ✅

- [x] **Metadata Filtering**: Server-side declarative filters with complex logic (AND/OR, nested filters)
- [x] **Custom Filter Functions**: Client-side Python filter functions for advanced use cases
- [x] **Data Persistence**: Save/restore database snapshots with automatic and manual triggers
- [x] **Python SDK Client**: Fully-featured, type-safe SDK with comprehensive documentation
- [x] **Batch Operations**: Efficient bulk insert for chunks with atomic transactions
- [x] **Comprehensive Testing**: Unit tests with >80% coverage
- [x] **Agent Framework Integration**: integration with Agno framework for basic RAG applications
- [x] **MCP Server**: MCP server for vector database

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker or Podman

> **Note:** This project uses `uv` for all Python operations including dependency management, testing, and running the server.

## Core Concepts

My Vector Database organizes data in a three-tier hierarchy:

```
Libraries → Documents → Chunks
```

- **Libraries**: Top-level containers with their own vector index configuration (FLAT, HNSW)
- **Documents**: Logical groupings of related text chunks within a library
- **Chunks**: Individual pieces of text with vector embeddings and metadata (the searchable units)

This structure allows you to organize and search your data at different levels of granularity. For example, a library might contain multiple PDF documents, each document split into searchable chunks with embeddings.

## Quick Start

### Start the Server

See the [Run the Vector Database Server](docs/README.md#run-the-vector-database-server) guide for detailed setup options.

Quick start with Docker Compose (recommended):

```bash
# Clone the repository
git clone https://github.com/ajshedivy/my-vector-db.git
cd my-vector-db

# Start with Docker Compose
docker compose up -d
```

Or run directly with Docker:

```bash
docker run -d --name my-vector-db -p 8000:8000 ghcr.io/ajshedivy/my-vector-db:latest
```

The API will be available at:
- **API Endpoint**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Install the SDK

```bash
# Install from PyPI
pip install my-vector-db

# Or install from source
git clone https://github.com/ajshedivy/my-vector-db.git
cd my-vector-db
uv sync
```

### Using the Python SDK

Here's a complete workflow in ~15 lines:

```python
from my_vector_db.sdk import VectorDBClient

client = VectorDBClient(base_url="http://localhost:8000")

# Create library → document → chunks
library = client.create_library(name="docs", index_type="flat")
document = client.create_document(library_id=library.id, name="intro")

# Add chunks with embeddings
chunks = client.add_chunks(
    document_id=document.id,
    chunks=[
        {
            "text": "AI enables pattern recognition",
            "embedding": [0.9, 0.8, 0.1, 0.2, 0.3],
        },
        {
            "text": "Deep learning uses neural networks",
            "embedding": [0.85, 0.75, 0.15, 0.25, 0.35],
        },
    ],
)

# Search for similar content
results = client.search(
    library_id=library.id, embedding=[0.88, 0.78, 0.12, 0.22, 0.32], k=5
)
for result in results.results:
    print(f"{result.score:.4f} - {result.text}")
```

**For comprehensive SDK documentation**, see the [SDK Reference Guide](docs/README.md) which covers:
- Detailed API reference for all operations
- Advanced filtering (metadata, time-based, custom functions)
- Persistence management (snapshots, backups, restore)
- Vector index configuration and best practices
- Performance optimization and error handling

**For more usage examples**, see the [`examples/`](examples/) directory.

## Architecture

### Data Model

The vector database has three main data components organized in a hierarchical structure:

![alt text](docs/data_model.png "Data Model Diagram")

**Key Relationships:**
- One Library contains **many** Documents (1:N)
- One Document contains **many** Chunks (1:N)
- Search operations are performed at the **Library** level across all chunks
- Vector indexes are built and configured per **Library**

![alt text](docs/api.png)

**Key Design Principles:**
1. **Layered Architecture**: Clean separation (API → Service → Storage → Index)
2. **Thread-Safe**: RLock-based synchronization for concurrent operations
3. **Type-Safe**: Full Pydantic validation throughout
4. **Persistence**: JSON snapshots with atomic writes
5. **Filtering**: Post-filtering strategy with declarative and custom options

#### Libraries
A library is a top-level container for documents with its own vector index configuration.

**Attributes:**
- `id`: Unique identifier (UUID)
- `name`: Library name
- `index_type`: Type of vector index (`flat`, `hnsw`)
- `index_config`: Index-specific configuration (e.g., distance metric)
- `metadata`: Optional metadata dictionary
- `document_ids`: List of document IDs in the library
- `created_at`, `updated_at`: Timestamps

#### Documents
A document represents a logical grouping of text chunks within a library.

**Attributes:**
- `id`: Unique identifier (UUID)
- `library_id`: Parent library ID
- `name`: Document name
- `metadata`: Optional metadata dictionary
- `chunk_ids`: List of chunk IDs in the document
- `created_at`, `updated_at`: Timestamps

#### Chunks
A chunk is a single piece of text with its vector embedding and metadata (the searchable unit).

**Attributes:**
- `id`: Unique identifier (UUID)
- `document_id`: Parent document ID
- `text`: Text content
- `embedding`: Vector embedding (list of floats)
- `metadata`: Optional metadata dictionary
- `created_at`, `updated_at`: Timestamps

### Project Structure

The project follows **Domain-Driven Design** principles with clear separation of concerns:

- **Domain Layer** (`src/my_vector_db/domain/`): Core business models using Pydantic
- **Service Layer** (`src/my_vector_db/services/`): Business logic and orchestration
- **API Layer** (`src/my_vector_db/api/`): FastAPI endpoints and request/response DTOs
- **Infrastructure Layer** (`src/my_vector_db/infrastructure/`): Storage and vector index implementations

### Vector Indexes

#### FLAT Index (Exact Search) ✅ Implemented

Exhaustive brute-force search comparing the query vector against every stored vector.

**Characteristics:**
- **Time Complexity**: O(n·d) search, O(1) insert
- **Space Complexity**: O(n·d)
- **Recall**: 100% (exact results, guaranteed true nearest neighbors)
- **Best For**: Small to medium datasets (< 10,000 vectors), when accuracy is critical

**Supported Metrics**: `cosine`, `euclidean`, `dot_product`

```python
library = client.create_library(
    name="my_library",
    index_type="flat",
    index_config={"metric": "cosine"}
)
```

#### HNSW Index (Approximate Search) ⚠️ Planned

Hierarchical Navigable Small World graph-based approximate nearest neighbor algorithm.

**Planned Characteristics:**
- **Time Complexity**: O(log n) search (approximate)
- **Space Complexity**: O(n·M·log n)
- **Recall**: 95-99% (configurable)
- **Best For**: Large datasets (> 10,000 vectors), when speed is more important than perfect accuracy

**Note**: The HNSW index is defined in the API but not yet fully implemented.

For detailed index configuration and performance considerations, see the [Vector Indexes Documentation](docs/README.md#vector-indexes).

###  API Reference

#### Libraries
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries` | Create a new library |
| GET | `/libraries` | List all libraries |
| GET | `/libraries/{library_id}` | Get library by ID |
| PUT | `/libraries/{library_id}` | Update library |
| DELETE | `/libraries/{library_id}` | Delete library |
| POST | `/libraries/{library_id}/build-index` | Build or rebuild vector index |

#### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries/{library_id}/documents` | Create document in library |
| POST | `/libraries/{library_id}/documents/batch` | Batch create documents in library |
| GET | `/libraries/{library_id}/documents` | List documents in library |
| GET | `/documents/{document_id}` | Get document by ID |
| PUT | `/documents/{document_id}` | Update document |
| DELETE | `/documents/{document_id}` | Delete document |

#### Chunks
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents/{document_id}/chunks` | Create chunk in document |
| POST | `/documents/{document_id}/chunks/batch` | Batch create chunks in document |
| GET | `/documents/{document_id}/chunks` | List chunks in document |
| GET | `/chunks/{chunk_id}` | Get chunk by ID |
| PUT | `/chunks/{chunk_id}` | Update chunk |
| DELETE | `/chunks/{chunk_id}` | Delete chunk |

#### Search
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries/{library_id}/query` | Perform k-nearest neighbor search |

#### Persistence
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/persistence/save` | Save database snapshot to disk |
| POST | `/persistence/restore` | Restore from latest snapshot |
| GET | `/persistence/status` | Get persistence status and statistics |

**API Design Note**: The API uses a simplified flat structure for direct resource access. Once created, documents and chunks can be accessed by their globally unique UUIDs without specifying parent resources, reducing API verbosity.

For interactive API exploration, visit http://localhost:8000/docs after starting the server.

## Demo

Here is an end-to-end demo showcasing key features of the vector database. The demo is divided into several parts:

- **Database setup and python client SDK**
- **Vector Search**
- **Vector Search with Filtering**
- **Persistence**
- **RAG pipeline with Agno Framework**
- **End-to-end document processing and Agent integration**
- **MCP Server**

The demo videos can be found at the links below. Set the playback quality to 1080p for best clarity. (Also set the speed to 1.25x or 1.5x to save time!). 

The videos walk through the demo notebook found here: [Demo Notebook](demo.ipynb)


### 1. Database setup and python client SDK

> [video link](https://ibm.box.com/s/ek0t1ro4iv97gzcv9uf0p0ydpnaglt4e)

This video covers:
- Installing dependencies
- Starting the server with Docker
- Using the Python SDK to create libraries, documents, and chunks

**📚 Documentation:**
- [Installation](docs/README.md#installation)
- [Quick Start](docs/README.md#quick-start)
- [VectorDBClient](docs/README.md#vectordbclient)
- [Library Operations](docs/README.md#library-operations)
- [Document Operations](docs/README.md#document-operations)
- [Chunk Operations](docs/README.md#chunk-operations)


### 2. Vector Search

> [video link](https://ibm.box.com/s/9hia05hsdq8o1ioh2517z260m7i6bhl9)

This video covers:
- Performing k-nearest neighbor search
- Understanding search results and scores

**📚 Documentation:**
- [Search Operations](docs/README.md#search-operations)
- [Vector Indexes](docs/README.md#vector-indexes)
- [FLAT Index](docs/README.md#flat-index-exact-search--implemented)
- [Best Practices - Index Selection](docs/README.md#index-selection)


### 3. Vector Search with Filtering
> [video link](https://ibm.box.com/s/tnhjho4fktqv1ifh9o8azaeputcf0pw7)

This video covers:
- Using metadata filters in search queries
- Creating custom filter functions for advanced filtering
- Combining multiple filter conditions

**📚 Documentation:**
- [Filtering Guide](docs/README.md#filtering-guide)
- [Declarative Filters](docs/README.md#declarative-filters)
- [Metadata Filters](docs/README.md#metadata-filters)
- [Custom Filter Functions](docs/README.md#custom-filter-functions)
- [Filter Composition](docs/README.md#filter-composition)
- [Best Practices - Filter Strategy](docs/README.md#filter-strategy)

### 4. Persistence

> [video link](https://ibm.box.com/s/sctelxnna2dzjhns85pz38uc6udnvxrr)

This video covers:
- Saving database snapshots to disk
- Restoring the database from snapshots
- Configuring automatic persistence settings

**📚 Documentation:**
- [Persistence Management](docs/README.md#persistence-management)
- [Container Setup](docs/README.md#container-setup)
- [save_snapshot](docs/README.md#save_snapshot)
- [restore_snapshot](docs/README.md#restore_snapshot)
- [get_persistence_status](docs/README.md#get_persistence_status)
- [Persistence Workflow](docs/README.md#persistence-workflow)

### 5. RAG pipeline with Agno Framework

> [video link](https://ibm.box.com/s/suk7qdwkghcv9n62h3a5f4ll98c71w12)

This video covers:
- Integrating the vector database with the Agno agent framework
- Creating a knowledge base using the vector database
- Building an agent that utilizes the knowledge base for retrieval-augmented generation (RAG)

**📚 Documentation:**
- [Agno Integration](docs/README.md#agno-integration)
- [Quick Start - Agno](docs/README.md#quick-start-1)
- [MyVectorDB Parameters](docs/README.md#myvectordb-parameters)
- [Document Management - Agno](docs/README.md#document-management)
- [Search - Agno](docs/README.md#search-2)
- [Example: Full Workflow](docs/README.md#example-full-workflow)

### 6. End-to-end document processing and Agent integration

> [video link](https://ibm.box.com/s/uxy9k8ff0hovtr0z874zhqnaztl9imaw)

This video covers:
- Processing documents and adding them to the vector database
- Using the Agno agent to answer questions based on the stored documents
- Demonstrating the complete workflow from document ingestion to question answering

**📚 Documentation:**
- [Agno Integration](docs/README.md#agno-integration)

### 7. MCP Server

> [video link](https://ibm.box.com/s/mkswhk7yxf86nzqsxxt81kfni3t6ug3x)

This video covers:
- MCP server setup and configuration
- connecting MCP server to LM Studio
- Invoking tools via chat for fetching library and document info
- Run vector search from the chat using `search` tool

**📚 Documentation:**
- [MCP Server](docs/README.md#mcp-server)
- [MCP Configuration](docs/README.md#configuration)
- [MCP Client Configuration](docs/README.md#mcp-client-configuration)
- [Available Tools](docs/README.md#available-tools)


## Integrations

### Agno Framework

The vector database integrates seamlessly with the [Agno](https://github.com/agno-ai/agno) agent framework for building RAG (Retrieval-Augmented Generation) applications.

```python
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.anthropic import Claude
from my_vector_db.db import MyVectorDB

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

For a complete working example, see [`examples/agno_example.py`](examples/agno_example.py).

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

# Persistence settings (optional)
ENABLE_STORAGE_PERSISTENCE=false
STORAGE_DIR=./data
STORAGE_SAVE_EVERY=-1  # -1 disables automatic saves
```

For detailed persistence configuration and workflows, see the [Persistence Management Documentation](docs/README.md#persistence-management).

## Examples

The [`examples/`](examples/README.md) directory contains complete usage examples

## Development

### Code Quality Tools

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uvx ty check src/my_vector_db
```

## Documentation

- **[SDK Reference Guide](docs/README.md)**: Complete SDK documentation with examples

## Author

**Adam Shedivy**
📧 ajshedivyaj@gmail.com
🔗 [GitHub](https://github.com/ajshedivy)
