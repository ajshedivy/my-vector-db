# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A lightweight, production-ready vector database with a RESTful API and Python SDK. Built with FastAPI and Pydantic, supporting CRUD operations, vector similarity search, metadata filtering, and persistence.

**Tech Stack:** Python 3.13+, FastAPI, Pydantic, NumPy, scikit-learn, uvicorn

## Development Commands

### Running the Server

```bash
# Start API server with auto-reload (development)
uv run uvicorn my_vector_db.main:app --reload

# Or use Docker Compose
docker compose up -d

# Or directly with Docker
docker run -d --name my-vector-db -p 8000:8000 ghcr.io/ajshedivy/my-vector-db:latest
```

The server runs on `http://localhost:8000` with interactive docs at `/docs`.

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=my_vector_db --cov-report=html

# Run specific test file
uv run pytest tests/test_sdk.py

# Run specific test with verbose output
uv run pytest tests/test_sdk.py::test_name -vv
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Auto-fix linting issues
uv run ruff check --fix

# Type checking
uv run mypy src
```

### Development Setup

```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Install only production dependencies
uv sync
```

## Architecture

The project follows **Domain-Driven Design** with clean layered architecture:

```
my_vector_db/
├── api/           # FastAPI routes and request/response schemas
├── services/      # Business logic (LibraryService, DocumentService, SearchService)
├── domain/        # Core Pydantic models (Library, Document, Chunk)
├── indexes/       # Vector index implementations (base, flat, hnsw, ivf)
├── filters/       # Metadata filtering system (evaluator)
├── storage.py     # Thread-safe in-memory storage with RLock
├── sdk/           # Python client SDK
├── db/            # Agno framework integration
├── mcp/           # MCP server
└── main.py        # FastAPI application entry point
```

### Key Architecture Principles

1. **Layered Architecture**: API → Service → Storage → Index
   - API layer validates requests/responses
   - Service layer contains business logic
   - Storage layer provides thread-safe CRUD with RLock
   - Index layer implements vector search algorithms

2. **Data Hierarchy**: `Library → Document → Chunk`
   - Libraries contain documents and have vector indexes
   - Documents group related chunks
   - Chunks are searchable units with text and embeddings

3. **Thread Safety**: All storage operations use `threading.RLock`
   - Reentrant lock allows nested operations
   - Read/write synchronization prevents data races
   - Safe for concurrent access with multiple Uvicorn workers

4. **Index Management**:
   - Each library has its own vector index (FLAT, HNSW, or IVF)
   - Indexes are created lazily on first access
   - `LibraryService` manages index lifecycle in `_indexes` dict
   - Dirty index tracking (`_dirty_indexes`) for rebuild management

5. **Filtering Strategy**: Post-filtering after vector search
   - First: kNN search returns top candidates from index
   - Then: Apply metadata filters to refine results
   - Supports declarative filters and custom Python functions

## Vector Indexes

The project implements multiple index types:

- **FLAT Index** (`indexes/flat.py`): Brute-force exact search, O(n·d) time, 100% recall
  - Best for: Small datasets (<10k vectors), when accuracy is critical
  - Metrics: cosine, euclidean, dot_product

- **IVF Index** (`indexes/ivf.py`): Inverted file index with K-means clustering
  - Best for: Medium to large datasets with faster search needs
  - Uses scikit-learn for clustering
  - Configurable: `n_lists` (clusters), `n_probe` (clusters to search)

- **HNSW Index** (`indexes/hnsw.py`): Hierarchical navigable small world (planned)
  - Best for: Large datasets (>10k vectors)
  - O(log n) approximate search

All indexes inherit from `VectorIndex` base class in `indexes/base.py`.

## Important Implementation Details

### Storage Persistence

- Optional JSON-based snapshot persistence
- Configured via environment variables:
  - `ENABLE_STORAGE_PERSISTENCE=true/false`
  - `STORAGE_DIR=./data`
  - `STORAGE_SAVE_EVERY=N` (N=-1 disables auto-save)
- Atomic writes prevent corruption
- Restore on startup if snapshot exists

### Service Layer Responsibilities

- **LibraryService** (`services/library_service.py`):
  - Creates libraries and manages their vector indexes
  - Index factory pattern: creates FLAT/HNSW/IVF based on library config
  - Tracks dirty indexes that need rebuilding

- **DocumentService** (`services/document_service.py`):
  - Manages documents and chunks
  - Coordinates with LibraryService for index updates

- **SearchService** (`services/search_service.py`):
  - Performs kNN search via library indexes
  - Applies post-filtering using `FilterEvaluator`
  - Returns results with similarity scores

### API Design

The API uses a simplified flat structure:
- Resources accessible by UUID: `/documents/{id}`, `/chunks/{id}`
- Hierarchical creation: `/libraries/{id}/documents`, `/documents/{id}/chunks`
- Search at library level: `/libraries/{id}/query`

### SDK Client

- **VectorDBClient** (`sdk/client.py`): Main entry point
- Type-safe Pydantic models for all responses
- Automatic error handling with `@handle_errors` decorator
- Supports declarative metadata filters and custom filter functions

## Testing Conventions

- Tests use `pytest` with fixtures defined in `tests/conftest.py`
- Test files mirror source structure: `test_<module>.py`
- Integration tests in `test_api.py` and `test_sdk.py`
- Thread safety tests in `test_thread_safety.py`
- Coverage target: >80%

## Release Process

This project uses semantic versioning with `bump-my-version`:

```bash
# Bump version (creates commit and tag)
uv run bump-my-version bump patch  # Bug fixes
uv run bump-my-version bump minor  # New features
uv run bump-my-version bump major  # Breaking changes

# Push version tag
git push --follow-tags

# Trigger release workflow manually on GitHub Actions:
# "Publish Python Package and Docker Image" for full release
# "Build and Push Docker Image" for Docker-only update
```

See `DEVELOPER.md` for detailed release workflow.

## Environment Variables

```bash
# API server
VECTOR_DB_HOST=0.0.0.0
VECTOR_DB_PORT=8000

# Persistence (optional)
ENABLE_STORAGE_PERSISTENCE=false
STORAGE_DIR=./data
STORAGE_SAVE_EVERY=-1  # -1 disables automatic saves

# SDK client
VECTOR_DB_BASE_URL=http://localhost:8000
VECTOR_DB_TIMEOUT=30
```

## Integration Points

- **Agno Framework**: Adapter in `db/my_vector_db.py` implements `VectorDb` interface for RAG applications
- **MCP Server**: Exposed via `mcp/server.py` for tool-based interactions
- **CLI**: Optional interactive CLI available via `pip install "my-vector-db[cli]"`

## Common Patterns

When implementing new vector indexes:
1. Inherit from `VectorIndex` in `indexes/base.py`
2. Implement required methods: `add`, `bulk_add`, `search`, `update`, `delete`, `clear`
3. Add to `LibraryService._create_index()` factory method
4. Update `IndexType` enum in `domain/models.py`

When adding new API endpoints:
1. Define request/response schemas in `api/schemas.py`
2. Add route handler in `api/routes.py`
3. Implement business logic in appropriate service class
4. Add corresponding SDK method in `sdk/client.py`
5. Write tests in `tests/test_api.py` and `tests/test_sdk.py`

## Documentation

- **SDK Reference**: Complete documentation in `docs/README.md`
- **Examples**: Working examples in `examples/` directory
- **Architecture Diagrams**: Mermaid diagrams in `docs/architecture-diagram.md`
- **API Docs**: Interactive Swagger UI at `http://localhost:8000/docs`
