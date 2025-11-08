# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-11-07

### Added
- IVF (Inverted File) index implementation for scalable vector search
- Support for IVF index configuration parameters: `nlist`, `nprobe`, `metric`
- Unit tests and end-to-end tests for IVF index functionality
- Documentation updates for IVF index usage and configuration

### Fixed
- CLI table formatting issues in help output
- SDK documentation typos

## [0.2.1] - 2025-11-06

### Fixed
- CLI extra now properly published to PyPI (moved from `[dependency-groups]` to `[project.optional-dependencies]`)

## [0.2.0] - 2025-11-06

### Added
- Interactive CLI for database management (optional install via `pip install my-vector-db[cli]`)

### Known Issues
- CLI extra not available on PyPI (fixed in v0.2.1)

### Changed

### Fixed

### Removed

## [0.1.0] - 2025-01-06

### Added

**Core Database Features**
- Vector database server with FastAPI REST API
- Support for FLAT and HNSW vector indexes
- Three-tier data model: Libraries → Documents → Chunks
- In-memory vector storage and search
- Metadata filtering with declarative and custom filter functions
- Optional persistence with snapshot save/restore functionality
- Docker and Docker Compose deployment support
- Multi-platform Docker images (linux/amd64, linux/arm64)

**Client SDK**
- Python SDK for programmatic database interaction
- Full CRUD operations for libraries, documents, and chunks
- Vector similarity search with k-NN
- Batch operations for efficient bulk inserts
- Server-side and client-side filtering capabilities
- Comprehensive error handling and retry logic
- Connection management and health checks

**Integrations**
- Agno AI framework integration for RAG applications
- MCP (Model Context Protocol) server implementation
- Support for multiple embedding providers (OpenAI, Cohere, Anthropic)

**Developer Tools**
- Comprehensive test suite with pytest
- Type checking with mypy
- Code formatting and linting with ruff
- Extensive documentation and examples

**CI/CD**
- GitHub Actions workflows for:
  - Publishing Python package to PyPI
  - Building and publishing Docker images to GitHub Container Registry
  - Docker-only rebuild workflow

### Notes

This is the initial release of my-vector-db, providing core vector database functionality with a Python client SDK. The database supports both FLAT and HNSW indexing strategies, with optional data persistence for production deployments.

[0.1.0]: https://github.com/ajshedivy/my-vector-db/releases/tag/v0.1.0
