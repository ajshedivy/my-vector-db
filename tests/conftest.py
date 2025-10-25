"""
Pytest fixtures and configuration.

This module provides reusable fixtures for testing.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from my_vector_db.main import app


@pytest.fixture
def client() -> TestClient:
    """
    Create a FastAPI test client.

    Returns:
        TestClient for making API requests in tests

    TODO: Optionally configure test-specific app settings
    """
    return TestClient(app)


@pytest.fixture
def sample_embedding() -> list[float]:
    """
    Create a sample embedding vector for testing.

    Returns:
        A dummy embedding vector

    TODO: Implement - return a list of floats (e.g., 384 dimensions)
    """
    # Example: return [0.1, 0.2, 0.3, ..., ]
    raise NotImplementedError("Create sample embedding")


@pytest.fixture
def sample_library_data() -> dict[str, Any]:
    """
    Create sample library creation data.

    Returns:
        Dictionary with library creation data

    TODO: Implement - return dict matching CreateLibraryRequest
    """
    raise NotImplementedError("Create sample library data")


@pytest.fixture
def sample_document_data() -> dict[str, Any]:
    """
    Create sample document creation data.

    Returns:
        Dictionary with document creation data

    TODO: Implement - return dict matching CreateDocumentRequest
    """
    raise NotImplementedError("Create sample document data")


@pytest.fixture
def sample_chunk_data(sample_embedding: list[float]) -> dict[str, Any]:
    """
    Create sample chunk creation data.

    Args:
        sample_embedding: Embedding fixture

    Returns:
        Dictionary with chunk creation data

    TODO: Implement - return dict matching CreateChunkRequest
    """
    raise NotImplementedError("Create sample chunk data")


# TODO: Add more fixtures as needed:
# - Fixture to create a library and return its ID
# - Fixture to create a library with documents
# - Fixture to reset storage between tests
# - Fixtures for different index types
