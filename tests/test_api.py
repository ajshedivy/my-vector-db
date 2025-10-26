"""
FastAPI Integration Tests

Tests the complete API endpoints using FastAPI's TestClient with pytest.
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient):
        """Test the health check endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "vector-db"
        assert "storage" in data
        assert "libraries" in data["storage"]
        assert "documents" in data["storage"]
        assert "chunks" in data["storage"]


class TestLibraryCRUD:
    """Tests for library CRUD operations."""

    def test_create_library(self, client: TestClient):
        """Test creating a new library."""
        response = client.post(
            "/libraries",
            json={
                "name": "Test Library",
                "metadata": {"description": "API test library"},
                "index_type": "flat",
                "index_config": {"metric": "cosine"},
            },
        )

        assert response.status_code == 201
        library = response.json()

        assert "id" in library
        assert library["name"] == "Test Library"
        assert library["index_type"] == "flat"
        assert library["metadata"]["description"] == "API test library"

        # Cleanup
        client.delete(f"/libraries/{library['id']}")

    def test_get_library(self, client: TestClient):
        """Test retrieving a library by ID."""
        # Create library
        create_response = client.post("/libraries", json={"name": "Get Test Library"})
        library_id = create_response.json()["id"]

        # Get library
        response = client.get(f"/libraries/{library_id}")

        assert response.status_code == 200
        library = response.json()
        assert library["id"] == library_id
        assert library["name"] == "Get Test Library"

        # Cleanup
        client.delete(f"/libraries/{library_id}")

    def test_list_libraries(self, client: TestClient):
        """Test listing all libraries."""
        # Create multiple libraries
        lib1 = client.post("/libraries", json={"name": "Library 1"}).json()
        lib2 = client.post("/libraries", json={"name": "Library 2"}).json()

        # List libraries
        response = client.get("/libraries")

        assert response.status_code == 200
        libraries = response.json()
        assert len(libraries) >= 2

        # Verify our libraries are in the list
        library_ids = [lib["id"] for lib in libraries]
        assert lib1["id"] in library_ids
        assert lib2["id"] in library_ids

        # Cleanup
        client.delete(f"/libraries/{lib1['id']}")
        client.delete(f"/libraries/{lib2['id']}")

    def test_update_library(self, client: TestClient):
        """Test updating a library."""
        # Create library
        create_response = client.post("/libraries", json={"name": "Original Name"})
        library_id = create_response.json()["id"]

        # Update library
        response = client.put(
            f"/libraries/{library_id}",
            json={"name": "Updated Name", "metadata": {"new_field": "value"}},
        )

        assert response.status_code == 200
        updated = response.json()
        assert updated["name"] == "Updated Name"
        assert updated["metadata"]["new_field"] == "value"

        # Cleanup
        client.delete(f"/libraries/{library_id}")

    def test_delete_library(self, client: TestClient):
        """Test deleting a library."""
        # Create library
        create_response = client.post(
            "/libraries", json={"name": "Delete Test Library"}
        )
        library_id = create_response.json()["id"]

        # Delete library
        response = client.delete(f"/libraries/{library_id}")
        assert response.status_code == 204

        # Verify deleted
        get_response = client.get(f"/libraries/{library_id}")
        assert get_response.status_code == 404


class TestDocumentCRUD:
    """Tests for document CRUD operations."""

    @pytest.fixture(autouse=True)
    def setup_library(self, client: TestClient):
        """Create a library for document tests."""
        response = client.post("/libraries", json={"name": "Doc Test Library"})
        self.library_id = response.json()["id"]
        yield
        # Cleanup library (cascades to documents)
        client.delete(f"/libraries/{self.library_id}")

    def test_create_document(self, client: TestClient):
        """Test creating a new document."""
        response = client.post(
            f"/libraries/{self.library_id}/documents",
            json={"name": "Test Document", "metadata": {"author": "Test Author"}},
        )

        assert response.status_code == 201
        document = response.json()

        assert "id" in document
        assert document["name"] == "Test Document"
        assert document["library_id"] == self.library_id
        assert document["metadata"]["author"] == "Test Author"

    def test_get_document(self, client: TestClient):
        """Test retrieving a document by ID."""
        # Create document
        create_response = client.post(
            f"/libraries/{self.library_id}/documents",
            json={"name": "Get Test Document"},
        )
        document_id = create_response.json()["id"]

        # Get document
        response = client.get(f"/documents/{document_id}")

        assert response.status_code == 200
        document = response.json()
        assert document["id"] == document_id

    def test_list_documents(self, client: TestClient):
        """Test listing documents in a library."""
        # Create documents
        client.post(f"/libraries/{self.library_id}/documents", json={"name": "Doc 1"})
        client.post(f"/libraries/{self.library_id}/documents", json={"name": "Doc 2"})

        # List documents
        response = client.get(f"/libraries/{self.library_id}/documents")

        assert response.status_code == 200
        documents = response.json()
        assert len(documents) >= 2

    def test_update_document(self, client: TestClient):
        """Test updating a document."""
        # Create document
        create_response = client.post(
            f"/libraries/{self.library_id}/documents", json={"name": "Original"}
        )
        document_id = create_response.json()["id"]

        # Update document
        response = client.put(
            f"/documents/{document_id}",
            json={"name": "Updated", "metadata": {"key": "value"}},
        )

        assert response.status_code == 200
        updated = response.json()
        assert updated["name"] == "Updated"
        assert updated["metadata"]["key"] == "value"

    def test_delete_document(self, client: TestClient):
        """Test deleting a document."""
        # Create document
        create_response = client.post(
            f"/libraries/{self.library_id}/documents", json={"name": "Delete Test"}
        )
        document_id = create_response.json()["id"]

        # Delete document
        response = client.delete(f"/documents/{document_id}")
        assert response.status_code == 204

        # Verify deleted
        get_response = client.get(f"/documents/{document_id}")
        assert get_response.status_code == 404


class TestChunkCRUD:
    """Tests for chunk CRUD operations."""

    @pytest.fixture(autouse=True)
    def setup_document(self, client: TestClient):
        """Create library and document for chunk tests."""
        lib_response = client.post("/libraries", json={"name": "Chunk Test Library"})
        self.library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/libraries/{self.library_id}/documents",
            json={"name": "Chunk Test Document"},
        )
        self.document_id = doc_response.json()["id"]

        yield

        # Cleanup
        client.delete(f"/libraries/{self.library_id}")

    def test_create_chunk(self, client: TestClient):
        """Test creating a new chunk."""
        response = client.post(
            f"/documents/{self.document_id}/chunks",
            json={
                "text": "This is a test chunk",
                "embedding": [1.0, 2.0, 3.0, 4.0, 5.0],
                "metadata": {"page": 1},
            },
        )

        assert response.status_code == 201
        chunk = response.json()

        assert "id" in chunk
        assert chunk["text"] == "This is a test chunk"
        assert len(chunk["embedding"]) == 5
        assert chunk["metadata"]["page"] == 1

    def test_get_chunk(self, client: TestClient):
        """Test retrieving a chunk by ID."""
        # Create chunk
        create_response = client.post(
            f"/documents/{self.document_id}/chunks",
            json={"text": "Get test chunk", "embedding": [1.0, 2.0, 3.0]},
        )
        chunk_id = create_response.json()["id"]

        # Get chunk
        response = client.get(f"/chunks/{chunk_id}")

        assert response.status_code == 200
        chunk = response.json()
        assert chunk["id"] == chunk_id

    def test_list_chunks(self, client: TestClient):
        """Test listing chunks in a document."""
        # Create chunks
        for i in range(3):
            client.post(
                f"/documents/{self.document_id}/chunks",
                json={"text": f"Chunk {i}", "embedding": [float(i)] * 3},
            )

        # List chunks
        response = client.get(f"/documents/{self.document_id}/chunks")

        assert response.status_code == 200
        chunks = response.json()
        assert len(chunks) == 3

    def test_update_chunk(self, client: TestClient):
        """Test updating a chunk."""
        # Create chunk
        create_response = client.post(
            f"/documents/{self.document_id}/chunks",
            json={"text": "Original", "embedding": [1.0, 2.0, 3.0]},
        )
        chunk_id = create_response.json()["id"]

        # Update chunk
        response = client.put(
            f"/chunks/{chunk_id}",
            json={"text": "Updated", "metadata": {"updated": True}},
        )

        assert response.status_code == 200
        updated = response.json()
        assert updated["text"] == "Updated"
        assert updated["metadata"]["updated"] is True

    def test_delete_chunk(self, client: TestClient):
        """Test deleting a chunk."""
        # Create chunk
        create_response = client.post(
            f"/documents/{self.document_id}/chunks",
            json={"text": "Delete test", "embedding": [1.0, 2.0, 3.0]},
        )
        chunk_id = create_response.json()["id"]

        # Delete chunk
        response = client.delete(f"/chunks/{chunk_id}")
        assert response.status_code == 204

        # Verify deleted
        get_response = client.get(f"/chunks/{chunk_id}")
        assert get_response.status_code == 404


class TestSearch:
    """Tests for k-NN search functionality."""

    @pytest.fixture(autouse=True)
    def setup_search_data(self, client: TestClient):
        """Create library with test vectors for search."""
        # Create library
        lib_response = client.post(
            "/libraries",
            json={
                "name": "Search Test Library",
                "index_type": "flat",
                "index_config": {"metric": "cosine"},
            },
        )
        self.library_id = lib_response.json()["id"]

        # Create document
        doc_response = client.post(
            f"/libraries/{self.library_id}/documents",
            json={"name": "Search Test Document"},
        )
        document_id = doc_response.json()["id"]

        # Add chunks with known embeddings
        chunk_data = [
            {"text": "Chunk about X", "embedding": [1.0, 0.0, 0.0]},
            {"text": "Chunk about Y", "embedding": [0.0, 1.0, 0.0]},
            {"text": "Chunk about Z", "embedding": [0.0, 0.0, 1.0]},
            {"text": "Chunk about XY", "embedding": [0.7, 0.7, 0.0]},
        ]

        for data in chunk_data:
            client.post(
                f"/documents/{document_id}/chunks",
                json=data,
            )

        yield

        # Cleanup
        client.delete(f"/libraries/{self.library_id}")

    def test_basic_search(self, client: TestClient):
        """Test basic k-NN search returns results."""
        response = client.post(
            f"/libraries/{self.library_id}/query",
            json={"embedding": [1.0, 0.0, 0.0], "k": 3},
        )

        assert response.status_code == 200
        result = response.json()

        assert "results" in result
        assert "total" in result
        assert "query_time_ms" in result
        assert len(result["results"]) == 3

    def test_search_ranking(self, client: TestClient):
        """Test that search results are ranked by similarity."""
        response = client.post(
            f"/libraries/{self.library_id}/query",
            json={
                "embedding": [1.0, 0.0, 0.0],  # Query for X-axis
                "k": 4,
            },
        )

        results = response.json()["results"]

        # First result should be "Chunk about X" (perfect match)
        assert results[0]["text"] == "Chunk about X"
        assert results[0]["score"] == pytest.approx(1.0)

        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_search_limit_k(self, client: TestClient):
        """Test that k parameter limits results."""
        for k in [1, 2, 3]:
            response = client.post(
                f"/libraries/{self.library_id}/query",
                json={"embedding": [1.0, 1.0, 1.0], "k": k},
            )

            results = response.json()["results"]
            assert len(results) == k


class TestErrorHandling:
    """Tests for API error handling."""

    def test_get_nonexistent_library(self, client: TestClient):
        """Test GET on non-existent library returns 404."""
        response = client.get("/libraries/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_nonexistent_library(self, client: TestClient):
        """Test DELETE on non-existent library returns 404."""
        response = client.delete("/libraries/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    def test_create_document_invalid_library(self, client: TestClient):
        """Test creating document in non-existent library returns 404."""
        response = client.post(
            "/libraries/00000000-0000-0000-0000-000000000000/documents",
            json={"name": "Test"},
        )
        assert response.status_code == 404

    def test_query_empty_library(self, client: TestClient):
        """Test querying library with no chunks returns 400."""
        # Create empty library
        lib_response = client.post("/libraries", json={"name": "Empty Library"})
        library_id = lib_response.json()["id"]

        # Try to query
        response = client.post(
            f"/libraries/{library_id}/query",
            json={"embedding": [1.0, 2.0, 3.0], "k": 5},
        )

        assert response.status_code == 400
        assert "no chunks" in response.json()["detail"].lower()

        # Cleanup
        client.delete(f"/libraries/{library_id}")


class TestCascadingDeletes:
    """Tests for cascading delete behavior."""

    def test_delete_library_cascades(self, client: TestClient):
        """Test that deleting library deletes all documents and chunks."""
        # Create hierarchy
        lib = client.post("/libraries", json={"name": "Cascade Library"}).json()
        doc = client.post(
            f"/libraries/{lib['id']}/documents", json={"name": "Cascade Document"}
        ).json()
        chunk = client.post(
            f"/documents/{doc['id']}/chunks",
            json={"text": "Cascade Chunk", "embedding": [1.0, 2.0, 3.0]},
        ).json()

        # Delete library
        delete_response = client.delete(f"/libraries/{lib['id']}")
        assert delete_response.status_code == 204

        # Verify all are deleted
        assert client.get(f"/libraries/{lib['id']}").status_code == 404
        assert client.get(f"/documents/{doc['id']}").status_code == 404
        assert client.get(f"/chunks/{chunk['id']}").status_code == 404
