"""
Tests for batch operations (storage, service, API, and SDK).
"""

import pytest
from uuid import uuid4

from my_vector_db.domain.models import Chunk, Document
from my_vector_db.services.document_service import DocumentService
from my_vector_db.services.library_service import LibraryService
from my_vector_db.storage import VectorStorage


class TestBatchStorage:
    """Test batch operations at the storage layer."""

    def test_create_chunks_batch_success(self):
        """Test creating multiple chunks in a single operation."""
        storage = VectorStorage()

        # Create library and document
        from my_vector_db.domain.models import Library

        library = Library(name="Test Library")
        storage.create_library(library)

        document = Document(name="Test Doc", library_id=library.id)
        storage.create_document(document)

        # Create chunks
        chunks = [
            Chunk(
                document_id=document.id,
                text=f"Chunk {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],
                metadata={"index": i},
            )
            for i in range(5)
        ]

        # Batch create
        created = storage.create_chunks_batch(chunks)

        assert len(created) == 5
        assert all(chunk.id in storage._chunks for chunk in created)
        assert len(document.chunk_ids) == 5

    def test_create_chunks_batch_validates_document_exists(self):
        """Test that batch create fails if document doesn't exist."""
        storage = VectorStorage()

        fake_doc_id = uuid4()
        chunks = [
            Chunk(
                document_id=fake_doc_id,
                text="Test",
                embedding=[0.1, 0.2],
                metadata={},
            )
        ]

        with pytest.raises(KeyError, match="Document with ID .* not found"):
            storage.create_chunks_batch(chunks)

    def test_create_chunks_batch_validates_no_duplicates(self):
        """Test that batch create fails if chunk ID already exists."""
        storage = VectorStorage()

        from my_vector_db.domain.models import Library

        library = Library(name="Test Library")
        storage.create_library(library)

        document = Document(name="Test Doc", library_id=library.id)
        storage.create_document(document)

        # Create a chunk
        chunk1 = Chunk(
            document_id=document.id,
            text="First",
            embedding=[0.1, 0.2],
            metadata={},
        )
        storage.create_chunk(chunk1)

        # Try to create it again in batch
        chunks = [
            chunk1,
            Chunk(
                document_id=document.id,
                text="Second",
                embedding=[0.3, 0.4],
                metadata={},
            ),
        ]

        with pytest.raises(ValueError, match="Chunk with ID .* already exists"):
            storage.create_chunks_batch(chunks)

    def test_create_chunks_batch_atomic(self):
        """Test that batch create is atomic - all or nothing."""
        storage = VectorStorage()

        from my_vector_db.domain.models import Library

        library = Library(name="Test Library")
        storage.create_library(library)

        document = Document(name="Test Doc", library_id=library.id)
        storage.create_document(document)

        # Create first chunk
        chunk1 = Chunk(
            document_id=document.id, text="First", embedding=[0.1], metadata={}
        )
        storage.create_chunk(chunk1)

        # Try to batch create with duplicate - should fail and not create any
        chunks = [
            Chunk(document_id=document.id, text="Second", embedding=[0.2], metadata={}),
            chunk1,  # Duplicate!
            Chunk(document_id=document.id, text="Third", embedding=[0.3], metadata={}),
        ]

        initial_count = len(storage._chunks)

        with pytest.raises(ValueError):
            storage.create_chunks_batch(chunks)

        # Verify nothing was created
        assert len(storage._chunks) == initial_count

    def test_create_documents_batch_success(self):
        """Test creating multiple documents in a single operation."""
        storage = VectorStorage()

        from my_vector_db.domain.models import Library

        library = Library(name="Test Library")
        storage.create_library(library)

        documents = [
            Document(name=f"Doc {i}", library_id=library.id, metadata={"index": i})
            for i in range(3)
        ]

        created = storage.create_documents_batch(documents)

        assert len(created) == 3
        assert all(doc.id in storage._documents for doc in created)
        assert len(library.document_ids) == 3


class TestBatchService:
    """Test batch operations at the service layer."""

    def test_create_chunks_batch_invalidates_index(self):
        """Test that batch chunk creation invalidates the library index."""
        storage = VectorStorage()
        library_service = LibraryService(storage)
        document_service = DocumentService(storage, library_service)

        # Create library and document
        library = library_service.create_library("Test Library", index_type="flat")
        document = document_service.create_document(library.id, "Test Doc")

        # Create some initial chunks to build index
        document_service.create_chunk(document.id, "Initial", [0.1, 0.2], {})

        # Build index
        library_service.build_index(library.id)
        index = library_service.get_index(library.id)
        assert index is not None
        assert library_service._indexes.get(library.id) is not None

        # Batch create chunks - should invalidate index
        chunks = [
            Chunk(
                document_id=document.id,
                text=f"Batch {i}",
                embedding=[0.1 * i, 0.2 * i],
                metadata={},
            )
            for i in range(3)
        ]
        document_service.create_chunks_batch(document.id, chunks)

        # Index should be marked as dirty (invalidated)
        assert library.id in library_service._dirty_indexes

    def test_create_chunks_batch_sets_document_id(self):
        """Test that batch create ensures all chunks have correct document_id."""
        storage = VectorStorage()
        library_service = LibraryService(storage)
        document_service = DocumentService(storage, library_service)

        library = library_service.create_library("Test Library")
        document = document_service.create_document(library.id, "Test Doc")

        # Create chunks with wrong/missing document_id
        wrong_doc_id = uuid4()
        chunks = [
            Chunk(
                document_id=wrong_doc_id, text="Test 1", embedding=[0.1], metadata={}
            ),
            Chunk(
                document_id=wrong_doc_id, text="Test 2", embedding=[0.2], metadata={}
            ),
        ]

        # Service should override document_id
        created = document_service.create_chunks_batch(document.id, chunks)

        assert all(chunk.document_id == document.id for chunk in created)

    def test_create_documents_batch(self):
        """Test batch document creation at service layer."""
        storage = VectorStorage()
        library_service = LibraryService(storage)
        document_service = DocumentService(storage, library_service)

        library = library_service.create_library("Test Library")

        documents = [
            Document(
                library_id=uuid4(), name=f"Doc {i}", metadata={}
            )  # Wrong library_id
            for i in range(3)
        ]

        # Service should override library_id
        created = document_service.create_documents_batch(library.id, documents)

        assert len(created) == 3
        assert all(doc.library_id == library.id for doc in created)


class TestBatchAPI:
    """Test batch operations via the API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient
        from my_vector_db.main import app

        return TestClient(app)

    @pytest.fixture
    def setup_library_and_doc(self, client):
        """Create a library and document for testing."""
        # Create library
        response = client.post(
            "/libraries",
            json={
                "name": "Test Library",
                "index_type": "flat",
                "metadata": {},
                "index_config": {},
            },
        )
        assert response.status_code == 201
        library = response.json()

        # Create document
        response = client.post(
            f"/libraries/{library['id']}/documents",
            json={"name": "Test Document", "metadata": {}},
        )
        assert response.status_code == 201
        document = response.json()

        return library, document

    def test_batch_create_chunks_success(self, client, setup_library_and_doc):
        """Test batch chunk creation via API."""
        library, document = setup_library_and_doc

        # Batch create chunks
        chunks_data = [
            {
                "text": f"Chunk {i}",
                "embedding": [0.1 * i, 0.2 * i, 0.3 * i],
                "metadata": {"index": i},
            }
            for i in range(5)
        ]

        response = client.post(
            f"/documents/{document['id']}/chunks/batch",
            json={"chunks": chunks_data},
        )

        assert response.status_code == 201
        result = response.json()
        assert result["total"] == 5
        assert len(result["chunks"]) == 5
        assert all("id" in chunk for chunk in result["chunks"])

    def test_batch_create_chunks_validates_document_exists(
        self, client, setup_library_and_doc
    ):
        """Test batch chunk creation fails if document doesn't exist."""
        library, _ = setup_library_and_doc

        fake_doc_id = str(uuid4())
        chunks_data = [{"text": "Test", "embedding": [0.1, 0.2], "metadata": {}}]

        response = client.post(
            f"/documents/{fake_doc_id}/chunks/batch",
            json={"chunks": chunks_data},
        )

        assert response.status_code == 404

    def test_batch_create_chunks_empty_list(self, client, setup_library_and_doc):
        """Test batch chunk creation with empty list."""
        library, document = setup_library_and_doc

        response = client.post(
            f"/documents/{document['id']}/chunks/batch",
            json={"chunks": []},
        )

        # Should fail validation (min_length=1)
        assert response.status_code == 422

    def test_batch_create_documents_success(self, client):
        """Test batch document creation via API."""
        # Create library
        response = client.post(
            "/libraries",
            json={
                "name": "Test Library",
                "index_type": "flat",
                "metadata": {},
                "index_config": {},
            },
        )
        assert response.status_code == 201
        library = response.json()

        # Batch create documents
        documents_data = [
            {"name": f"Doc {i}", "metadata": {"index": i}} for i in range(3)
        ]

        response = client.post(
            f"/libraries/{library['id']}/documents/batch",
            json={"documents": documents_data},
        )

        assert response.status_code == 201
        result = response.json()
        assert result["total"] == 3
        assert len(result["documents"]) == 3
        assert all("id" in doc for doc in result["documents"])


class TestBatchSDK:
    """Test batch operations via the SDK."""

    @pytest.fixture
    def test_client(self):
        """Create FastAPI test client."""
        from fastapi.testclient import TestClient
        from my_vector_db.main import app

        return TestClient(app)

    @pytest.fixture
    def client(self, test_client):
        """Create SDK client with test client base URL."""
        from my_vector_db.sdk import VectorDBClient

        # Use the test client's base URL
        sdk_client = VectorDBClient(base_url="http://testserver")
        # Replace the internal httpx client with the test client
        sdk_client._client = test_client
        return sdk_client

    def test_add_chunks_with_chunk_objects(self, client):
        """Test add_chunks with Chunk objects."""
        # Create library and document
        library = client.create_library(name="SDK Test Library")
        document = client.create_document(library_id=library.id, name="SDK Test Doc")

        # Create chunk objects
        chunks = [
            Chunk(
                document_id=document.id,
                text=f"SDK Chunk {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],
                metadata={"index": i},
            )
            for i in range(3)
        ]

        # Batch add
        created = client.add_chunks(
            chunks=chunks,
        )

        assert len(created) == 3
        assert all(isinstance(chunk, Chunk) for chunk in created)
        assert all(chunk.document_id == document.id for chunk in created)

        # Cleanup
        client.delete_library(library.id)

    def test_add_chunks_with_dicts(self, client):
        """Test add_chunks with dict objects."""
        # Create library and document
        library = client.create_library(name="SDK Test Library")
        document = client.create_document(library_id=library.id, name="SDK Test Doc")

        # Create chunk dicts
        chunk_dicts = [
            {
                "text": f"Dict Chunk {i}",
                "embedding": [0.1 * i, 0.2 * i],
                "metadata": {"index": i},
            }
            for i in range(3)
        ]

        # Batch add
        created = client.add_chunks(
            document_id=document.id,
            chunks=chunk_dicts,
        )

        assert len(created) == 3
        assert all(isinstance(chunk, Chunk) for chunk in created)

        # Cleanup
        client.delete_library(library.id)

    def test_add_chunks_validates_dict_format(self, client):
        """Test that add_chunks validates dict format."""
        library = client.create_library(name="SDK Test Library")
        document = client.create_document(library_id=library.id, name="SDK Test Doc")

        # Missing 'embedding' field
        invalid_dicts = [{"text": "Missing embedding", "metadata": {}}]

        with pytest.raises(ValueError, match="must have 'text' and 'embedding'"):
            client.add_chunks(
                document_id=document.id,
                chunks=invalid_dicts,
            )

        # Cleanup
        client.delete_library(library.id)

    def test_add_chunk_object_style(self, client):
        """Test add_chunk with Chunk object (new style)."""
        library = client.create_library(name="SDK Test Library")
        document = client.create_document(library_id=library.id, name="SDK Test Doc")

        chunk_obj = Chunk(
            document_id=document.id,
            text="Object style chunk",
            embedding=[0.1, 0.2, 0.3],
            metadata={"style": "object"},
        )

        created = client.add_chunk(
            document_id=document.id,
            chunk=chunk_obj,
        )

        assert created.text == "Object style chunk"
        assert created.metadata["style"] == "object"

        # Cleanup
        client.delete_library(library.id)

    def test_add_chunk_primitive_style(self, client):
        """Test add_chunk with primitives (old style)."""
        library = client.create_library(name="SDK Test Library")
        document = client.create_document(library_id=library.id, name="SDK Test Doc")

        created = client.add_chunk(
            document_id=document.id,
            text="Primitive style chunk",
            embedding=[0.1, 0.2, 0.3],
            metadata={"style": "primitive"},
        )

        assert created.text == "Primitive style chunk"
        assert created.metadata["style"] == "primitive"

        # Cleanup
        client.delete_library(library.id)

    def test_add_chunk_validates_input(self, client):
        """Test that add_chunk validates input properly."""
        library = client.create_library(name="SDK Test Library")
        document = client.create_document(library_id=library.id, name="SDK Test Doc")

        # Neither chunk nor text+embedding provided
        with pytest.raises(
            ValueError,
            match="Must provide either 'chunk' object OR both 'text' and 'embedding'",
        ):
            client.add_chunk(
                document_id=document.id,
            )

        # Only text provided (missing embedding)
        with pytest.raises(
            ValueError,
            match="Must provide either 'chunk' object OR both 'text' and 'embedding'",
        ):
            client.add_chunk(
                document_id=document.id,
                text="Missing embedding",
            )

        # Cleanup
        client.delete_library(library.id)

    def test_create_chunk_deprecation_warning(self, client):
        """Test that create_chunk shows deprecation warning."""
        library = client.create_library(name="SDK Test Library")
        document = client.create_document(library_id=library.id, name="SDK Test Doc")

        with pytest.warns(
            DeprecationWarning, match="create_chunk.*deprecated.*add_chunk"
        ):
            client.create_chunk(
                document_id=document.id,
                text="Deprecated method",
                embedding=[0.1, 0.2],
            )

        # Cleanup
        client.delete_library(library.id)
