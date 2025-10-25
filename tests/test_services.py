"""
Service Layer Tests

Tests the LibraryService and DocumentService directly (without HTTP layer).
Run with: pytest tests/test_services.py -v
"""

import pytest
from uuid import UUID

from my_vector_db.domain.models import IndexType
from my_vector_db.services.document_service import DocumentService
from my_vector_db.services.library_service import LibraryService
from my_vector_db.storage import VectorStorage


@pytest.fixture
def storage():
    """Create a fresh VectorStorage instance for each test."""
    return VectorStorage()


@pytest.fixture
def library_service(storage):
    """Create a LibraryService with fresh storage."""
    return LibraryService(storage)


@pytest.fixture
def document_service(storage, library_service):
    """Create a DocumentService with fresh storage and library service."""
    return DocumentService(storage, library_service)


class TestLibraryCRUD:
    """Tests for library CRUD operations via LibraryService."""

    def test_create_library(self, library_service: LibraryService):
        """Test creating a new library."""
        library = library_service.create_library(
            name="Test Library",
            metadata={"description": "A test library"},
            index_type=IndexType.FLAT,
        )

        assert library.name == "Test Library"
        assert library.index_type == IndexType.FLAT
        assert library.metadata["description"] == "A test library"
        assert library.id is not None

    def test_get_library(self, library_service: LibraryService):
        """Test retrieving a library by ID."""
        # Create library
        library = library_service.create_library(name="Get Test Library")

        # Retrieve library
        retrieved = library_service.get_library(library.id)

        assert retrieved is not None
        assert retrieved.id == library.id
        assert retrieved.name == "Get Test Library"

    def test_get_nonexistent_library(self, library_service: LibraryService):
        """Test retrieving a non-existent library returns None."""
        invalid_id = UUID("00000000-0000-0000-0000-000000000000")
        result = library_service.get_library(invalid_id)

        assert result is None

    def test_list_libraries(self, library_service: LibraryService):
        """Test listing all libraries."""
        # Create multiple libraries
        library_service.create_library(name="Library 1")
        library_service.create_library(name="Library 2")
        library_service.create_library(name="Library 3")

        # List libraries
        libraries = library_service.list_libraries()

        assert len(libraries) == 3
        library_names = [lib.name for lib in libraries]
        assert "Library 1" in library_names
        assert "Library 2" in library_names
        assert "Library 3" in library_names

    def test_update_library(self, library_service: LibraryService):
        """Test updating a library."""
        # Create library
        library = library_service.create_library(
            name="Original Name", index_type=IndexType.FLAT
        )

        # Update library
        updated = library_service.update_library(
            library.id, name="Updated Library", index_type=IndexType.HNSW
        )

        assert updated.name == "Updated Library"
        assert updated.index_type == IndexType.HNSW
        assert updated.id == library.id

    def test_delete_library(self, library_service: LibraryService):
        """Test deleting a library."""
        # Create library
        library = library_service.create_library(name="Delete Test Library")

        # Delete library
        deleted = library_service.delete_library(library.id)

        assert deleted is True
        assert library_service.get_library(library.id) is None


class TestDocumentCRUD:
    """Tests for document CRUD operations via DocumentService."""

    @pytest.fixture(autouse=True)
    def setup_library(self, library_service: LibraryService):
        """Create a library for document tests."""
        self.library = library_service.create_library(name="Doc Test Library")
        self.library_id = self.library.id

    def test_create_document(self, document_service: DocumentService):
        """Test creating a new document."""
        document = document_service.create_document(
            library_id=self.library_id,
            name="Test Document",
            metadata={"author": "Test Author"},
        )

        assert document.name == "Test Document"
        assert document.library_id == self.library_id
        assert document.metadata["author"] == "Test Author"
        assert document.id is not None

    def test_create_document_invalid_library(self, document_service: DocumentService):
        """Test creating a document with non-existent library raises KeyError."""
        invalid_id = UUID("00000000-0000-0000-0000-000000000000")

        with pytest.raises(KeyError):
            document_service.create_document(
                library_id=invalid_id, name="Invalid Document"
            )

    def test_get_document(self, document_service: DocumentService):
        """Test retrieving a document by ID."""
        # Create document
        document = document_service.create_document(
            library_id=self.library_id, name="Get Test Document"
        )

        # Retrieve document
        retrieved = document_service.get_document(document.id)

        assert retrieved is not None
        assert retrieved.id == document.id
        assert retrieved.name == "Get Test Document"

    def test_get_nonexistent_document(self, document_service: DocumentService):
        """Test retrieving a non-existent document returns None."""
        invalid_id = UUID("00000000-0000-0000-0000-000000000000")
        result = document_service.get_document(invalid_id)

        assert result is None

    def test_list_documents(self, document_service: DocumentService):
        """Test listing documents in a library."""
        # Create documents
        document_service.create_document(library_id=self.library_id, name="Doc 1")
        document_service.create_document(library_id=self.library_id, name="Doc 2")
        document_service.create_document(library_id=self.library_id, name="Doc 3")

        # List documents
        documents = document_service.list_documents(self.library_id)

        assert len(documents) == 3
        doc_names = [doc.name for doc in documents]
        assert "Doc 1" in doc_names
        assert "Doc 2" in doc_names

    def test_update_document(self, document_service: DocumentService):
        """Test updating a document."""
        # Create document
        document = document_service.create_document(
            library_id=self.library_id, name="Original Document"
        )

        # Update document
        updated = document_service.update_document(
            document.id, name="Updated Document", metadata={"author": "New Author"}
        )

        assert updated.name == "Updated Document"
        assert updated.metadata["author"] == "New Author"
        assert updated.id == document.id

    def test_delete_document(self, document_service: DocumentService):
        """Test deleting a document."""
        # Create document
        document = document_service.create_document(
            library_id=self.library_id, name="Delete Test Document"
        )

        # Delete document
        deleted = document_service.delete_document(document.id)

        assert deleted is True
        assert document_service.get_document(document.id) is None


class TestChunkCRUD:
    """Tests for chunk CRUD operations via DocumentService."""

    @pytest.fixture(autouse=True)
    def setup_document(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Create library and document for chunk tests."""
        library = library_service.create_library(name="Chunk Test Library")
        document = document_service.create_document(
            library_id=library.id, name="Chunk Test Document"
        )
        self.library_id = library.id
        self.document_id = document.id

    def test_create_chunk(self, document_service: DocumentService):
        """Test creating a new chunk."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk = document_service.create_chunk(
            document_id=self.document_id,
            text="This is a test chunk",
            embedding=embedding,
            metadata={"page": 1},
        )

        assert chunk.text == "This is a test chunk"
        assert len(chunk.embedding) == 5
        assert chunk.embedding == embedding
        assert chunk.metadata["page"] == 1
        assert chunk.id is not None

    def test_create_chunk_invalid_document(self, document_service: DocumentService):
        """Test creating a chunk with non-existent document raises KeyError."""
        invalid_id = UUID("00000000-0000-0000-0000-000000000000")

        with pytest.raises(KeyError):
            document_service.create_chunk(
                document_id=invalid_id, text="Invalid chunk", embedding=[1.0, 2.0]
            )

    def test_get_chunk(self, document_service: DocumentService):
        """Test retrieving a chunk by ID."""
        # Create chunk
        chunk = document_service.create_chunk(
            document_id=self.document_id,
            text="Get test chunk",
            embedding=[1.0, 2.0, 3.0],
        )

        # Retrieve chunk
        retrieved = document_service.get_chunk(chunk.id)

        assert retrieved is not None
        assert retrieved.id == chunk.id
        assert retrieved.text == "Get test chunk"

    def test_get_nonexistent_chunk(self, document_service: DocumentService):
        """Test retrieving a non-existent chunk returns None."""
        invalid_id = UUID("00000000-0000-0000-0000-000000000000")
        result = document_service.get_chunk(invalid_id)

        assert result is None

    def test_list_chunks(self, document_service: DocumentService):
        """Test listing chunks in a document."""
        # Create chunks
        for i in range(3):
            document_service.create_chunk(
                document_id=self.document_id,
                text=f"Chunk {i}",
                embedding=[float(i)] * 3,
            )

        # List chunks
        chunks = document_service.list_chunks(self.document_id)

        assert len(chunks) == 3

    def test_update_chunk(self, document_service: DocumentService):
        """Test updating a chunk."""
        # Create chunk
        chunk = document_service.create_chunk(
            document_id=self.document_id,
            text="Original text",
            embedding=[1.0, 2.0, 3.0],
        )

        # Update chunk
        new_embedding = [0.6, 0.7, 0.8, 0.9, 1.0]
        updated = document_service.update_chunk(
            chunk.id,
            text="Updated chunk text",
            embedding=new_embedding,
            metadata={"page": 2},
        )

        assert updated.text == "Updated chunk text"
        assert updated.embedding == new_embedding
        assert updated.metadata["page"] == 2
        assert updated.id == chunk.id

    def test_delete_chunk(self, document_service: DocumentService):
        """Test deleting a chunk."""
        # Create chunk
        chunk = document_service.create_chunk(
            document_id=self.document_id,
            text="Delete test chunk",
            embedding=[1.0, 2.0, 3.0],
        )

        # Delete chunk
        deleted = document_service.delete_chunk(chunk.id)

        assert deleted is True
        assert document_service.get_chunk(chunk.id) is None


class TestIndexBuilding:
    """Tests for vector index building and management."""

    def test_build_flat_index(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test building a FlatIndex with multiple documents and chunks."""
        # Create library with FlatIndex
        library = library_service.create_library(
            name="Index Test Library", index_type=IndexType.FLAT
        )

        # Add documents and chunks with embeddings
        num_docs = 3
        chunks_per_doc = 5
        dimension = 5

        for i in range(num_docs):
            doc = document_service.create_document(
                library_id=library.id, name=f"Document {i}"
            )

            for j in range(chunks_per_doc):
                # Create simple embeddings [i.j, i.j, i.j, i.j, i.j]
                embedding = [float(i) + j / 10] * dimension
                document_service.create_chunk(
                    document_id=doc.id,
                    text=f"Chunk {j} from doc {i}",
                    embedding=embedding,
                )

        total_chunks = num_docs * chunks_per_doc

        # Build index
        library_service.build_index(library.id)
        index = library_service.get_index(library.id)

        assert index is not None
        assert index.dimension == dimension
        assert len(index._vectors) == total_chunks

    def test_index_auto_build(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that get_index auto-builds if index doesn't exist."""
        library = library_service.create_library(
            name="Auto-build Test", index_type=IndexType.FLAT
        )

        doc = document_service.create_document(library_id=library.id, name="Doc")
        document_service.create_chunk(
            document_id=doc.id, text="Test chunk", embedding=[1.0, 2.0, 3.0, 4.0, 5.0]
        )

        # Call get_index without build_index - should auto-build
        index = library_service.get_index(library.id)

        assert index is not None
        assert len(index._vectors) == 1

    def test_build_index_empty_library(self, library_service: LibraryService):
        """Test building index on empty library raises ValueError."""
        library = library_service.create_library(name="Empty Library")

        with pytest.raises(ValueError, match="no chunks"):
            library_service.build_index(library.id)

    def test_build_index_inconsistent_dimensions(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test building index with inconsistent dimensions raises ValueError."""
        library = library_service.create_library(name="Bad Dimensions")
        doc = document_service.create_document(library_id=library.id, name="Bad Doc")

        # Add chunks with different dimensions
        document_service.create_chunk(
            document_id=doc.id, text="Chunk 1", embedding=[1.0, 2.0, 3.0]
        )
        document_service.create_chunk(
            document_id=doc.id,
            text="Chunk 2",
            embedding=[1.0, 2.0],  # Different dimension!
        )

        with pytest.raises(ValueError, match="(?i)(inconsistent|dimension)"):
            library_service.build_index(library.id)

    def test_index_rebuilding(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that index can be rebuilt after adding more chunks."""
        library = library_service.create_library(
            name="Rebuild Test", index_type=IndexType.FLAT
        )

        # Add initial chunks
        doc1 = document_service.create_document(library_id=library.id, name="Doc 1")
        for i in range(3):
            document_service.create_chunk(
                document_id=doc1.id, text=f"Chunk {i}", embedding=[float(i)] * 5
            )

        # Build index
        library_service.build_index(library.id)
        index = library_service.get_index(library.id)
        original_count = len(index._vectors)

        assert original_count == 3

        # Add more chunks
        doc2 = document_service.create_document(library_id=library.id, name="Doc 2")
        document_service.create_chunk(
            document_id=doc2.id, text="New chunk", embedding=[9.9] * 5
        )

        # Rebuild index
        library_service.build_index(library.id)
        rebuilt_index = library_service.get_index(library.id)

        assert len(rebuilt_index._vectors) == original_count + 1


class TestCascadingDeletes:
    """Tests for cascading delete behavior."""

    def test_delete_library_cascades_to_documents_and_chunks(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that deleting a library deletes all documents and chunks."""
        # Create hierarchy: library -> document -> chunk
        library = library_service.create_library(name="Cascade Test Library")
        document = document_service.create_document(
            library_id=library.id, name="Cascade Test Document"
        )
        chunk = document_service.create_chunk(
            document_id=document.id,
            text="Cascade test chunk",
            embedding=[1.0, 2.0, 3.0],
        )

        library_id = library.id
        document_id = document.id
        chunk_id = chunk.id

        # Verify all exist
        assert library_service.get_library(library_id) is not None
        assert document_service.get_document(document_id) is not None
        assert document_service.get_chunk(chunk_id) is not None

        # Delete library (should cascade)
        library_service.delete_library(library_id)

        # Verify all are deleted
        assert library_service.get_library(library_id) is None
        assert document_service.get_document(document_id) is None
        assert document_service.get_chunk(chunk_id) is None

    def test_delete_document_cascades_to_chunks(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that deleting a document deletes all its chunks."""
        # Create hierarchy
        library = library_service.create_library(name="Doc Cascade Library")
        document = document_service.create_document(
            library_id=library.id, name="Doc Cascade Document"
        )

        # Add multiple chunks
        chunk_ids = []
        for i in range(3):
            chunk = document_service.create_chunk(
                document_id=document.id, text=f"Chunk {i}", embedding=[float(i)] * 3
            )
            chunk_ids.append(chunk.id)

        document_id = document.id

        # Verify all exist
        for chunk_id in chunk_ids:
            assert document_service.get_chunk(chunk_id) is not None

        # Delete document (should cascade to chunks)
        document_service.delete_document(document_id)

        # Verify document and all chunks are deleted
        assert document_service.get_document(document_id) is None
        for chunk_id in chunk_ids:
            assert document_service.get_chunk(chunk_id) is None


class TestDirtyCacheInvalidation:
    """Tests for automatic index invalidation when data changes."""

    def test_create_chunk_invalidates_index(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that creating a chunk invalidates the index."""
        # Create library and document
        library = library_service.create_library(name="Test Library")
        document = document_service.create_document(
            library_id=library.id, name="Test Document"
        )

        # Add first chunk and build index
        chunk1 = document_service.create_chunk(
            document_id=document.id, text="First chunk", embedding=[1.0, 0.0, 0.0]
        )
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 1

        # Add second chunk - should mark index as dirty
        chunk2 = document_service.create_chunk(
            document_id=document.id, text="Second chunk", embedding=[0.0, 1.0, 0.0]
        )

        # Get index again - should rebuild automatically
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 2

    def test_update_chunk_invalidates_index(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that updating a chunk's embedding invalidates the index."""
        import numpy as np

        # Create library and document with chunk
        library = library_service.create_library(name="Test Library")
        document = document_service.create_document(
            library_id=library.id, name="Test Document"
        )
        chunk = document_service.create_chunk(
            document_id=document.id, text="Test chunk", embedding=[1.0, 0.0, 0.0]
        )

        # Build index
        index = library_service.get_index(library.id)
        original_vector = index._vectors[chunk.id]
        assert np.allclose(original_vector, [1.0, 0.0, 0.0])

        # Update embedding - should mark index as dirty
        document_service.update_chunk(chunk_id=chunk.id, embedding=[0.0, 1.0, 0.0])

        # Get index again - should rebuild with new embedding
        index = library_service.get_index(library.id)
        updated_vector = index._vectors[chunk.id]
        assert np.allclose(updated_vector, [0.0, 1.0, 0.0])

    def test_update_chunk_text_only_does_not_invalidate(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that updating only text doesn't invalidate the index."""
        # Create library and document with chunk
        library = library_service.create_library(name="Test Library")
        document = document_service.create_document(
            library_id=library.id, name="Test Document"
        )
        chunk = document_service.create_chunk(
            document_id=document.id, text="Original text", embedding=[1.0, 0.0, 0.0]
        )

        # Build index
        library_service.get_index(library.id)

        # Verify index is not in dirty set
        assert library.id not in library_service._dirty_indexes

        # Update text only (not embedding)
        document_service.update_chunk(chunk_id=chunk.id, text="Updated text")

        # Index should still not be dirty
        assert library.id not in library_service._dirty_indexes

    def test_delete_chunk_invalidates_index(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that deleting a chunk invalidates the index."""
        # Create library and document with multiple chunks
        library = library_service.create_library(name="Test Library")
        document = document_service.create_document(
            library_id=library.id, name="Test Document"
        )
        chunk1 = document_service.create_chunk(
            document_id=document.id, text="First chunk", embedding=[1.0, 0.0, 0.0]
        )
        chunk2 = document_service.create_chunk(
            document_id=document.id, text="Second chunk", embedding=[0.0, 1.0, 0.0]
        )

        # Build index
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 2

        # Delete first chunk - should mark index as dirty
        document_service.delete_chunk(chunk1.id)

        # Get index again - should rebuild with only one chunk
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 1
        assert chunk2.id in index._vectors
        assert chunk1.id not in index._vectors

    def test_delete_document_invalidates_index(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that deleting a document (which cascades to chunks) invalidates the index."""
        # Create library with two documents
        library = library_service.create_library(name="Test Library")
        doc1 = document_service.create_document(
            library_id=library.id, name="Document 1"
        )
        doc2 = document_service.create_document(
            library_id=library.id, name="Document 2"
        )

        # Add chunks to both documents
        chunk1 = document_service.create_chunk(
            document_id=doc1.id, text="Chunk from doc 1", embedding=[1.0, 0.0, 0.0]
        )
        chunk2 = document_service.create_chunk(
            document_id=doc2.id, text="Chunk from doc 2", embedding=[0.0, 1.0, 0.0]
        )

        # Build index
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 2

        # Delete first document - should mark index as dirty
        document_service.delete_document(doc1.id)

        # Get index again - should rebuild with only doc2's chunk
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 1
        assert chunk2.id in index._vectors
        assert chunk1.id not in index._vectors

    def test_batch_insert_then_query(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test the real-world scenario: batch insert, query, add more, query again."""
        # Create library and document
        library = library_service.create_library(name="Test Library")
        document = document_service.create_document(
            library_id=library.id, name="Test Document"
        )

        # Batch insert initial chunks
        chunk1 = document_service.create_chunk(
            document_id=document.id,
            text="Python programming",
            embedding=[1.0, 0.0, 0.0],
        )
        chunk2 = document_service.create_chunk(
            document_id=document.id,
            text="JavaScript programming",
            embedding=[0.9, 0.1, 0.0],
        )

        # First query - builds index automatically
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 2

        # Add more chunks after first query
        chunk3 = document_service.create_chunk(
            document_id=document.id, text="Rust programming", embedding=[0.8, 0.2, 0.0]
        )

        # Next query - should rebuild automatically and include new chunk
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 3
        assert chunk1.id in index._vectors
        assert chunk2.id in index._vectors
        assert chunk3.id in index._vectors
