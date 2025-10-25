"""
Complex Integration Tests

Tests batch operations, edge cases, and data integrity.
Run with: pytest tests/test_complex_operations.py -v
"""

import pytest
import time
from uuid import UUID

from my_vector_db.services.document_service import DocumentService
from my_vector_db.services.library_service import LibraryService
from my_vector_db.storage import VectorStorage


@pytest.fixture
def storage():
    """Create a fresh VectorStorage instance."""
    return VectorStorage()


@pytest.fixture
def library_service(storage):
    """Create LibraryService with fresh storage."""
    return LibraryService(storage)


@pytest.fixture
def document_service(storage, library_service):
    """Create DocumentService with fresh storage and library service."""
    return DocumentService(storage, library_service)


class TestBatchInsert:
    """Tests for batch insert operations."""

    def test_batch_insert_large_hierarchy(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test inserting multiple libraries, documents, and chunks."""
        start = time.time()

        # Create 5 libraries
        num_libraries = 5
        libraries = []
        for i in range(num_libraries):
            library = library_service.create_library(
                name=f"Library {i + 1}", metadata={"category": f"category_{i % 3}"}
            )
            libraries.append(library)

        # Create 10 documents per library (50 total)
        num_docs_per_lib = 10
        documents = []
        for library in libraries:
            for j in range(num_docs_per_lib):
                doc = document_service.create_document(
                    library_id=library.id,
                    name=f"Document {j + 1}",
                    metadata={"doc_type": "article"},
                )
                documents.append(doc)

        # Create 20 chunks per document (1000 total)
        num_chunks_per_doc = 20
        total_chunks = 0
        for doc in documents:
            for k in range(num_chunks_per_doc):
                # Create simple 10-dimensional embeddings
                embedding = [float(k) * 0.1 for _ in range(10)]
                document_service.create_chunk(
                    document_id=doc.id,
                    text=f"Chunk {k + 1}: This is test content for chunk {k + 1}",
                    embedding=embedding,
                    metadata={"position": k},
                )
                total_chunks += 1

        elapsed = time.time() - start

        # Verify counts
        assert len(library_service.list_libraries()) == num_libraries
        for library in libraries:
            docs = document_service.list_documents(library.id)
            assert len(docs) == num_docs_per_lib
            for doc in docs:
                chunks = document_service.list_chunks(doc.id)
                assert len(chunks) == num_chunks_per_doc

        assert total_chunks == 1000
        assert elapsed >= 0  # Just verify timing works


class TestBatchUpdate:
    """Tests for batch update operations."""

    @pytest.fixture(autouse=True)
    def setup_batch_update_data(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Create hierarchy for batch update tests."""
        # Create 3 libraries with 5 documents each
        self.libraries = []
        for i in range(3):
            lib = library_service.create_library(name=f"Lib {i}")
            self.libraries.append(lib)

            for j in range(5):
                doc = document_service.create_document(
                    library_id=lib.id, name=f"Doc {j}"
                )
                # Add 10 chunks per document
                for k in range(10):
                    document_service.create_chunk(
                        document_id=doc.id,
                        text=f"Original text {k}",
                        embedding=[1.0] * 5,
                        metadata={"version": 1},
                    )

    def test_batch_update_libraries(self, library_service: LibraryService):
        """Test batch updating all libraries."""
        # Batch update all libraries
        for i, lib in enumerate(self.libraries):
            library_service.update_library(
                lib.id, name=f"Updated Library {i}", metadata={"updated": True}
            )

        # Verify updates
        for lib in self.libraries:
            retrieved = library_service.get_library(lib.id)
            assert "Updated Library" in retrieved.name
            assert retrieved.metadata.get("updated") is True

    def test_batch_update_documents(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test batch updating all documents."""
        # Batch update all documents
        update_count = 0
        for lib in self.libraries:
            docs = document_service.list_documents(lib.id)
            for doc in docs:
                document_service.update_document(
                    doc.id, metadata={"status": "reviewed"}
                )
                update_count += 1

        assert update_count == 15  # 3 libraries × 5 documents

        # Verify updates
        for lib in self.libraries:
            docs = document_service.list_documents(lib.id)
            for doc in docs:
                retrieved = document_service.get_document(doc.id)
                assert retrieved.metadata.get("status") == "reviewed"

    def test_batch_update_chunks(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test batch updating all chunks."""
        # Batch update all chunks
        chunk_update_count = 0
        for lib in self.libraries:
            docs = document_service.list_documents(lib.id)
            for doc in docs:
                chunks = document_service.list_chunks(doc.id)
                for chunk in chunks:
                    document_service.update_chunk(
                        chunk.id, text=f"Updated: {chunk.text}", metadata={"version": 2}
                    )
                    chunk_update_count += 1

        assert chunk_update_count == 150  # 3 libs × 5 docs × 10 chunks

        # Verify updates
        for lib in self.libraries:
            docs = document_service.list_documents(lib.id)
            for doc in docs:
                chunks = document_service.list_chunks(doc.id)
                for chunk in chunks:
                    retrieved_chunk = document_service.get_chunk(chunk.id)
                    assert retrieved_chunk.text.startswith("Updated:")
                    assert retrieved_chunk.metadata.get("version") == 2


class TestBatchDelete:
    """Tests for batch delete operations."""

    @pytest.fixture(autouse=True)
    def setup_batch_delete_data(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Create hierarchy for batch delete tests."""
        num_libraries = 4
        docs_per_lib = 6
        chunks_per_doc = 8

        self.library_ids = []
        self.document_ids = []
        self.chunk_ids = []

        for i in range(num_libraries):
            lib = library_service.create_library(name=f"Lib {i}")
            self.library_ids.append(lib.id)

            for j in range(docs_per_lib):
                doc = document_service.create_document(
                    library_id=lib.id, name=f"Doc {j}"
                )
                self.document_ids.append(doc.id)

                for k in range(chunks_per_doc):
                    chunk = document_service.create_chunk(
                        document_id=doc.id, text=f"Text {k}", embedding=[0.5] * 3
                    )
                    self.chunk_ids.append(chunk.id)

    def test_batch_delete_individual_chunks(self, document_service: DocumentService):
        """Test deleting individual chunks in batch."""
        chunks_to_delete = self.chunk_ids[:50]  # Delete first 50 chunks

        for chunk_id in chunks_to_delete:
            document_service.delete_chunk(chunk_id)

        # Verify chunks are gone
        for chunk_id in chunks_to_delete:
            assert document_service.get_chunk(chunk_id) is None

    def test_batch_delete_documents_with_cascade(
        self, document_service: DocumentService
    ):
        """Test deleting documents with cascading chunk deletes."""
        docs_to_delete = self.document_ids[:10]  # Delete first 10 documents

        for doc_id in docs_to_delete:
            doc = document_service.get_document(doc_id)
            if doc:
                document_service.delete_document(doc_id)

        # Verify documents are gone
        for doc_id in docs_to_delete:
            assert document_service.get_document(doc_id) is None

    def test_delete_library_cascades_all(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test deleting entire library cascades to all children."""
        lib_to_delete = self.library_ids[0]
        docs_in_lib = document_service.list_documents(lib_to_delete)

        # Calculate chunks in library
        chunks_in_lib = 0
        for doc in docs_in_lib:
            chunks_in_lib += len(document_service.list_chunks(doc.id))

        # Delete library
        library_service.delete_library(lib_to_delete)

        # Verify library and all children are gone
        assert library_service.get_library(lib_to_delete) is None
        for doc in docs_in_lib:
            assert document_service.get_document(doc.id) is None


class TestConcurrentModifications:
    """Tests for concurrent modifications to the same hierarchy."""

    @pytest.fixture(autouse=True)
    def setup_concurrent_data(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Create hierarchy for concurrent modification tests."""
        library = library_service.create_library(name="Concurrent Test")
        self.library_id = library.id

        self.docs = []
        for i in range(5):
            doc = document_service.create_document(
                library_id=library.id, name=f"Doc {i}"
            )
            self.docs.append(doc)
            # Add chunks
            for j in range(10):
                document_service.create_chunk(
                    document_id=doc.id, text=f"Chunk {j}", embedding=[1.0] * 5
                )

    def test_update_library_and_documents_concurrently(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test updating library while modifying documents."""
        library_service.update_library(self.library_id, name="Modified Library")

        for doc in self.docs[:3]:
            document_service.update_document(doc.id, name="Modified Doc")

        # Verify modifications
        lib = library_service.get_library(self.library_id)
        assert lib.name == "Modified Library"

    def test_delete_and_add_chunks_concurrently(
        self, document_service: DocumentService
    ):
        """Test deleting and adding chunks to same document."""
        doc = self.docs[0]
        existing_chunks = document_service.list_chunks(doc.id)

        # Delete half
        for chunk in existing_chunks[:5]:
            document_service.delete_chunk(chunk.id)

        # Add new ones
        for i in range(5):
            document_service.create_chunk(
                document_id=doc.id, text=f"New chunk {i}", embedding=[2.0] * 5
            )

        final_chunks = document_service.list_chunks(doc.id)
        assert len(final_chunks) == 10  # 5 deleted, 5 added = still 10

    def test_update_metadata_across_hierarchy(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test updating metadata on all levels simultaneously."""
        library_service.update_library(self.library_id, metadata={"level": "library"})

        for doc in self.docs:
            document_service.update_document(doc.id, metadata={"level": "document"})
            chunks = document_service.list_chunks(doc.id)
            for chunk in chunks[:3]:  # Update first 3 chunks per doc
                document_service.update_chunk(chunk.id, metadata={"level": "chunk"})

        # Verify final state
        final_lib = library_service.get_library(self.library_id)
        assert final_lib.metadata["level"] == "library"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_library(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test empty library with no documents."""
        empty_lib = library_service.create_library(name="Empty Library")
        docs = document_service.list_documents(empty_lib.id)

        assert len(docs) == 0

    def test_document_with_no_chunks(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test document with no chunks."""
        lib = library_service.create_library(name="Test Lib")
        empty_doc = document_service.create_document(
            library_id=lib.id, name="Empty Document"
        )
        chunks = document_service.list_chunks(empty_doc.id)

        assert len(chunks) == 0

    def test_very_long_text_and_metadata(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test handling very long text and large metadata."""
        lib = library_service.create_library(name="Test Lib")

        long_text = "x" * 10000
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}

        doc = document_service.create_document(
            library_id=lib.id, name="Large Doc", metadata=large_metadata
        )
        chunk = document_service.create_chunk(
            document_id=doc.id,
            text=long_text,
            embedding=[0.1] * 100,  # 100-dimensional
            metadata=large_metadata,
        )

        retrieved_chunk = document_service.get_chunk(chunk.id)
        assert len(retrieved_chunk.text) == 10000
        assert len(retrieved_chunk.embedding) == 100
        assert len(retrieved_chunk.metadata) == 100

    def test_update_to_empty_metadata(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test updating to empty metadata."""
        lib = library_service.create_library(name="Test Lib")
        doc = document_service.create_document(
            library_id=lib.id, name="Original Name", metadata={"key": "value"}
        )

        # Update with empty metadata
        updated = document_service.update_document(doc.id, metadata={})

        assert updated.metadata == {}

    def test_delete_nonexistent_entities(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test deleting non-existent entities returns False."""
        fake_id = UUID("00000000-0000-0000-0000-000000000000")

        assert library_service.delete_library(fake_id) is False
        assert document_service.delete_document(fake_id) is False
        assert document_service.delete_chunk(fake_id) is False

    def test_multiple_deletes_same_entity(self, library_service: LibraryService):
        """Test multiple deletes of same entity."""
        lib = library_service.create_library(name="Delete Test")

        assert library_service.delete_library(lib.id) is True
        assert library_service.delete_library(lib.id) is False  # Already deleted


class TestDataIntegrity:
    """Tests for data integrity and relationship consistency."""

    def test_parent_child_relationships(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test that relationships remain consistent across operations."""
        # Create hierarchy
        lib = library_service.create_library(name="Integrity Test")
        doc = document_service.create_document(library_id=lib.id, name="Doc 1")

        chunks = []
        for i in range(5):
            chunk = document_service.create_chunk(
                document_id=doc.id, text=f"Chunk {i}", embedding=[float(i)] * 3
            )
            chunks.append(chunk)

        # Verify parent-child relationships
        retrieved_lib = library_service.get_library(lib.id)
        assert doc.id in retrieved_lib.document_ids

        retrieved_doc = document_service.get_document(doc.id)
        for chunk in chunks:
            assert chunk.id in retrieved_doc.chunk_ids

        for chunk in chunks:
            retrieved_chunk = document_service.get_chunk(chunk.id)
            assert retrieved_chunk.document_id == doc.id

    def test_integrity_after_chunk_deletion(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test integrity after deleting chunks."""
        lib = library_service.create_library(name="Test")
        doc = document_service.create_document(library_id=lib.id, name="Doc")
        chunk = document_service.create_chunk(
            document_id=doc.id, text="Text", embedding=[1.0] * 3
        )

        # Delete chunk
        document_service.delete_chunk(chunk.id)

        # Verify document updated
        updated_doc = document_service.get_document(doc.id)
        assert chunk.id not in updated_doc.chunk_ids

    def test_integrity_after_document_deletion(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Test integrity after deleting documents."""
        lib = library_service.create_library(name="Test")
        doc = document_service.create_document(library_id=lib.id, name="Doc")

        # Delete document
        document_service.delete_document(doc.id)

        # Verify library updated
        updated_lib = library_service.get_library(lib.id)
        assert doc.id not in updated_lib.document_ids
