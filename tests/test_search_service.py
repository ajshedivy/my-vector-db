"""
SearchService Tests

Tests the k-NN search functionality via SearchService.
Run with: pytest tests/test_search_service.py -v
"""

import pytest
from uuid import uuid4

from my_vector_db.domain.models import IndexType
from my_vector_db.services.document_service import DocumentService
from my_vector_db.services.library_service import LibraryService
from my_vector_db.services.search_service import SearchService
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


@pytest.fixture
def search_service(storage, library_service):
    """Create SearchService with dependencies."""
    return SearchService(storage, library_service)


class TestBasicSearch:
    """Tests for basic k-NN search functionality."""

    @pytest.fixture(autouse=True)
    def setup_search_data(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Create library with test data for search."""
        # Create library with cosine similarity
        library = library_service.create_library(
            name="Search Test Library",
            index_type=IndexType.FLAT,
            index_config={"metric": "cosine"},
        )

        # Create document
        document = document_service.create_document(
            library_id=library.id, name="Test Document"
        )

        # Add chunks with known embeddings
        self.chunk1 = document_service.create_chunk(
            document_id=document.id, text="Chunk about X", embedding=[1.0, 0.0, 0.0]
        )

        self.chunk2 = document_service.create_chunk(
            document_id=document.id, text="Chunk about Y", embedding=[0.0, 1.0, 0.0]
        )

        self.chunk3 = document_service.create_chunk(
            document_id=document.id, text="Chunk about XY", embedding=[0.7, 0.7, 0.0]
        )

        self.library_id = library.id

        # Build index
        library_service.build_index(library.id)

    def test_basic_search_returns_results(self, search_service: SearchService):
        """Test that basic search returns correct results."""
        # Query with [1, 0, 0] - should match chunk1 best
        query = [1.0, 0.0, 0.0]
        results, query_time = search_service.search(
            library_id=self.library_id, query_embedding=query, k=3
        )

        assert len(results) == 3
        assert results[0][0].id == self.chunk1.id  # Best match
        assert results[0][1] == pytest.approx(1.0)  # Perfect cosine similarity
        assert query_time >= 0  # Query time should be non-negative

    def test_search_results_sorted_by_score(self, search_service: SearchService):
        """Test that search results are sorted by similarity score."""
        query = [1.0, 0.0, 0.0]
        results, _ = search_service.search(
            library_id=self.library_id, query_embedding=query, k=3
        )

        # Extract scores and verify they're descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestSearchKValues:
    """Tests for search with different k values."""

    @pytest.fixture(autouse=True)
    def setup_k_test_data(
        self, library_service: LibraryService, document_service: DocumentService
    ):
        """Create library with 10 chunks for k-value testing."""
        library = library_service.create_library(name="K Test Library")
        document = document_service.create_document(library_id=library.id, name="Doc")

        # Add 10 chunks
        for i in range(10):
            document_service.create_chunk(
                document_id=document.id, text=f"Chunk {i}", embedding=[float(i)] * 5
            )

        self.library_id = library.id
        library_service.build_index(library.id)

    def test_search_k_equals_1(self, search_service: SearchService):
        """Test search with k=1 returns exactly 1 result."""
        results, _ = search_service.search(
            library_id=self.library_id, query_embedding=[5.0] * 5, k=1
        )

        assert len(results) == 1

    def test_search_k_equals_5(self, search_service: SearchService):
        """Test search with k=5 returns exactly 5 results."""
        results, _ = search_service.search(
            library_id=self.library_id, query_embedding=[5.0] * 5, k=5
        )

        assert len(results) == 5

    def test_search_k_greater_than_chunks(self, search_service: SearchService):
        """Test search with k > num_chunks returns all chunks."""
        results, _ = search_service.search(
            library_id=self.library_id, query_embedding=[5.0] * 5, k=20
        )

        assert len(results) == 10  # Only 10 chunks exist


class TestSearchMetrics:
    """Tests for search with different distance metrics."""

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "dot_product"])
    def test_search_with_metric(
        self,
        metric: str,
        library_service: LibraryService,
        document_service: DocumentService,
        search_service: SearchService,
    ):
        """Test search works with each distance metric."""
        library = library_service.create_library(
            name=f"{metric} Library", index_config={"metric": metric}
        )

        document = document_service.create_document(library_id=library.id, name="Doc")

        # Add test chunks
        for i in range(3):
            document_service.create_chunk(
                document_id=document.id,
                text=f"Chunk {i}",
                embedding=[float(i), float(i), 0.0],
            )

        library_service.build_index(library.id)

        # Perform search
        results, query_time = search_service.search(
            library_id=library.id, query_embedding=[1.0, 1.0, 0.0], k=3
        )

        assert len(results) == 3
        assert query_time >= 0


class TestSearchErrorHandling:
    """Tests for search error handling."""

    def test_search_empty_library(
        self, library_service: LibraryService, search_service: SearchService
    ):
        """Test search on empty library raises ValueError."""
        library = library_service.create_library(name="Empty Library")

        with pytest.raises(ValueError, match="no chunks"):
            search_service.search(
                library_id=library.id, query_embedding=[1.0, 2.0, 3.0], k=5
            )

    def test_search_invalid_library(self, search_service: SearchService):
        """Test search on non-existent library raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            search_service.search(
                library_id=uuid4(), query_embedding=[1.0, 2.0, 3.0], k=5
            )


class TestSearchPerformance:
    """Tests for search performance with larger datasets."""

    def test_search_with_1000_chunks(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        search_service: SearchService,
    ):
        """Test search performance with 1000 chunks."""
        # Create library
        library = library_service.create_library(
            name="Performance Test", index_config={"metric": "cosine"}
        )

        # Add 100 documents with 10 chunks each = 1000 total chunks
        dimension = 128
        num_docs = 100
        chunks_per_doc = 10

        for i in range(num_docs):
            doc = document_service.create_document(
                library_id=library.id, name=f"Doc {i}"
            )

            for j in range(chunks_per_doc):
                # Create varied embeddings
                embedding = [float((i * chunks_per_doc + j) % 100) / 100.0] * dimension
                document_service.create_chunk(
                    document_id=doc.id,
                    text=f"Chunk {j} from doc {i}",
                    embedding=embedding,
                )

        total_chunks = num_docs * chunks_per_doc

        # Build index
        library_service.build_index(library.id)

        # Perform search
        query = [0.5] * dimension
        results, query_time = search_service.search(
            library_id=library.id, query_embedding=query, k=10
        )

        assert len(results) == 10
        assert query_time >= 0

        # Verify results are sorted by score (descending)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


class TestAutoIndexBuilding:
    """Tests for automatic index building."""

    def test_search_auto_builds_index(
        self,
        library_service: LibraryService,
        document_service: DocumentService,
        search_service: SearchService,
    ):
        """Test that search auto-builds index if not already built."""
        # Create library and chunks WITHOUT building index
        library = library_service.create_library(name="Auto Build Library")
        document = document_service.create_document(library_id=library.id, name="Doc")

        document_service.create_chunk(
            document_id=document.id, text="Chunk 1", embedding=[1.0, 0.0, 0.0]
        )

        # Search WITHOUT calling build_index first
        # SearchService should auto-build via get_index()
        results, query_time = search_service.search(
            library_id=library.id, query_embedding=[1.0, 0.0, 0.0], k=5
        )

        assert len(results) == 1

        # Verify index is now cached
        index = library_service.get_index(library.id)
        assert len(index._vectors) == 1
