"""
Thread Safety Tests

Tests concurrent access from multiple threads to verify RLock implementation.
Run with: pytest tests/test_thread_safety.py -v
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from my_vector_db.domain.models import Library, Document, Chunk
from my_vector_db.storage import VectorStorage


@pytest.fixture
def storage():
    """Create a fresh VectorStorage instance."""
    return VectorStorage()


class TestRLockReentrancy:
    """Tests for RLock reentrant behavior."""

    def test_cascading_delete_with_reentrancy(self, storage: VectorStorage):
        """Test that RLock allows reentrant calls during cascading deletes."""
        # Create hierarchy
        library = Library(name="Test Library")
        storage.create_library(library)

        document = Document(name="Test Doc", library_id=library.id)
        storage.create_document(document)

        chunk = Chunk(text="Test", embedding=[1.0, 2.0], document_id=document.id)
        storage.create_chunk(chunk)

        # Delete library - this tests reentrant locking
        # delete_library → delete_document → delete_chunk
        # All acquire the same RLock
        result = storage.delete_library(library.id)

        assert result is True
        assert storage.get_library(library.id) is None
        assert storage.get_document(document.id) is None
        assert storage.get_chunk(chunk.id) is None


class TestConcurrentReads:
    """Tests for concurrent read operations."""

    @pytest.fixture(autouse=True)
    def setup_read_data(self, storage: VectorStorage):
        """Create test data for concurrent reads."""
        self.num_libraries = 100
        self.library_ids = []

        for i in range(self.num_libraries):
            lib = Library(name=f"Library {i}")
            storage.create_library(lib)
            self.library_ids.append(lib.id)

    def test_concurrent_reads_no_corruption(self, storage: VectorStorage):
        """Test that multiple threads can safely read data concurrently."""
        num_threads = 50
        read_count = [0]  # Use list for mutability
        errors = []

        def read_libraries():
            """Read libraries repeatedly."""
            try:
                for _ in range(20):  # Each thread does 20 reads
                    for lib_id in self.library_ids[:10]:
                        lib = storage.get_library(lib_id)
                        assert lib is not None
                        read_count[0] += 1
            except Exception as e:
                errors.append(str(e))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(read_libraries) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0


class TestConcurrentWrites:
    """Tests for concurrent write operations."""

    def test_concurrent_writes_no_data_loss(self, storage: VectorStorage):
        """Test that concurrent writes are serialized and don't corrupt data."""
        # Create shared library
        library = Library(name="Shared Library")
        storage.create_library(library)

        num_threads = 20
        docs_per_thread = 50
        errors = []
        created_docs = []
        lock = threading.Lock()

        def create_documents(thread_id):
            """Each thread creates multiple documents."""
            try:
                thread_docs = []
                for i in range(docs_per_thread):
                    doc = Document(
                        name=f"Thread-{thread_id}-Doc-{i}", library_id=library.id
                    )
                    storage.create_document(doc)
                    thread_docs.append(doc.id)

                with lock:
                    created_docs.extend(thread_docs)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_documents, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        expected_docs = num_threads * docs_per_thread

        assert len(errors) == 0
        assert len(created_docs) == expected_docs

        # Verify all documents exist
        retrieved_docs = storage.list_documents_by_library(library.id)
        assert len(retrieved_docs) == expected_docs

        # Verify no duplicates
        assert len(set(created_docs)) == expected_docs


class TestMixedOperations:
    """Tests for mixed concurrent operations."""

    @pytest.fixture(autouse=True)
    def setup_mixed_data(self, storage: VectorStorage):
        """Create initial data for mixed operations."""
        library = Library(name="Mixed Ops Library")
        storage.create_library(library)
        self.library_id = library.id

        # Create 50 documents with 10 chunks each
        self.initial_docs = []
        for i in range(50):
            doc = Document(name=f"Doc {i}", library_id=library.id)
            storage.create_document(doc)
            self.initial_docs.append(doc)

            for j in range(10):
                chunk = Chunk(
                    text=f"Chunk {j}", embedding=[float(j)] * 5, document_id=doc.id
                )
                storage.create_chunk(chunk)

    def test_concurrent_mixed_operations(self, storage: VectorStorage):
        """Test mixed reads, writes, updates, and deletes from multiple threads."""
        errors = []
        stats = {"reads": 0, "creates": 0, "updates": 0, "deletes": 0}
        stats_lock = threading.Lock()

        def reader_thread(thread_id):
            """Continuously read documents and chunks."""
            try:
                for _ in range(100):
                    for doc in self.initial_docs[:10]:
                        retrieved = storage.get_document(doc.id)
                        if retrieved:
                            chunks = storage.list_chunks_by_document(doc.id)
                            with stats_lock:
                                stats["reads"] += 1
            except Exception as e:
                errors.append(f"Reader {thread_id}: {e}")

        def writer_thread(thread_id):
            """Create new documents and chunks."""
            try:
                for i in range(20):
                    doc = Document(
                        name=f"Writer-{thread_id}-Doc-{i}", library_id=self.library_id
                    )
                    storage.create_document(doc)

                    for j in range(5):
                        chunk = Chunk(
                            text=f"New chunk {j}",
                            embedding=[1.0] * 5,
                            document_id=doc.id,
                        )
                        storage.create_chunk(chunk)

                    with stats_lock:
                        stats["creates"] += 1
            except Exception as e:
                errors.append(f"Writer {thread_id}: {e}")

        def updater_thread(thread_id):
            """Update existing chunks."""
            try:
                for _ in range(50):
                    for doc in self.initial_docs[thread_id::5]:
                        chunks = storage.list_chunks_by_document(doc.id)
                        for chunk in chunks[:2]:
                            updated = storage.get_chunk(chunk.id)
                            if updated:
                                updated.text = f"Updated by thread {thread_id}"
                                storage.update_chunk(chunk.id, updated)
                                with stats_lock:
                                    stats["updates"] += 1
            except Exception as e:
                errors.append(f"Updater {thread_id}: {e}")

        def deleter_thread(thread_id):
            """Delete some chunks."""
            try:
                for _ in range(30):
                    for doc in self.initial_docs[thread_id * 10 : (thread_id + 1) * 10]:
                        chunks = storage.list_chunks_by_document(doc.id)
                        if chunks:
                            deleted = storage.delete_chunk(chunks[0].id)
                            if deleted:
                                with stats_lock:
                                    stats["deletes"] += 1
            except Exception as e:
                errors.append(f"Deleter {thread_id}: {e}")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []

            # Launch different types of threads
            for i in range(5):
                futures.append(executor.submit(reader_thread, i))
            for i in range(5):
                futures.append(executor.submit(writer_thread, i))
            for i in range(5):
                futures.append(executor.submit(updater_thread, i))
            for i in range(5):
                futures.append(executor.submit(deleter_thread, i))

            # Wait for all
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0
        assert sum(stats.values()) > 0  # Some operations completed


class TestCascadingDeleteThreadSafety:
    """Tests for thread-safe cascading deletes."""

    def test_concurrent_cascading_deletes(self, storage: VectorStorage):
        """Test that cascading deletes are thread-safe when called concurrently."""
        num_libraries = 10
        libraries = []

        for i in range(num_libraries):
            lib = Library(name=f"Lib {i}")
            storage.create_library(lib)
            libraries.append(lib)

            # Each library has 10 documents
            for j in range(10):
                doc = Document(name=f"Doc {j}", library_id=lib.id)
                storage.create_document(doc)

                # Each document has 5 chunks
                for k in range(5):
                    chunk = Chunk(
                        text=f"Chunk {k}", embedding=[1.0] * 3, document_id=doc.id
                    )
                    storage.create_chunk(chunk)

        errors = []

        def delete_library(lib):
            """Delete a library (cascades to all children)."""
            try:
                result = storage.delete_library(lib.id)
                assert result is True
            except Exception as e:
                errors.append(f"Delete {lib.id}: {e}")

        with ThreadPoolExecutor(max_workers=num_libraries) as executor:
            futures = [executor.submit(delete_library, lib) for lib in libraries]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0

        # Verify everything is deleted
        remaining_libraries = storage.list_libraries()
        assert len(remaining_libraries) == 0


class TestStressTest:
    """Stress tests with high concurrency."""

    def test_high_concurrency_stress(self, storage: VectorStorage):
        """Stress test with high concurrent load."""
        library = Library(name="Stress Test Library")
        storage.create_library(library)

        num_threads = 100
        ops_per_thread = 50
        errors = []

        def stress_worker(thread_id):
            """Perform mixed operations rapidly."""
            try:
                for i in range(ops_per_thread):
                    # Create document
                    doc = Document(name=f"T{thread_id}-D{i}", library_id=library.id)
                    storage.create_document(doc)

                    # Create chunk
                    chunk = Chunk(
                        text=f"Content {i}",
                        embedding=[float(i % 10)] * 3,
                        document_id=doc.id,
                    )
                    storage.create_chunk(chunk)

                    # Read it back
                    retrieved = storage.get_chunk(chunk.id)
                    assert retrieved is not None

                    # Update it
                    retrieved.text = f"Updated {i}"
                    storage.update_chunk(chunk.id, retrieved)

                    # Delete it
                    storage.delete_chunk(chunk.id)

            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0


class TestPerformanceBenchmarks:
    """Performance benchmarks for concurrent operations."""

    def test_concurrent_read_throughput(self, storage: VectorStorage):
        """Benchmark concurrent read throughput."""
        # Setup
        num_libraries = 100
        library_ids = []
        for i in range(num_libraries):
            lib = Library(name=f"Lib {i}")
            storage.create_library(lib)
            library_ids.append(lib.id)

        # Benchmark
        num_threads = 50
        reads_per_thread = 100
        start = time.time()

        def read_worker():
            for _ in range(reads_per_thread):
                for lib_id in library_ids[:10]:
                    storage.get_library(lib_id)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(read_worker) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        elapsed = time.time() - start
        total_reads = num_threads * reads_per_thread * 10
        throughput = total_reads / elapsed

        # Just verify it completes without error
        assert throughput > 0

    def test_concurrent_write_throughput(self, storage: VectorStorage):
        """Benchmark concurrent write throughput."""
        library = Library(name="Write Test")
        storage.create_library(library)

        num_threads = 20
        writes_per_thread = 50
        start = time.time()

        def write_worker(thread_id):
            for i in range(writes_per_thread):
                doc = Document(name=f"T{thread_id}-D{i}", library_id=library.id)
                storage.create_document(doc)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        elapsed = time.time() - start
        total_writes = num_threads * writes_per_thread
        throughput = total_writes / elapsed

        # Verify all writes completed
        docs = storage.list_documents_by_library(library.id)
        assert len(docs) == total_writes
        assert throughput > 0
