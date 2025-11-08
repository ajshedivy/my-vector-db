"""
Unit tests for IVF (Inverted File) Index.

Tests cover:
- Index creation and initialization
- Config validation
- Vector add/update/delete operations
- Dimension validation
- Basic functionality before full clustering implementation
"""

import pytest
import numpy as np
from uuid import UUID, uuid4

from my_vector_db.indexes.ivf import IVFIndex


class TestIVFIndexCreation:
    """Test IVF index creation and initialization."""

    def test_create_index_default_config(self):
        """Can create IVF index with default configuration."""
        index = IVFIndex(dimension=384)

        assert index.dimension == 384
        assert index.config == {}
        assert not index._is_built
        assert len(index._vectors) == 0
        assert len(index._clusters) == 0

    def test_create_index_with_config(self):
        """Can create IVF index with custom configuration."""
        config = {"nlist": 100, "nprobe": 10, "metric": "cosine"}
        index = IVFIndex(dimension=384, config=config)

        assert index.dimension == 384
        assert index.config["nlist"] == 100
        assert index.config["nprobe"] == 10
        assert index.config["metric"] == "cosine"

    def test_config_validation_invalid_nlist(self):
        """Raises ValueError for invalid nlist."""
        with pytest.raises(ValueError, match="nlist must be a positive integer"):
            IVFIndex(dimension=384, config={"nlist": 0})

        with pytest.raises(ValueError, match="nlist must be a positive integer"):
            IVFIndex(dimension=384, config={"nlist": -10})

        with pytest.raises(ValueError, match="nlist must be a positive integer"):
            IVFIndex(dimension=384, config={"nlist": "invalid"})

    def test_config_validation_invalid_nprobe(self):
        """Raises ValueError for invalid nprobe."""
        with pytest.raises(ValueError, match="nprobe must be a positive integer"):
            IVFIndex(dimension=384, config={"nprobe": 0})

        with pytest.raises(ValueError, match="nprobe must be a positive integer"):
            IVFIndex(dimension=384, config={"nprobe": -5})

    def test_config_validation_invalid_metric(self):
        """Raises ValueError for invalid metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            IVFIndex(dimension=384, config={"metric": "invalid_metric"})


class TestIVFIndexAdd:
    """Test adding vectors to IVF index."""

    def test_add_single_vector(self):
        """Can add a single vector to the index."""
        index = IVFIndex(dimension=3)
        vector_id = uuid4()
        vector = [1.0, 2.0, 3.0]

        index.add(vector_id, vector)

        assert len(index._vectors) == 1
        assert vector_id in index._vectors
        np.testing.assert_array_almost_equal(index._vectors[vector_id], np.array(vector, dtype=np.float32))

    def test_add_multiple_vectors(self):
        """Can add multiple vectors to the index."""
        index = IVFIndex(dimension=3)

        vectors = [
            (uuid4(), [1.0, 0.0, 0.0]),
            (uuid4(), [0.0, 1.0, 0.0]),
            (uuid4(), [0.0, 0.0, 1.0]),
        ]

        for vid, vec in vectors:
            index.add(vid, vec)

        assert len(index._vectors) == 3

    def test_add_dimension_validation(self):
        """Raises ValueError for dimension mismatch."""
        index = IVFIndex(dimension=3)
        vector_id = uuid4()

        # Too short
        with pytest.raises(ValueError, match="doesn't match index dimension"):
            index.add(vector_id, [1.0, 2.0])

        # Too long
        with pytest.raises(ValueError, match="doesn't match index dimension"):
            index.add(vector_id, [1.0, 2.0, 3.0, 4.0])

    def test_bulk_add(self):
        """Can bulk add multiple vectors."""
        index = IVFIndex(dimension=3)

        vectors = [
            (uuid4(), [1.0, 0.0, 0.0]),
            (uuid4(), [0.0, 1.0, 0.0]),
            (uuid4(), [0.0, 0.0, 1.0]),
        ]

        index.bulk_add(vectors)

        assert len(index._vectors) == 3
        for vid, _ in vectors:
            assert vid in index._vectors


class TestIVFIndexUpdate:
    """Test updating vectors in IVF index."""

    def test_update_existing_vector(self):
        """Can update an existing vector."""
        index = IVFIndex(dimension=3)
        vector_id = uuid4()

        # Add initial vector
        index.add(vector_id, [1.0, 2.0, 3.0])

        # Update vector
        new_vector = [4.0, 5.0, 6.0]
        index.update(vector_id, new_vector)

        assert len(index._vectors) == 1
        np.testing.assert_array_almost_equal(index._vectors[vector_id], np.array(new_vector, dtype=np.float32))

    def test_update_nonexistent_vector(self):
        """Raises KeyError for nonexistent vector."""
        index = IVFIndex(dimension=3)
        vector_id = uuid4()

        with pytest.raises(KeyError, match="not found"):
            index.update(vector_id, [1.0, 2.0, 3.0])


class TestIVFIndexDelete:
    """Test deleting vectors from IVF index."""

    def test_delete_existing_vector(self):
        """Can delete an existing vector."""
        index = IVFIndex(dimension=3)
        vector_id = uuid4()

        # Add and delete
        index.add(vector_id, [1.0, 2.0, 3.0])
        assert len(index._vectors) == 1

        index.delete(vector_id)
        assert len(index._vectors) == 0
        assert vector_id not in index._vectors

    def test_delete_nonexistent_vector(self):
        """Raises KeyError for nonexistent vector."""
        index = IVFIndex(dimension=3)
        vector_id = uuid4()

        with pytest.raises(KeyError, match="not found"):
            index.delete(vector_id)


class TestIVFIndexClear:
    """Test clearing IVF index."""

    def test_clear_index(self):
        """Can clear all vectors from index."""
        index = IVFIndex(dimension=3)

        # Add vectors
        for i in range(5):
            index.add(uuid4(), [float(i), float(i), float(i)])

        assert len(index._vectors) == 5

        # Clear
        index.clear()

        assert len(index._vectors) == 0
        assert len(index._clusters) == 0
        assert index._centroids is None
        assert not index._is_built


class TestIVFIndexHelpers:
    """Test IVF index helper methods."""

    def test_compute_default_nlist_small_dataset(self):
        """Default nlist is 1 for very small datasets."""
        index = IVFIndex(dimension=3)

        # Add 5 vectors (< 10)
        for i in range(5):
            index.add(uuid4(), [float(i), 0.0, 0.0])

        nlist = index._compute_default_nlist()
        assert nlist == 1

    def test_compute_default_nlist_medium_dataset(self):
        """Default nlist is sqrt(n) for medium datasets."""
        index = IVFIndex(dimension=3)

        # Add 100 vectors
        for i in range(100):
            index.add(uuid4(), [float(i), 0.0, 0.0])

        nlist = index._compute_default_nlist()
        assert nlist == 10  # sqrt(100) = 10

    def test_compute_default_nlist_large_dataset(self):
        """Default nlist scales with sqrt(n) for large datasets."""
        index = IVFIndex(dimension=3)

        # Add 10000 vectors (simulated - just set _vectors dict)
        for i in range(10000):
            index._vectors[uuid4()] = np.array([float(i), 0.0, 0.0], dtype=np.float32)

        nlist = index._compute_default_nlist()
        assert nlist == 100  # sqrt(10000) = 100


class TestIVFIndexClustering:
    """Test IVF index clustering functionality."""

    def test_explicit_build(self):
        """Can explicitly build clusters."""
        index = IVFIndex(dimension=3, config={"nlist": 2})

        # Add vectors in two clear groups
        vectors = [
            (uuid4(), [1.0, 0.0, 0.0]),
            (uuid4(), [0.9, 0.1, 0.0]),
            (uuid4(), [0.0, 0.0, 1.0]),
            (uuid4(), [0.1, 0.0, 0.9]),
        ]

        for vid, vec in vectors:
            index.add(vid, vec)

        # Build clusters
        index.build()

        assert index._is_built
        assert len(index._clusters) == 2
        assert index._centroids is not None
        assert index._centroids.shape == (2, 3)

    def test_lazy_build_on_search(self):
        """Clustering triggers automatically on first search."""
        index = IVFIndex(dimension=3, config={"nlist": 2, "nprobe": 2})

        # Add vectors
        vectors = [
            (uuid4(), [1.0, 0.0, 0.0]),
            (uuid4(), [0.0, 1.0, 0.0]),
            (uuid4(), [0.0, 0.0, 1.0]),
        ]

        for vid, vec in vectors:
            index.add(vid, vec)

        assert not index._is_built

        # Search triggers lazy build
        results = index.search([1.0, 0.0, 0.0], k=2)

        assert index._is_built
        assert len(results) > 0

    def test_nlist_capped_to_vector_count(self):
        """nlist is capped to number of vectors if too large."""
        index = IVFIndex(dimension=3, config={"nlist": 100})

        # Add only 5 vectors
        for i in range(5):
            index.add(uuid4(), [float(i), 0.0, 0.0])

        index.build()

        # Should create 5 clusters, not 100
        assert len(index._clusters) == 5
        assert index._centroids.shape[0] == 5

    def test_add_after_build_assigns_to_cluster(self):
        """Vectors added after build are assigned to nearest cluster."""
        index = IVFIndex(dimension=3, config={"nlist": 2})

        # Add initial vectors and build
        initial_vectors = [
            (uuid4(), [1.0, 0.0, 0.0]),
            (uuid4(), [0.0, 1.0, 0.0]),
        ]

        for vid, vec in initial_vectors:
            index.add(vid, vec)

        index.build()
        assert index._is_built

        # Add new vector after build
        new_id = uuid4()
        index.add(new_id, [0.9, 0.1, 0.0])

        # Verify it was added to a cluster
        found_in_cluster = False
        for cluster_vectors in index._clusters.values():
            if any(vid == new_id for vid, _ in cluster_vectors):
                found_in_cluster = True
                break

        assert found_in_cluster


class TestIVFIndexSearch:
    """Test IVF index search functionality."""

    def test_search_empty_index(self):
        """Searching empty index returns empty results."""
        index = IVFIndex(dimension=3, config={"nlist": 2, "nprobe": 1})

        results = index.search([1.0, 0.0, 0.0], k=5)

        assert results == []

    def test_search_returns_results(self):
        """Search returns results sorted by similarity."""
        index = IVFIndex(dimension=3, config={"nlist": 2, "nprobe": 2})

        # Add vectors
        id1 = uuid4()
        id2 = uuid4()
        id3 = uuid4()

        index.add(id1, [1.0, 0.0, 0.0])
        index.add(id2, [0.9, 0.1, 0.0])
        index.add(id3, [0.0, 1.0, 0.0])

        # Search for vector similar to id1
        results = index.search([1.0, 0.0, 0.0], k=3)

        assert len(results) == 3
        # First result should be most similar (id1)
        assert results[0][0] == id1
        # Scores should be in descending order
        assert results[0][1] >= results[1][1] >= results[2][1]

    def test_search_respects_k_limit(self):
        """Search returns at most k results."""
        index = IVFIndex(dimension=3, config={"nlist": 2, "nprobe": 2})

        # Add 10 vectors
        for i in range(10):
            index.add(uuid4(), [float(i) / 10.0, 0.0, 0.0])

        # Search for k=3
        results = index.search([0.5, 0.0, 0.0], k=3)

        assert len(results) == 3

    def test_search_with_nprobe_1(self):
        """Search with nprobe=1 searches only 1 cluster."""
        index = IVFIndex(dimension=3, config={"nlist": 3, "nprobe": 1})

        # Add vectors in distinct groups
        vectors = [
            (uuid4(), [1.0, 0.0, 0.0]),
            (uuid4(), [0.9, 0.0, 0.0]),
            (uuid4(), [0.0, 1.0, 0.0]),
            (uuid4(), [0.0, 0.9, 0.0]),
            (uuid4(), [0.0, 0.0, 1.0]),
            (uuid4(), [0.0, 0.0, 0.9]),
        ]

        for vid, vec in vectors:
            index.add(vid, vec)

        # Build and search
        results = index.search([1.0, 0.0, 0.0], k=10)

        # Should find results (may not be all 6 due to single cluster search)
        assert len(results) > 0
        assert len(results) <= 6

    def test_search_with_different_metrics(self):
        """Search works with different distance metrics."""
        for metric in ["cosine", "euclidean", "dot_product"]:
            index = IVFIndex(dimension=3, config={"nlist": 2, "nprobe": 2, "metric": metric})

            # Add vectors
            id1 = uuid4()
            index.add(id1, [1.0, 0.0, 0.0])
            index.add(uuid4(), [0.0, 1.0, 0.0])
            index.add(uuid4(), [0.0, 0.0, 1.0])

            # Search should work
            results = index.search([1.0, 0.0, 0.0], k=2)

            assert len(results) > 0
            # First result should be exact match
            assert results[0][0] == id1

    def test_search_dimension_validation(self):
        """Search validates query vector dimension."""
        index = IVFIndex(dimension=3)

        index.add(uuid4(), [1.0, 0.0, 0.0])

        # Wrong dimension should raise
        with pytest.raises(ValueError, match="doesn't match index dimension"):
            index.search([1.0, 0.0], k=5)

    def test_empty_cluster_skipping(self):
        """Search skips empty clusters correctly."""
        index = IVFIndex(dimension=3, config={"nlist": 3, "nprobe": 3})

        # Add vectors and build
        vectors = [
            (uuid4(), [1.0, 0.0, 0.0]),
            (uuid4(), [0.9, 0.0, 0.0]),
        ]

        for vid, vec in vectors:
            index.add(vid, vec)

        index.build()

        # Delete one vector to create empty cluster
        index.delete(vectors[0][0])

        # Search should still work
        results = index.search([1.0, 0.0, 0.0], k=5)

        assert len(results) > 0
