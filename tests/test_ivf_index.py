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
