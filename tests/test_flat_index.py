"""
FlatIndex Unit Tests

Tests the FlatIndex implementation with different distance metrics and edge cases.
Run with: pytest tests/test_flat_index.py -v
"""

import pytest
import numpy as np
from uuid import uuid4

from src.my_vector_db.indexes.flat import FlatIndex


@pytest.fixture
def flat_index():
    """Create a basic FlatIndex with dimension 3."""
    return FlatIndex(dimension=3)


class TestBasicOperations:
    """Tests for basic FlatIndex CRUD operations."""

    def test_add_vectors(self, flat_index: FlatIndex):
        """Test adding vectors to the index."""
        id1 = uuid4()
        id2 = uuid4()
        id3 = uuid4()

        flat_index.add(id1, [1.0, 0.0, 0.0])
        flat_index.add(id2, [0.0, 1.0, 0.0])
        flat_index.add(id3, [0.0, 0.0, 1.0])

        assert len(flat_index._vectors) == 3

    def test_update_vector(self, flat_index: FlatIndex):
        """Test updating an existing vector."""
        vector_id = uuid4()
        flat_index.add(vector_id, [1.0, 0.0, 0.0])

        # Update vector
        flat_index.update(vector_id, [1.0, 1.0, 0.0])
        updated = flat_index._vectors[vector_id]

        assert np.allclose(updated, [1.0, 1.0, 0.0])

    def test_delete_vector(self, flat_index: FlatIndex):
        """Test deleting a vector from the index."""
        id1 = uuid4()
        id2 = uuid4()

        flat_index.add(id1, [1.0, 0.0, 0.0])
        flat_index.add(id2, [0.0, 1.0, 0.0])

        # Delete id2
        flat_index.delete(id2)

        assert len(flat_index._vectors) == 1
        assert id2 not in flat_index._vectors
        assert id1 in flat_index._vectors

    def test_clear_index(self, flat_index: FlatIndex):
        """Test clearing all vectors from the index."""
        flat_index.add(uuid4(), [1.0, 0.0, 0.0])
        flat_index.add(uuid4(), [0.0, 1.0, 0.0])
        flat_index.add(uuid4(), [0.0, 0.0, 1.0])

        flat_index.clear()

        assert len(flat_index._vectors) == 0

    def test_bulk_add(self):
        """Test bulk addition of vectors."""
        index = FlatIndex(dimension=5)

        # Create 100 vectors
        vectors = []
        for i in range(100):
            vector_id = uuid4()
            vector = [float(i % 10)] * 5
            vectors.append((vector_id, vector))

        # Bulk add
        index.bulk_add(vectors)

        assert len(index._vectors) == 100


class TestDimensionValidation:
    """Tests for dimension validation."""

    def test_add_wrong_dimension(self, flat_index: FlatIndex):
        """Test adding vector with wrong dimension raises ValueError."""
        with pytest.raises(ValueError, match="dimension"):
            flat_index.add(uuid4(), [1.0, 2.0])  # Only 2 dimensions

    def test_update_wrong_dimension(self, flat_index: FlatIndex):
        """Test updating vector with wrong dimension raises ValueError."""
        vector_id = uuid4()
        flat_index.add(vector_id, [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="dimension"):
            flat_index.update(vector_id, [1.0, 2.0, 3.0, 4.0])  # 4 dimensions

    def test_search_wrong_dimension(self, flat_index: FlatIndex):
        """Test searching with wrong dimension raises ValueError."""
        flat_index.add(uuid4(), [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="dimension"):
            flat_index.search([1.0, 2.0], k=1)  # Only 2 dimensions


class TestErrorHandling:
    """Tests for error handling."""

    def test_update_nonexistent_vector(self, flat_index: FlatIndex):
        """Test updating non-existent vector raises KeyError."""
        flat_index.add(uuid4(), [1.0, 2.0, 3.0])

        with pytest.raises(KeyError):
            flat_index.update(uuid4(), [1.0, 2.0, 3.0])

    def test_delete_nonexistent_vector(self, flat_index: FlatIndex):
        """Test deleting non-existent vector raises KeyError."""
        flat_index.add(uuid4(), [1.0, 2.0, 3.0])

        with pytest.raises(KeyError):
            flat_index.delete(uuid4())


class TestCosineSimilarity:
    """Tests for cosine similarity search."""

    @pytest.fixture
    def cosine_index(self):
        """Create FlatIndex with cosine similarity metric."""
        return FlatIndex(dimension=3, config={"metric": "cosine"})

    def test_cosine_similarity_search(self, cosine_index: FlatIndex):
        """Test search with cosine similarity metric."""
        # Add vectors with known similarities
        id1 = uuid4()  # [1, 0, 0]
        id2 = uuid4()  # [1, 1, 0] - 45 degrees from id1
        id3 = uuid4()  # [0, 1, 0] - 90 degrees from id1
        id4 = uuid4()  # [-1, 0, 0] - 180 degrees from id1 (opposite)

        cosine_index.add(id1, [1.0, 0.0, 0.0])
        cosine_index.add(id2, [1.0, 1.0, 0.0])
        cosine_index.add(id3, [0.0, 1.0, 0.0])
        cosine_index.add(id4, [-1.0, 0.0, 0.0])

        # Query with [1, 0, 0] - should match id1 perfectly
        results = cosine_index.search([1.0, 0.0, 0.0], k=4)

        # Check order: id1 (cos=1.0) > id2 (cos=0.707) > id3 (cos=0) > id4 (cos=-1.0)
        assert results[0][0] == id1  # Most similar
        assert results[0][1] == pytest.approx(1.0)  # Perfect match

        assert results[1][0] == id2  # Second most similar
        assert 0.7 < results[1][1] < 0.8  # cos(45°) ≈ 0.707

        assert results[2][0] == id3  # Orthogonal
        assert abs(results[2][1]) < 0.01  # cos(90°) = 0

        assert results[3][0] == id4  # Opposite
        assert results[3][1] == pytest.approx(-1.0)  # cos(180°) = -1

    def test_cosine_perfect_match(self, cosine_index: FlatIndex):
        """Test that identical vectors have cosine similarity of 1.0."""
        vector_id = uuid4()
        cosine_index.add(vector_id, [1.0, 2.0, 3.0])

        results = cosine_index.search([1.0, 2.0, 3.0], k=1)

        assert results[0][0] == vector_id
        assert results[0][1] == pytest.approx(1.0)


class TestEuclideanDistance:
    """Tests for Euclidean distance search."""

    @pytest.fixture
    def euclidean_index(self):
        """Create FlatIndex with Euclidean distance metric."""
        return FlatIndex(dimension=2, config={"metric": "euclidean"})

    def test_euclidean_distance_search(self, euclidean_index: FlatIndex):
        """Test search with Euclidean distance metric."""
        # Add points with known distances
        id1 = uuid4()  # [0, 0]
        id2 = uuid4()  # [1, 0] - distance 1
        id3 = uuid4()  # [0, 1] - distance 1
        id4 = uuid4()  # [3, 4] - distance 5

        euclidean_index.add(id1, [0.0, 0.0])
        euclidean_index.add(id2, [1.0, 0.0])
        euclidean_index.add(id3, [0.0, 1.0])
        euclidean_index.add(id4, [3.0, 4.0])

        # Query with [0, 0] - closest to id1
        results = euclidean_index.search([0.0, 0.0], k=4)

        # Check order: id1 (dist=0) > id2/id3 (dist=1) > id4 (dist=5)
        # Remember: scores are negated, so higher = better
        assert results[0][0] == id1
        assert results[0][1] == pytest.approx(0.0)  # -distance = -0 = 0

        # id2 and id3 have same distance
        assert results[1][0] in [id2, id3]
        assert results[2][0] in [id2, id3]
        assert abs(results[1][1] - (-1.0)) < 0.01  # -distance = -1

        assert results[3][0] == id4
        assert abs(results[3][1] - (-5.0)) < 0.01  # -distance = -5

    def test_euclidean_perfect_match(self, euclidean_index: FlatIndex):
        """Test that identical points have distance of 0."""
        vector_id = uuid4()
        euclidean_index.add(vector_id, [1.0, 2.0])

        results = euclidean_index.search([1.0, 2.0], k=1)

        assert results[0][0] == vector_id
        assert results[0][1] == pytest.approx(0.0)


class TestDotProduct:
    """Tests for dot product search."""

    @pytest.fixture
    def dot_product_index(self):
        """Create FlatIndex with dot product metric."""
        return FlatIndex(dimension=3, config={"metric": "dot_product"})

    def test_dot_product_search(self, dot_product_index: FlatIndex):
        """Test search with dot product metric."""
        # Add vectors with known dot products
        id1 = uuid4()  # [2, 0, 0]
        id2 = uuid4()  # [1, 1, 1]
        id3 = uuid4()  # [0, 0, 1]
        id4 = uuid4()  # [-1, 0, 0]

        dot_product_index.add(id1, [2.0, 0.0, 0.0])
        dot_product_index.add(id2, [1.0, 1.0, 1.0])
        dot_product_index.add(id3, [0.0, 0.0, 1.0])
        dot_product_index.add(id4, [-1.0, 0.0, 0.0])

        # Query with [1, 1, 1]
        # Dot products: id1=2, id2=3, id3=1, id4=-1
        results = dot_product_index.search([1.0, 1.0, 1.0], k=4)

        assert results[0][0] == id2
        assert results[0][1] == pytest.approx(3.0)  # [1,1,1]·[1,1,1] = 3

        assert results[1][0] == id1
        assert results[1][1] == pytest.approx(2.0)  # [1,1,1]·[2,0,0] = 2

        assert results[2][0] == id3
        assert results[2][1] == pytest.approx(1.0)  # [1,1,1]·[0,0,1] = 1

        assert results[3][0] == id4
        assert results[3][1] == pytest.approx(-1.0)  # [1,1,1]·[-1,0,0] = -1


class TestSearchEdgeCases:
    """Tests for search edge cases."""

    def test_search_empty_index(self, flat_index: FlatIndex):
        """Test searching an empty index returns empty list."""
        results = flat_index.search([1.0, 2.0, 3.0], k=5)

        assert len(results) == 0

    def test_search_k_greater_than_vectors(self, flat_index: FlatIndex):
        """Test search with k > number of vectors returns all vectors."""
        # Add 3 vectors
        for i in range(3):
            flat_index.add(uuid4(), [float(i)] * 3)

        # Request 10 results
        results = flat_index.search([1.0, 1.0, 1.0], k=10)

        assert len(results) == 3  # Only 3 vectors exist

    def test_search_k_zero(self, flat_index: FlatIndex):
        """Test search with k=0 returns empty list."""
        flat_index.add(uuid4(), [1.0, 2.0, 3.0])

        results = flat_index.search([1.0, 1.0, 1.0], k=0)

        assert len(results) == 0


class TestInvalidMetric:
    """Tests for invalid metric handling."""

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError during search."""
        index = FlatIndex(dimension=3, config={"metric": "invalid_metric"})
        index.add(uuid4(), [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="(?i)metric"):
            index.search([1.0, 2.0, 3.0], k=1)
