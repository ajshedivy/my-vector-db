"""
Flat (Brute Force) Vector Index.

This is the simplest vector index that performs exhaustive linear search.
It compares the query vector against every vector in the index.

Time Complexity:
- Add: O(1) - Simply append to storage
- Search: O(n * d) where n = number of vectors, d = dimension
  Must compute similarity for every vector
- Update: O(1) - Direct dictionary access
- Delete: O(1) - Direct dictionary removal

Space Complexity: O(n * d)
- Stores all vectors in memory

Pros:
- Exact search (100% recall)
- Simple implementation
- Fast for small datasets

Cons:
- Slow for large datasets (scales linearly)
- No optimization for high-dimensional spaces
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from my_vector_db.indexes.base import VectorIndex


class FlatIndex(VectorIndex):
    """
    Flat index using brute-force linear search.

    Stores vectors in a dictionary and computes similarity against all vectors
    during search. Best for small to medium datasets where exact results are required.
    """

    def __init__(self, dimension: int, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the flat index.

        Args:
            dimension: Dimensionality of the vectors
            config: Optional configuration (not used for flat index)
        """
        super().__init__(dimension, config)
        self._vectors: Dict[UUID, np.ndarray] = {}

    def add(self, vector_id: UUID, vector: List[float]) -> None:
        """
        Add a vector to the index.

        Args:
            vector_id: Unique identifier for the vector
            vector: The vector to add

        Raises:
            ValueError: If vector dimension doesn't match
        """
        if len(vector) != self.dimension:
            raise ValueError("Vector dimension does not match index dimension")
        self._vectors[vector_id] = np.array(vector)

    def bulk_add(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """
        Add multiple vectors efficiently.

        For flat index, this is just multiple add() calls.

        Args:
            vectors: List of (vector_id, vector) tuples
        """
        for vector_id, vector in vectors:
            self.add(vector_id, vector)

    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors using brute force.

        Algorithm:
        1. Convert query to numpy array
        2. Compute similarity with every vector in the index
        3. Sort by similarity (descending)
        4. Return top k results

        Args:
            query_vector: The query vector
            k: Number of nearest neighbors to return

        Returns:
            List of (vector_id, similarity_score) tuples
        """
        if len(query_vector) != self.dimension:
            raise ValueError("Query vector dimension does not match index dimension")

        query_np = np.array(query_vector)
        metric = self.config.get("metric", "cosine")
        similarities = []

        for vector_id, vector in self._vectors.items():
            if metric == "cosine":
                sim = self.cosine_similarity(query_np, vector)
            elif metric == "euclidean":
                sim = -self.euclidean_distance(query_np, vector)
            elif metric == "dot_product":
                sim = self.dot_product(query_np, vector)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            similarities.append((vector_id, sim))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def update(self, vector_id: UUID, vector: List[float]) -> None:
        """
        Update an existing vector.

        Args:
            vector_id: ID of the vector to update
            vector: The new vector

        Raises:
            KeyError: If vector_id doesn't exist
            ValueError: If vector dimension doesn't match
        """
        if vector_id not in self._vectors:
            raise KeyError(f"Vector ID {vector_id} not found")
        if len(vector) != self.dimension:
            raise ValueError("Vector dimension does not match index dimension")
        self._vectors[vector_id] = np.array(vector)

    def delete(self, vector_id: UUID) -> None:
        """
        Delete a vector from the index.

        Args:
            vector_id: ID of the vector to delete

        Raises:
            KeyError: If vector_id doesn't exist
        """
        if vector_id not in self._vectors:
            raise KeyError(f"Vector ID {vector_id} not found")
        del self._vectors[vector_id]

    def clear(self) -> None:
        """
        Remove all vectors from the index.
        """
        self._vectors.clear()
