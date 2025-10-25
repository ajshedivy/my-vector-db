"""
Abstract base class for vector indexes.

All vector index implementations must inherit from VectorIndex and implement
the required methods for adding, searching, updating, and deleting vectors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np


class VectorIndex(ABC):
    """
    Abstract base class for vector indexing algorithms.

    This class defines the interface that all vector index implementations must follow.
    """

    def __init__(self, dimension: int, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the vector index.

        Args:
            dimension: Dimensionality of the vectors
            config: Optional configuration parameters specific to the index type
        """
        self.dimension = dimension
        self.config = config or {}

    @abstractmethod
    def add(self, vector_id: UUID, vector: List[float]) -> None:
        """
        Add a single vector to the index.

        Args:
            vector_id: Unique identifier for the vector (chunk ID)
            vector: The vector to add

        Raises:
            ValueError: If vector dimension doesn't match index dimension

        TODO: Implement in subclasses
        """
        pass

    @abstractmethod
    def bulk_add(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """
        Add multiple vectors to the index efficiently.

        Args:
            vectors: List of (vector_id, vector) tuples

        TODO: Implement in subclasses. Can be optimized for batch insertion.
        """
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_vector: The query vector
            k: Number of nearest neighbors to return

        Returns:
            List of (vector_id, similarity_score) tuples, sorted by score (descending)

        TODO: Implement in subclasses
        """
        pass

    @abstractmethod
    def update(self, vector_id: UUID, vector: List[float]) -> None:
        """
        Update an existing vector in the index.

        Args:
            vector_id: ID of the vector to update
            vector: The new vector

        Raises:
            KeyError: If vector_id doesn't exist

        TODO: Implement in subclasses
        """
        pass

    @abstractmethod
    def delete(self, vector_id: UUID) -> None:
        """
        Delete a vector from the index.

        Args:
            vector_id: ID of the vector to delete

        Raises:
            KeyError: If vector_id doesn't exist

        TODO: Implement in subclasses
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all vectors from the index.

        TODO: Implement in subclasses
        """
        pass

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity = (A · B) / (||A|| * ||B||)
        Returns a value between -1 and 1, where 1 means identical direction.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        # Avoid division by zero for zero-norm vectors
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.

        Distance = sqrt(sum((A - B)^2))

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(vec1 - vec2))

    @staticmethod
    def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate dot product (inner product) between two vectors.

        Dot product = Σ(A_i * B_i)
        Higher values indicate more similar vectors (for normalized vectors).

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Dot product score
        """
        return float(np.dot(vec1, vec2))
