"""
IVF (Inverted File) Index Implementation.

Cluster-based approximate nearest neighbor search using K-means clustering.
This implementation stores raw vectors (FLAT) without compression or quantization.

Algorithm:
1. Vectors are partitioned into nlist clusters using K-means
2. Each cluster stores vectors that are similar to the cluster centroid
3. During search, only nprobe nearest clusters are searched (not all vectors)
4. This provides faster search at the cost of approximate results

Time Complexity:
- Add (after clustering): O(1) - Assign to nearest cluster
- Build: O(n*d*k*i) - K-means clustering (n=vectors, d=dimension, k=nlist, i=iterations)
- Search: O(c*d + m*d) - c=nprobe clusters, m=avg cluster size
- Update: O(1) - Reassign to nearest cluster
- Delete: O(1) - Remove from cluster

Space Complexity: O(n*d + k*d) where k=nlist (centroids overhead)

Pros:
- Much faster than FLAT on large datasets (10,000+ vectors)
- Configurable speed/accuracy tradeoff (nprobe parameter)
- Handles high-dimensional vectors efficiently

Cons:
- Approximate results (80-95% recall typical)
- Requires clustering build step
- Memory overhead for cluster centroids
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from sklearn.cluster import KMeans

from my_vector_db.indexes.base import VectorIndex


class IVFIndex(VectorIndex):
    """
    IVF index with FLAT storage (no compression).

    Uses K-means clustering to partition vectors into clusters for faster
    approximate nearest neighbor search.

    Configuration Parameters (index_config):
        - nlist (int): Number of clusters (default: sqrt(n))
        - nprobe (int): Number of clusters to search (default: 1)
        - metric (str): Distance metric - "cosine", "euclidean", or "dot_product"

    Example:
        >>> index = IVFIndex(dimension=384, config={"nlist": 100, "nprobe": 10})
        >>> index.add(vector_id, vector)
        >>> index.build()  # Explicit clustering
        >>> results = index.search(query_vector, k=10)
    """

    def __init__(self, dimension: int, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the IVF index.

        Args:
            dimension: Dimensionality of the vectors
            config: Optional configuration with keys:
                - nlist: Number of clusters (positive integer)
                - nprobe: Number of clusters to search (positive integer, <= nlist)
                - metric: Distance metric ("cosine", "euclidean", "dot_product")

        Raises:
            ValueError: If config validation fails
        """
        super().__init__(dimension, config)

        # Validate config parameters
        self._validate_config()

        # Vector storage: all vectors indexed by ID
        self._vectors: Dict[UUID, np.ndarray] = {}

        # Cluster storage: cluster_idx -> list of (vector_id, vector) tuples
        self._clusters: Dict[int, List[Tuple[UUID, np.ndarray]]] = {}

        # Cluster centroids: shape (nlist, dimension)
        self._centroids: Optional[np.ndarray] = None

        # K-means model instance
        self._kmeans: Optional[KMeans] = None

        # Whether clustering has been performed
        self._is_built = False

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any config parameter is invalid
        """
        nlist = self.config.get("nlist")
        nprobe = self.config.get("nprobe")
        metric = self.config.get("metric", "cosine")

        # Validate nlist
        if nlist is not None:
            if not isinstance(nlist, int) or nlist <= 0:
                raise ValueError("nlist must be a positive integer")

        # Validate nprobe
        if nprobe is not None:
            if not isinstance(nprobe, int) or nprobe <= 0:
                raise ValueError("nprobe must be a positive integer")

        # Validate metric
        if metric not in ["cosine", "euclidean", "dot_product"]:
            raise ValueError(f"Unknown metric: {metric}. Must be 'cosine', 'euclidean', or 'dot_product'")

    def add(self, vector_id: UUID, vector: List[float]) -> None:
        """
        Add a single vector to the index.

        If the index is already built (clustered), the vector is assigned to the
        nearest cluster. Otherwise, it's queued for the next clustering operation.

        Args:
            vector_id: Unique identifier for the vector
            vector: The vector to add (must match index dimension)

        Raises:
            ValueError: If vector dimension doesn't match index dimension
        """
        # Validate dimension
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} doesn't match index dimension {self.dimension}"
            )

        # Convert to numpy array
        np_vector = np.array(vector, dtype=np.float32)

        # Store in main vector dictionary
        self._vectors[vector_id] = np_vector

        # If index is built, assign to nearest cluster
        if self._is_built and self._centroids is not None:
            cluster_idx = self._find_nearest_cluster(np_vector)
            self._clusters[cluster_idx].append((vector_id, np_vector))

    def bulk_add(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """
        Add multiple vectors efficiently.

        For IVF index, this delegates to add() for each vector.
        After bulk adding, consider calling build() to reoptimize clusters.

        Args:
            vectors: List of (vector_id, vector) tuples
        """
        for vector_id, vector in vectors:
            self.add(vector_id, vector)

    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors using cluster-based pruning.

        If index not built, triggers lazy clustering before search.

        Args:
            query_vector: The query vector
            k: Number of nearest neighbors to return

        Returns:
            List of (vector_id, similarity_score) tuples, sorted by score descending

        Raises:
            ValueError: If query vector dimension doesn't match
        """
        # Validate dimension
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} doesn't match index dimension {self.dimension}"
            )

        # Lazy build on first search
        if not self._is_built:
            self._build_clusters()

        # Handle empty index
        if len(self._vectors) == 0:
            return []

        # TODO: Implement cluster-based search (Phase 4 - User Story 2)
        raise NotImplementedError("Search to be implemented in Phase 4 (User Story 2)")

    def update(self, vector_id: UUID, vector: List[float]) -> None:
        """
        Update an existing vector.

        Implementation uses delete + add pattern to ensure proper cluster reassignment.

        Args:
            vector_id: ID of the vector to update
            vector: The new vector

        Raises:
            KeyError: If vector_id doesn't exist
            ValueError: If vector dimension doesn't match
        """
        if vector_id not in self._vectors:
            raise KeyError(f"Vector ID {vector_id} not found")

        # Delete old vector then add new one
        self.delete(vector_id)
        self.add(vector_id, vector)

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

        # Remove from main storage
        del self._vectors[vector_id]

        # Remove from cluster if built
        if self._is_built:
            for cluster_vectors in self._clusters.values():
                # Filter out the deleted vector
                cluster_vectors[:] = [
                    (vid, vec) for vid, vec in cluster_vectors if vid != vector_id
                ]

    def clear(self) -> None:
        """
        Remove all vectors from the index and reset clustering state.
        """
        self._vectors.clear()
        self._clusters.clear()
        self._centroids = None
        self._kmeans = None
        self._is_built = False

    def build(self) -> None:
        """
        Explicitly build/rebuild clusters.

        Triggers K-means clustering to partition vectors into nlist clusters.
        Can also happen lazily on first search if not called explicitly.
        """
        self._build_clusters()

    def _build_clusters(self) -> None:
        """
        Internal: Perform K-means clustering.

        Creates nlist clusters and assigns all vectors to their nearest cluster.
        Sets _is_built flag and initializes _centroids and _clusters.
        """
        # Handle empty index
        if len(self._vectors) == 0:
            self._is_built = True
            return

        # TODO: Implement K-means clustering (Phase 3 - T015)
        raise NotImplementedError("Clustering to be fully implemented in T015")

    def _compute_default_nlist(self) -> int:
        """
        Compute default nlist as sqrt(n) where n is number of vectors.

        Returns:
            Default number of clusters (minimum 1)
        """
        n = len(self._vectors)
        if n < 10:
            return 1
        return max(1, int(np.sqrt(n)))

    def _find_nearest_cluster(self, vector: np.ndarray) -> int:
        """
        Find the nearest cluster index for a vector.

        Uses the configured distance metric to compute distances to all centroids.

        Args:
            vector: The vector to assign

        Returns:
            Index of the nearest cluster (0 to nlist-1)

        Raises:
            RuntimeError: If index not built (no centroids)
        """
        if self._centroids is None:
            raise RuntimeError("Index not built - no centroids available")

        # TODO: Implement centroid distance computation (Phase 3 - T017)
        raise NotImplementedError("Nearest cluster finding to be implemented in T017")
