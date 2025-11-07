# Quick Start Guide: IVF Index

**Feature**: 001-ivfflat-index
**Date**: 2025-11-07
**Audience**: Developers implementing the IVF index

## Overview

This guide provides a step-by-step implementation roadmap for the IVF index feature. Follow this sequence to deliver the feature incrementally with working checkpoints.

## Prerequisites

Before starting implementation:

1. ✅ **Read design artifacts**:
   - [spec.md](./spec.md) - Feature requirements and user stories
   - [research.md](./research.md) - Design decisions and algorithms
   - [data-model.md](./data-model.md) - Data structures and validation
   - [contracts/api-contract.md](./contracts/api-contract.md) - API integration

2. ✅ **Environment setup**:
   - Python 3.11+ installed
   - `uv` package manager configured
   - Development dependencies installed: `uv sync`
   - Familiarize with existing FLAT index: `src/my_vector_db/indexes/flat.py`

3. ✅ **Constitutional compliance**:
   - Review [.specify/memory/constitution.md](../../.specify/memory/constitution.md)
   - All principles marked COMPLIANT in plan.md

## Implementation Phases

### Phase 1: Domain Model Update (Foundation)

**Goal**: Enable IVF as a valid index type throughout the system

**Files to Modify**:
1. `src/my_vector_db/domain/models.py`

**Tasks**:
```python
# In IndexType enum, add IVF variant
class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"  # NEW
```

**Checkpoint**: Run existing tests to verify backward compatibility
```bash
uv run pytest tests/unit/test_models.py -v
```

**Success Criteria**:
- ✅ Tests pass
- ✅ IndexType.IVF accessible
- ✅ Pydantic validation accepts "ivf" string value

---

### Phase 2: IVF Index Skeleton (P1 Foundation)

**Goal**: Create IVFIndex class implementing VectorIndex interface with basic structure

**Files to Create**:
1. `src/my_vector_db/indexes/ivf.py`

**Minimal Implementation**:

```python
"""
IVF (Inverted File) Index Implementation.
Cluster-based approximate nearest neighbor search.
"""
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import numpy as np
from sklearn.cluster import KMeans

from my_vector_db.indexes.base import VectorIndex


class IVFIndex(VectorIndex):
    """IVF index with FLAT storage (no compression)."""

    def __init__(self, dimension: int, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(dimension, config)
        self._vectors: Dict[UUID, np.ndarray] = {}
        self._clusters: Dict[int, List[Tuple[UUID, np.ndarray]]] = {}
        self._centroids: Optional[np.ndarray] = None
        self._kmeans: Optional[KMeans] = None
        self._is_built = False

    def add(self, vector_id: UUID, vector: List[float]) -> None:
        """Add vector to index."""
        if len(vector) != self.dimension:
            raise ValueError("Vector dimension doesn't match index dimension")
        np_vector = np.array(vector)
        self._vectors[vector_id] = np_vector

        if self._is_built:
            # Assign to nearest cluster
            cluster_idx = self._find_nearest_cluster(np_vector)
            self._clusters[cluster_idx].append((vector_id, np_vector))

    def bulk_add(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """Add multiple vectors."""
        for vector_id, vector in vectors:
            self.add(vector_id, vector)

    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
        """Search for k nearest neighbors."""
        if len(query_vector) != self.dimension:
            raise ValueError("Query vector dimension doesn't match")

        # Lazy build on first search
        if not self._is_built:
            self._build_clusters()

        # TODO: Implement cluster-based search
        raise NotImplementedError("Search to be implemented in Phase 3")

    def update(self, vector_id: UUID, vector: List[float]) -> None:
        """Update vector."""
        if vector_id not in self._vectors:
            raise KeyError(f"Vector ID {vector_id} not found")
        self.delete(vector_id)
        self.add(vector_id, vector)

    def delete(self, vector_id: UUID) -> None:
        """Delete vector."""
        if vector_id not in self._vectors:
            raise KeyError(f"Vector ID {vector_id} not found")

        # Remove from _vectors
        del self._vectors[vector_id]

        # Remove from cluster if built
        if self._is_built:
            for cluster_vectors in self._clusters.values():
                cluster_vectors[:] = [
                    (vid, vec) for vid, vec in cluster_vectors if vid != vector_id
                ]

    def clear(self) -> None:
        """Clear all vectors."""
        self._vectors.clear()
        self._clusters.clear()
        self._centroids = None
        self._is_built = False

    def build(self) -> None:
        """Explicitly build clusters (can also happen lazily on first search)."""
        self._build_clusters()

    def _build_clusters(self) -> None:
        """Internal: Perform K-means clustering."""
        if len(self._vectors) == 0:
            return  # No vectors to cluster

        # Get nlist from config or compute default
        nlist = self.config.get("nlist")
        if nlist is None:
            nlist = self._compute_default_nlist()

        # TODO: Implement K-means clustering
        # This is Phase 3 work
        raise NotImplementedError("Clustering to be implemented in Phase 3")

    def _compute_default_nlist(self) -> int:
        """Compute default nlist as sqrt(n)."""
        n = len(self._vectors)
        if n < 10:
            return 1
        return max(1, int(np.sqrt(n)))

    def _find_nearest_cluster(self, vector: np.ndarray) -> int:
        """Find nearest cluster index for vector."""
        # TODO: Implement centroid distance computation
        raise NotImplementedError("To be implemented in Phase 3")
```

**Checkpoint**: Create basic test
```python
# tests/unit/test_ivf_index.py
def test_ivf_index_creation():
    from my_vector_db.indexes.ivf import IVFIndex
    index = IVFIndex(dimension=384, config={"nlist": 10})
    assert index.dimension == 384
    assert not index._is_built
```

**Success Criteria**:
- ✅ IVFIndex class imports without errors
- ✅ Implements all VectorIndex abstract methods (even as NotImplementedError)
- ✅ Basic structure test passes

---

### Phase 3: Clustering Implementation (P1 Core)

**Goal**: Implement K-means clustering and centroid management

**Files to Modify**:
1. `src/my_vector_db/indexes/ivf.py` - Complete `_build_clusters()` and `_find_nearest_cluster()`

**Implementation**:

```python
def _build_clusters(self) -> None:
    """Perform K-means clustering."""
    if len(self._vectors) == 0:
        return

    nlist = self.config.get("nlist")
    if nlist is None:
        nlist = self._compute_default_nlist()

    # K-means cannot create more clusters than vectors
    nlist = min(nlist, len(self._vectors))

    # Prepare vectors for clustering
    vector_ids = list(self._vectors.keys())
    vectors_array = np.array([self._vectors[vid] for vid in vector_ids])

    # Perform K-means
    self._kmeans = KMeans(
        n_clusters=nlist,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    labels = self._kmeans.fit_predict(vectors_array)
    self._centroids = self._kmeans.cluster_centers_

    # Assign vectors to clusters
    self._clusters = {i: [] for i in range(nlist)}
    for idx, label in enumerate(labels):
        vector_id = vector_ids[idx]
        vector = self._vectors[vector_id]
        self._clusters[label].append((vector_id, vector))

    self._is_built = True

def _find_nearest_cluster(self, vector: np.ndarray) -> int:
    """Find cluster with nearest centroid."""
    if self._centroids is None:
        raise RuntimeError("Index not built")

    metric = self.config.get("metric", "cosine")

    # Compute distances to all centroids
    distances = []
    for centroid in self._centroids:
        if metric == "cosine":
            dist = 1 - self.cosine_similarity(vector, centroid)
        elif metric == "euclidean":
            dist = self.euclidean_distance(vector, centroid)
        elif metric == "dot_product":
            dist = -self.dot_product(vector, centroid)  # Negate for ascending sort
        else:
            raise ValueError(f"Unknown metric: {metric}")
        distances.append(dist)

    # Return index of nearest centroid
    return int(np.argmin(distances))
```

**Tests**:

```python
def test_clustering():
    """Test K-means clustering creates correct number of clusters."""
    index = IVFIndex(dimension=3, config={"nlist": 2})

    # Add vectors
    vectors = [
        (UUID("00000000-0000-0000-0000-000000000001"), [1.0, 0.0, 0.0]),
        (UUID("00000000-0000-0000-0000-000000000002"), [1.0, 0.1, 0.0]),
        (UUID("00000000-0000-0000-0000-000000000003"), [0.0, 0.0, 1.0]),
        (UUID("00000000-0000-0000-0000-000000000004"), [0.0, 0.1, 1.0]),
    ]
    for vid, vec in vectors:
        index.add(vid, vec)

    # Build clusters
    index.build()

    assert index._is_built
    assert len(index._clusters) == 2
    assert index._centroids.shape == (2, 3)
```

**Success Criteria**:
- ✅ K-means clustering completes
- ✅ Centroids computed
- ✅ Vectors assigned to clusters
- ✅ Test verifies cluster count and structure

---

### Phase 4: Search Implementation (P2 Core)

**Goal**: Implement cluster-based approximate search

**Files to Modify**:
1. `src/my_vector_db/indexes/ivf.py` - Complete `search()` method

**Implementation**:

```python
def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]:
    """Search for k nearest neighbors using cluster pruning."""
    if len(query_vector) != self.dimension:
        raise ValueError("Query vector dimension doesn't match")

    # Lazy build on first search
    if not self._is_built:
        self._build_clusters()

    if len(self._vectors) == 0:
        return []

    query_np = np.array(query_vector)
    metric = self.config.get("metric", "cosine")
    nprobe = self.config.get("nprobe", 1)

    # Get nprobe nearest clusters
    nearest_clusters = self._get_nprobe_nearest_clusters(query_np, nprobe)

    # Search within selected clusters
    candidates = []
    for cluster_idx in nearest_clusters:
        for vector_id, vector in self._clusters[cluster_idx]:
            if metric == "cosine":
                score = self.cosine_similarity(query_np, vector)
            elif metric == "euclidean":
                score = -self.euclidean_distance(query_np, vector)
            elif metric == "dot_product":
                score = self.dot_product(query_np, vector)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            candidates.append((vector_id, score))

    # Sort by score descending and return top k
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]

def _get_nprobe_nearest_clusters(self, query_vector: np.ndarray, nprobe: int) -> List[int]:
    """Get indices of nprobe nearest clusters to query."""
    if self._centroids is None:
        raise RuntimeError("Index not built")

    metric = self.config.get("metric", "cosine")
    nlist = len(self._centroids)
    nprobe = min(nprobe, nlist)  # Clamp to nlist

    # Compute similarities to centroids
    similarities = []
    for idx, centroid in enumerate(self._centroids):
        # Skip empty clusters
        if len(self._clusters[idx]) == 0:
            continue

        if metric == "cosine":
            sim = self.cosine_similarity(query_vector, centroid)
        elif metric == "euclidean":
            sim = -self.euclidean_distance(query_vector, centroid)
        elif metric == "dot_product":
            sim = self.dot_product(query_vector, centroid)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        similarities.append((idx, sim))

    # Sort by similarity descending and take top nprobe
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in similarities[:nprobe]]
```

**Tests**:

```python
def test_search():
    """Test IVF search returns results."""
    index = IVFIndex(dimension=3, config={"nlist": 2, "nprobe": 1})

    # Add and build
    vectors = [
        (UUID("00000000-0000-0000-0000-000000000001"), [1.0, 0.0, 0.0]),
        (UUID("00000000-0000-0000-0000-000000000002"), [0.9, 0.1, 0.0]),
        (UUID("00000000-0000-0000-0000-000000000003"), [0.0, 0.0, 1.0]),
    ]
    for vid, vec in vectors:
        index.add(vid, vec)

    # Search (triggers lazy build)
    results = index.search([1.0, 0.0, 0.0], k=2)

    assert len(results) > 0
    assert results[0][0] in [v[0] for v in vectors]
```

**Success Criteria**:
- ✅ Search returns results
- ✅ Results ordered by similarity
- ✅ Lazy build triggers on first search
- ✅ nprobe controls cluster count searched

---

### Phase 5: Service Layer Integration (P1/P2 Connection)

**Goal**: Wire IVF index into service layer factory

**Files to Modify**:
1. `src/my_vector_db/services/library_service.py`

**Changes**:

```python
from my_vector_db.indexes.ivf import IVFIndex  # NEW import

def _create_index(self, index_type: IndexType, dimension: int, config: Dict[str, Any]):
    """Factory method for creating index instances."""
    if index_type == IndexType.FLAT:
        return FlatIndex(dimension, config)
    elif index_type == IndexType.IVF:  # NEW
        return IVFIndex(dimension, config)
    elif index_type == IndexType.HNSW:
        raise NotImplementedError("HNSW not yet implemented")
    else:
        raise ValueError(f"Unknown index type: {index_type}")
```

**Checkpoint**: Integration test

```python
def test_create_ivf_library(client: VectorDBClient):
    """End-to-end: Create IVF library via SDK."""
    library = client.create_library(
        name="test_ivf",
        index_type="ivf",
        index_config={"nlist": 10, "nprobe": 2}
    )
    assert library.index_type == "ivf"
    assert library.index_config["nlist"] == 10
```

**Success Criteria**:
- ✅ IVF libraries can be created via API
- ✅ SDK create_library() works with index_type="ivf"
- ✅ Index factory returns IVFIndex instance

---

### Phase 6: Testing & Validation (P1/P2 Completion)

**Goal**: Comprehensive test coverage

**Tests to Create**:

1. **tests/unit/test_ivf_index.py**:
   - Clustering edge cases (nlist > n, identical vectors)
   - Empty cluster handling
   - Default nlist calculation
   - Metric support (cosine, euclidean, dot_product)
   - Add/update/delete operations
   - Clear operation

2. **tests/integration/test_ivf_search.py**:
   - End-to-end workflow (create, add, search)
   - Lazy build verification
   - Explicit build_index call
   - Update nprobe via update_library
   - Recall measurement vs FLAT

3. **tests/contract/test_ivf_contract.py**:
   - VectorIndex interface compliance
   - All abstract methods implemented
   - Exception contracts honored

**Run Test Suite**:
```bash
# All tests
uv run pytest -v

# Coverage check (must be >80%)
uv run pytest --cov=my_vector_db --cov-report=term-missing

# IVF-specific tests
uv run pytest tests/unit/test_ivf_index.py -v
uv run pytest tests/integration/test_ivf_search.py -v
```

**Success Criteria**:
- ✅ All tests pass
- ✅ Coverage >80% on IVF index code
- ✅ Contract tests verify interface compliance

---

### Phase 7: Configuration & Documentation (P3)

**Goal**: Enable nprobe tuning and document usage

**Tasks**:

1. **Verify update_library for nprobe**:
   - Test updating index_config.nprobe
   - Verify next search uses new nprobe
   - Document in API docs

2. **Update OpenAPI docs**:
   - Add IVF examples to FastAPI route docstrings
   - Update automated docs at /docs endpoint

3. **Documentation**:
   - Update README.md with IVF example
   - Add to SDK documentation
   - Performance comparison guide (IVF vs FLAT)

**Example for README**:
```python
# Create IVF library for faster search on large datasets
library = client.create_library(
    name="large_dataset",
    index_type="ivf",
    index_config={
        "nlist": 100,      # 100 clusters
        "nprobe": 10,      # Search 10 clusters (10% for 80%+ recall)
        "metric": "cosine"
    }
)
```

---

### Phase 8: Maintenance Operations (P4 - Optional)

**Goal**: Verify incremental operations work correctly

**Already Implemented** (verify with tests):
- ✅ Add vector after clustering (Phase 2)
- ✅ Update vector (Phase 2)
- ✅ Delete vector (Phase 2)

**Additional Tests**:
```python
def test_add_after_build():
    """Adding vectors after build assigns to nearest cluster."""
    # Build with initial vectors
    # Add new vector
    # Verify it's in a cluster without full rebuild

def test_update_reassigns_cluster():
    """Updating vector reassigns it to appropriate cluster."""
    # Build, then update a vector significantly
    # Verify it moved to a different cluster

def test_empty_cluster_after_deletes():
    """Deleting all vectors from a cluster leaves it empty but valid."""
    # Build, delete all vectors from one cluster
    # Verify cluster exists but is empty
    # Verify search skips empty cluster
```

---

## Checkpoints & Validation

### After Each Phase

1. **Run tests**: `uv run pytest -v`
2. **Check linting**: `uv run ruff check`
3. **Check formatting**: `uv run ruff format --check`
4. **Type check**: `uvx ty check src/my_vector_db`

### Pre-Merge Checklist

- [ ] All tests pass (>80% coverage)
- [ ] Constitution compliance verified
- [ ] API documentation updated
- [ ] README examples added
- [ ] Code reviewed (if team workflow)
- [ ] Performance baseline measured (IVF vs FLAT)

## Common Pitfalls

### 1. K-means Initialization
**Issue**: Random initialization can cause clustering instability
**Solution**: Use `random_state=42` and `n_init=10` in KMeans

### 2. Metric Consistency
**Issue**: Using different metrics for clustering vs search
**Solution**: Always read `metric` from `self.config` consistently

### 3. Empty Clusters
**Issue**: Not handling empty clusters causes search errors
**Solution**: Skip empty clusters in `_get_nprobe_nearest_clusters()`

### 4. Dimension Validation
**Issue**: Forgetting to validate vector dimensions
**Solution**: Check `len(vector) == self.dimension` in add() and search()

### 5. Thread Safety
**Issue**: Assuming index is thread-safe in isolation
**Solution**: Rely on service-layer RLock (same as FLAT)

## Performance Tuning

### Recommended nlist Values

| Dataset Size | Recommended nlist | Rationale |
|--------------|-------------------|-----------|
| < 1,000 | 10-20 | Small overhead acceptable |
| 1,000 - 10,000 | 50-100 | Balance speed/accuracy |
| 10,000 - 100,000 | 100-500 | sqrt(n) rule of thumb |
| > 100,000 | 1,000+ | More clusters for scaling |

### Recommended nprobe Values

- **Fast search, lower recall**: nprobe = 1-5
- **Balanced**: nprobe = 10% of nlist
- **High recall**: nprobe = 20-30% of nlist
- **Near-exact**: nprobe = 50%+ of nlist (defeats purpose)

## Next Steps

After completing this implementation:

1. **Run `/speckit.tasks`**: Generate detailed task breakdown
2. **Implement in order**: Follow phases 1-8 sequentially
3. **Test incrementally**: Don't skip checkpoints
4. **Measure performance**: Compare IVF vs FLAT on real data
5. **Document learnings**: Update this guide with insights

## Resources

- **K-means documentation**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- **FAISS IVF reference**: https://github.com/facebookresearch/faiss/wiki/Faster-search
- **Existing FLAT index**: `src/my_vector_db/indexes/flat.py` (reference implementation)
- **VectorIndex interface**: `src/my_vector_db/indexes/base.py` (contract to implement)

---

**Ready to implement!** Follow phases sequentially, test at each checkpoint, and deliver incrementally per constitutional Principle VII.
