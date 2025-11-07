# Data Model: IVFFLAT Index

**Feature**: 001-ivfflat-index
**Date**: 2025-11-07
**Phase**: 1 - Design

## Overview

This document defines the data structures and relationships for the IVF index implementation. The IVF index extends the existing VectorIndex interface and integrates with the existing domain model through minimal changes.

## Core Entities

### 1. IVFIndex (New Class)

**Location**: `src/my_vector_db/indexes/ivf.py`

**Purpose**: Implements VectorIndex interface with cluster-based approximate nearest neighbor search

**Attributes**:

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `dimension` | `int` | Vector dimensionality | Inherited from VectorIndex, > 0 |
| `config` | `Dict[str, Any]` | Index configuration (nlist, nprobe, metric) | Inherited from VectorIndex |
| `_vectors` | `Dict[UUID, np.ndarray]` | All vectors indexed by ID | Private, in-memory |
| `_clusters` | `Dict[int, List[Tuple[UUID, np.ndarray]]]` | Cluster assignments: cluster_idx → [(vector_id, vector)] | Private, 0 to nlist-1 keys |
| `_centroids` | `Optional[np.ndarray]` | Cluster centroids, shape (nlist, dimension) | Private, None until built |
| `_kmeans` | `Optional[KMeans]` | Scikit-learn K-means instance | Private, None until built |
| `_is_built` | `bool` | Whether clustering has been performed | Private, default False |

**State Transitions**:

```
[Created] --add vectors--> [Pending Build]
                                |
                                | build() or search()
                                v
[Pending Build] ------------> [Built]
        ^                        |
        |  rebuild() or clear()  |
        +------------------------+
```

**Configuration Schema** (index_config dict):

| Parameter | Type | Default | Description | Validation |
|-----------|------|---------|-------------|------------|
| `nlist` | `int` | `sqrt(n)` | Number of clusters | Must be > 0, ≤ num vectors |
| `nprobe` | `int` | `1` | Number of clusters to search | Must be > 0, ≤ nlist |
| `metric` | `str` | `"cosine"` | Distance metric | Must be "cosine", "euclidean", or "dot_product" |

**Methods** (Implementing VectorIndex Interface):

```python
class IVFIndex(VectorIndex):
    def __init__(self, dimension: int, config: Optional[Dict[str, Any]] = None) -> None
    def add(self, vector_id: UUID, vector: List[float]) -> None
    def bulk_add(self, vectors: List[Tuple[UUID, List[float]]]) -> None
    def search(self, query_vector: List[float], k: int) -> List[Tuple[UUID, float]]
    def update(self, vector_id: UUID, vector: List[float]) -> None
    def delete(self, vector_id: UUID) -> None
    def clear(self) -> None

    # IVF-specific methods (not in interface)
    def build(self) -> None  # Explicit clustering trigger
    def _build_clusters(self) -> None  # Internal clustering logic
    def _find_nearest_cluster(self, vector: np.ndarray) -> int
    def _get_nprobe_nearest_clusters(self, query_vector: np.ndarray) -> List[int]
```

### 2. IndexType Enum (Modified)

**Location**: `src/my_vector_db/domain/models.py`

**Change**: Add IVF variant

**Before**:
```python
class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
```

**After**:
```python
class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"  # NEW
```

**Impact**: Backward compatible - existing values unchanged

### 3. Library Model (No Structural Changes)

**Location**: `src/my_vector_db/domain/models.py`

**Existing Structure** (already supports IVF):

```python
class Library(BaseModel):
    id: UUID
    name: str
    document_ids: List[UUID]
    metadata: Dict[str, Any]
    index_type: IndexType  # Can now be IndexType.IVF
    index_config: Dict[str, Any]  # Contains nlist, nprobe, metric
    created_at: datetime
    updated_at: datetime
```

**IVF Example**:
```python
library = Library(
    name="ivf_library",
    index_type=IndexType.IVF,
    index_config={
        "nlist": 100,
        "nprobe": 10,
        "metric": "cosine"
    }
)
```

## Cluster Data Structure

### Cluster Representation

**Internal Structure**: Dictionary mapping cluster index to list of (vector_id, vector) tuples

```python
_clusters: Dict[int, List[Tuple[UUID, np.ndarray]]] = {
    0: [(uuid1, array1), (uuid2, array2), ...],  # Cluster 0
    1: [(uuid5, array5), ...],                   # Cluster 1
    ...
    99: [(uuid99, array99), ...]                 # Cluster 99 (for nlist=100)
}
```

**Properties**:
- Keys: 0 to (nlist - 1)
- Values: Variable-length lists (empty if no vectors assigned)
- Total vectors across all clusters = total vectors in index
- Each vector appears in exactly one cluster

### Centroid Representation

**Structure**: NumPy array of shape (nlist, dimension)

```python
_centroids: np.ndarray  # Shape: (100, 384) for nlist=100, dimension=384

# Each row is a cluster centroid
_centroids[0]  # Centroid of cluster 0, shape (384,)
_centroids[1]  # Centroid of cluster 1, shape (384,)
```

**Computation**: Set by scikit-learn K-means (`kmeans.cluster_centers_`)

## Relationships

### IVF Index to Vector Mapping

```
IVFIndex
├── _vectors: {UUID → np.ndarray}  # All vectors (fast lookup)
└── _clusters: {int → [(UUID, np.ndarray)]}  # Clustered vectors (fast search)

Invariant: keys(_vectors) == {uuid for cluster in _clusters for (uuid, _) in cluster}
```

### Library to IVF Index (Runtime)

```
Library (domain model)
    ↓ index_type="ivf", index_config={nlist, nprobe}
LibraryService
    ↓ _create_index() factory
IVFIndex instance (runtime)
```

**Persistence**: Index state not persisted; recreated on load via:
1. Load Library model (has index_type, index_config)
2. Factory creates IVFIndex instance
3. Vectors loaded from chunks
4. Clustering triggered on first operation (lazy build)

## Data Validation

### Index Config Validation

**Validation Rules** (enforced in IVFIndex.__init__ or build):

```python
def validate_config(self):
    nlist = self.config.get("nlist")
    nprobe = self.config.get("nprobe")
    metric = self.config.get("metric", "cosine")

    # nlist validation
    if nlist is not None:
        if not isinstance(nlist, int) or nlist <= 0:
            raise ValueError("nlist must be positive integer")

    # nprobe validation
    if nprobe is not None:
        if not isinstance(nprobe, int) or nprobe <= 0:
            raise ValueError("nprobe must be positive integer")
        if nlist and nprobe > nlist:
            # Allow but warn or clamp
            nprobe = nlist

    # metric validation
    if metric not in ["cosine", "euclidean", "dot_product"]:
        raise ValueError(f"Unknown metric: {metric}")
```

### Vector Dimension Validation

**Inherited from VectorIndex**:

```python
def add(self, vector_id: UUID, vector: List[float]) -> None:
    if len(vector) != self.dimension:
        raise ValueError(
            f"Vector dimension {len(vector)} doesn't match index dimension {self.dimension}"
        )
```

## Type Definitions

### Python Type Hints

```python
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
import numpy as np
from sklearn.cluster import KMeans

# Cluster ID type
ClusterIdx = int

# Cluster assignment type
ClusterAssignment = Dict[ClusterIdx, List[Tuple[UUID, np.ndarray]]]

# Search result type (inherited)
SearchResult = List[Tuple[UUID, float]]

# Config type (inherited)
IndexConfig = Dict[str, Any]
```

### Pydantic Models (No New Models)

**Existing models used**:
- `Library`: Contains index_type and index_config
- `BuildIndexResult`: Returned by build_index endpoint (already supports IVF)
- `SearchResponse`: Returned by search endpoint (already supports IVF)

**No new Pydantic models needed** - IVF parameters fit in existing Dict[str, Any] config fields

## Serialization & Persistence

### Library Model Serialization (Existing)

```python
# Library with IVF config serializes as JSON
library_json = {
    "id": "uuid-string",
    "name": "my_library",
    "index_type": "ivf",  # String value from IndexType enum
    "index_config": {
        "nlist": 100,
        "nprobe": 10,
        "metric": "cosine"
    },
    "document_ids": [...],
    "metadata": {},
    "created_at": "2025-11-07T...",
    "updated_at": "2025-11-07T..."
}
```

### Index State (Not Persisted)

**IVF index state is ephemeral**:
- `_clusters`: Rebuilt on load via clustering
- `_centroids`: Recomputed via K-means
- `_kmeans`: Recreated on load
- `_is_built`: Starts False, triggers rebuild on first operation

**Rationale**:
- Consistent with FLAT index (no special persistence)
- Clustering is fast enough for typical dataset sizes
- Simpler implementation (no centroid serialization logic)
- Future enhancement: persist centroids if rebuild becomes expensive

## Edge Cases & Invariants

### Invariants

1. **Vector presence**: Every vector_id in `_vectors` appears in exactly one cluster in `_clusters`
2. **Cluster range**: All cluster indices in `_clusters` are in range [0, nlist-1]
3. **Centroid shape**: If `_centroids` exists, shape is (nlist, dimension)
4. **Build flag**: `_is_built == True` implies `_clusters`, `_centroids`, and `_kmeans` are not None

### Edge Case Handling

| Edge Case | Behavior | Data State |
|-----------|----------|------------|
| nlist > num_vectors | K-means creates min(nlist, n) clusters | Fewer clusters than requested |
| Empty cluster after delete | Cluster exists but empty list | `_clusters[idx] == []` |
| Search with 0 vectors | Return empty results | No clustering attempted |
| nprobe > nlist | Clamp to nlist (search all clusters) | Effective nprobe = nlist |
| Identical vectors | K-means may create degenerate clusters | All clusters valid, some may overlap |

## Summary

**New Entities**:
- ✅ IVFIndex class implementing VectorIndex interface
- ✅ Cluster data structure (Dict[int, List[Tuple[UUID, np.ndarray]]])
- ✅ Centroid array (np.ndarray shape (nlist, dimension))

**Modified Entities**:
- ✅ IndexType enum (add IVF variant)

**Unchanged Entities**:
- ✅ Library model (already supports arbitrary index_config)
- ✅ VectorIndex interface (IVF implements existing contract)
- ✅ API schemas (no new request/response types)

**Data Validation**:
- ✅ Config validation (nlist, nprobe, metric)
- ✅ Dimension validation (inherited)
- ✅ Cluster invariants enforced

**Ready for Phase 1 completion**: Contracts and quickstart guide.
