# Research & Design Decisions: IVFFLAT Index

**Feature**: 001-ivfflat-index
**Date**: 2025-11-07
**Phase**: 0 - Research

## Overview

This document captures the research findings and design decisions for implementing an Inverted File (IVF) index with FLAT storage in the vector database. The IVF index enables approximate nearest neighbor search through vector space partitioning.

## Key Design Decisions

### 1. Clustering Algorithm Selection

**Decision**: Use scikit-learn's K-means clustering (KMeans class)

**Rationale**:
- Well-tested, production-ready implementation
- Handles n-dimensional vectors efficiently
- Supports configurable cluster counts (n_clusters parameter maps to nlist)
- Pure Python/numpy implementation (no C++ dependencies for simple cases)
- Widely documented with clear API
- Fits constitutional constraint (allowed dependency)

**Alternatives Considered**:
- **Custom K-means from scratch**: Rejected due to unnecessary complexity. scikit-learn implementation is battle-tested and optimized. Constitutional requirement is "no third-party indexing libraries", not "no ML libraries". K-means is a building block, not a complete vector index.
- **DBSCAN clustering**: Rejected - cannot guarantee fixed cluster count (nlist parameter), produces variable clusters based on density
- **Hierarchical clustering**: Rejected - O(n²) complexity too expensive for large datasets, incompatible with incremental updates

**Implementation Details**:
```python
from sklearn.cluster import KMeans

# Clustering configuration
kmeans = KMeans(
    n_clusters=nlist,  # User-configured cluster count
    random_state=42,   # Reproducible clustering
    n_init=10,         # Multiple initializations for stability
    max_iter=300       # Sufficient for convergence
)
```

### 2. Index Building Strategy

**Decision**: Lazy building with explicit override - cluster on first search OR on explicit build_index call

**Rationale**:
- **User flexibility**: Power users can pre-build for predictable performance; casual users get automatic behavior
- **Resource efficiency**: Avoid expensive clustering on empty/small libraries
- **Consistency with FLAT**: FLAT index doesn't require explicit build; IVF shouldn't either for basic use
- **Clear from clarification**: User specified this exact pattern in clarification session

**Alternatives Considered**:
- **Mandatory build_index call**: Rejected - adds API friction, inconsistent with FLAT index UX
- **Always lazy (no explicit build option)**: Rejected - removes control for performance-sensitive users
- **Automatic on add/bulk_add**: Rejected - expensive repeated clustering, breaks incremental update pattern

**Implementation Details**:
```python
class IVFIndex:
    def __init__(self, ...):
        self._is_built = False  # Track build state
        self._needs_rebuild = False  # Track if data changed since build

    def search(self, query_vector, k):
        if not self._is_built:
            self._build_clusters()  # Lazy build on first search
        # ... proceed with search

    def build(self):  # Explicit build (called by build_index API)
        self._build_clusters()
        self._is_built = True
```

### 3. Parameter Naming Convention

**Decision**: Use "nlist" and "nprobe" for cluster count and probe count parameters

**Rationale**:
- Industry standard from FAISS (Facebook AI Similarity Search)
- Users migrating from other vector DBs recognize these names immediately
- Clear semantic meaning: nlist = number of lists (clusters), nprobe = number to probe (search)
- Confirmed in clarification session
- Short, memorable, widely documented in vector search literature

**Alternatives Considered**:
- **cluster_count / probe_count**: Rejected - verbose, not industry standard
- **n_clusters / n_search**: Rejected - partially standard (n_clusters from sklearn) but n_search is non-standard
- **num_clusters / num_probes**: Rejected - longer without added clarity

**Usage Example**:
```python
client.create_library(
    name="my_lib",
    index_type="ivf",
    index_config={
        "nlist": 100,      # 100 clusters
        "nprobe": 10,      # Search 10 nearest clusters
        "metric": "cosine" # Distance metric
    }
)
```

### 4. Centroid Update Strategy

**Decision**: Update centroids only on explicit build/rebuild, not on incremental operations

**Rationale**:
- **Performance**: Centroid recomputation is expensive (O(n) per cluster)
- **Stability**: Prevents cluster drift during incremental adds
- **Predictability**: Users know when clustering occurs (explicit build or first search)
- **Simplicity**: Incremental adds use nearest-cluster assignment only

**Alternatives Considered**:
- **Incremental centroid updates**: Rejected - complex online learning, potential instability, marginal accuracy gain
- **Periodic automatic recomputation**: Rejected - unpredictable performance impact, hard to tune threshold
- **Never update (build once only)**: Rejected - requires rebuild even for legitimate cluster optimization needs

**Implementation Pattern**:
```python
def add(self, vector_id, vector):
    if self._is_built:
        # Assign to nearest existing cluster (no centroid update)
        cluster_idx = self._find_nearest_cluster(vector)
        self._clusters[cluster_idx].append((vector_id, vector))
    else:
        # Queue for next clustering
        self._pending_vectors.append((vector_id, vector))

def build(self):
    # Full K-means clustering - updates all centroids
    all_vectors = self._get_all_vectors()
    labels = self._kmeans.fit_predict(all_vectors)
    self._centroids = self._kmeans.cluster_centers_
    # ... assign vectors to clusters
```

### 5. Empty Cluster Handling

**Decision**: Preserve empty clusters in structure but skip during search

**Rationale**:
- **Consistency**: Maintain nlist cluster count even if some are empty
- **Performance**: Avoid wasted distance computations for empty clusters
- **Simplicity**: No complex cluster merging or reindexing logic
- **Recovery**: Clusters can be repopulated on rebuild or future adds

**Alternatives Considered**:
- **Merge empty clusters**: Rejected - complex logic, changes nlist semantically, invalidates nprobe calculations
- **Error on empty clusters**: Rejected - normal scenario (deletions, small datasets), shouldn't fail
- **Remove from structure**: Rejected - breaks nlist invariant, complicates cluster indexing

**Implementation**:
```python
def search(self, query_vector, k):
    # Find nprobe nearest non-empty clusters
    cluster_similarities = []
    for idx, centroid in enumerate(self._centroids):
        if len(self._clusters[idx]) > 0:  # Skip empty
            sim = self._compute_similarity(query_vector, centroid)
            cluster_similarities.append((idx, sim))

    # Sort and take top nprobe
    top_clusters = sorted(cluster_similarities, key=lambda x: x[1], reverse=True)[:nprobe]
    # ... search within selected clusters
```

### 6. Thread Safety Approach

**Decision**: Inherit service-layer RLock synchronization from existing pattern

**Rationale**:
- **Consistency**: Matches FLAT index pattern
- **Proven approach**: Existing thread safety works well
- **Service-layer protection**: LibraryService already manages index access with RLock
- **Simplicity**: No index-specific locking needed

**Existing Pattern**:
```python
# In library_service.py
class LibraryService:
    def __init__(self):
        self._lock = RLock()

    def search(self, library_id, query_vector, k):
        with self._lock:
            index = self._get_index(library_id)
            return index.search(query_vector, k)  # Protected
```

**IVF Implementation**:
- IVF index methods (add, search, update, delete) are NOT thread-safe in isolation
- Thread safety provided by service layer lock (same as FLAT)
- Constitutional requirement satisfied at correct layer

### 7. Default Values Strategy

**Decision**:
- **nlist default**: `sqrt(n)` where n = number of vectors (computed at build time)
- **nprobe default**: 1 (fastest, lowest recall)
- **metric default**: "cosine" (inherited from config or FLAT default)

**Rationale**:
- **sqrt(n) for nlist**: Rule of thumb from vector search literature, balances cluster size vs cluster count
- **nprobe=1**: Conservative default favors speed; users explicitly opt into higher accuracy
- **Metric inheritance**: Consistency with existing index behavior

**Implementation**:
```python
def _compute_default_nlist(self, num_vectors):
    if num_vectors < 10:
        return 1  # Don't cluster tiny datasets
    return max(1, int(math.sqrt(num_vectors)))

def build(self):
    if "nlist" not in self.config:
        self.config["nlist"] = self._compute_default_nlist(len(vectors))
    if "nprobe" not in self.config:
        self.config["nprobe"] = 1
```

## Integration Points

### 1. Domain Model Update

**File**: `src/my_vector_db/domain/models.py`

**Change**:
```python
class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"  # NEW
```

**Impact**: Minimal - enum addition only, backward compatible

### 2. Index Factory Pattern

**File**: `src/my_vector_db/services/library_service.py`

**Existing Pattern**:
```python
def _create_index(self, index_type, dimension, config):
    if index_type == IndexType.FLAT:
        return FlatIndex(dimension, config)
    elif index_type == IndexType.HNSW:
        raise NotImplementedError("HNSW not yet implemented")
```

**Update**:
```python
def _create_index(self, index_type, dimension, config):
    if index_type == IndexType.FLAT:
        return FlatIndex(dimension, config)
    elif index_type == IndexType.IVF:
        return IVFIndex(dimension, config)  # NEW
    elif index_type == IndexType.HNSW:
        raise NotImplementedError("HNSW not yet implemented")
```

### 3. Persistence (No Changes)

**Files**: `src/my_vector_db/infrastructure/storage.py`

**Decision**: IVF index serializes via existing Pydantic model serialization

**Rationale**:
- Library model already stores index_type and index_config
- Index state (clusters, centroids) reconstructed on load via build
- Consistent with FLAT index (no special persistence logic)

## Performance Considerations

### Time Complexity

| Operation | FLAT Index | IVF Index | Notes |
|-----------|------------|-----------|-------|
| Add | O(1) | O(1) if built, O(1) queued | Nearest cluster assignment |
| Build | N/A | O(n·d·k·i) | K-means: n vectors, d dims, k clusters, i iterations |
| Search | O(n·d) | O(c·d + m·d) | c = nprobe, m = avg cluster size |
| Update | O(1) | O(1) | Reassign to nearest cluster |
| Delete | O(1) | O(1) | Remove from cluster |

**Expected Speedup**:
- For n=100,000, nlist=100, nprobe=10: ~10x faster search (searches 10% of vectors)
- Trade-off: 80-95% recall vs 100% recall (FLAT)

### Space Complexity

**IVF Storage**:
- Vectors: O(n·d) - same as FLAT
- Centroids: O(k·d) where k=nlist - minimal overhead
- Cluster mappings: O(n) - small overhead (vector ID → cluster ID)

**Total**: O(n·d + k·d) ≈ O(n·d) since k << n typically

## Testing Strategy

### Unit Tests (test_ivf_index.py)

1. **Clustering Tests**:
   - Verify K-means creates nlist clusters
   - Test default nlist calculation (sqrt(n))
   - Edge case: nlist > n (creates min(nlist, n) clusters)
   - Edge case: identical vectors (degenerate clustering)

2. **Search Tests**:
   - Verify nprobe clusters searched
   - Test lazy building on first search
   - Test empty cluster skipping
   - Verify result ordering (top k by similarity)

3. **Maintenance Tests**:
   - Add vector after clustering (nearest cluster assignment)
   - Update vector (reassignment)
   - Delete vector (cluster removal)

4. **VectorIndex Interface Compliance**:
   - All abstract methods implemented
   - Signature matching
   - Exception contracts honored

### Integration Tests (test_ivf_search.py)

1. **End-to-End Workflow**:
   - Create library with IVF
   - Add vectors
   - Search without build (lazy)
   - Explicit rebuild
   - Search after rebuild

2. **SDK Transparency**:
   - Create IVF library via SDK create_library
   - Verify existing search() method works
   - Update nprobe via update_library
   - Confirm zero SDK changes needed

3. **Recall Measurement**:
   - Compare IVF vs FLAT results
   - Measure recall at different nprobe values
   - Verify 80%+ recall at nprobe=10% of nlist

### Contract Tests (test_ivf_contract.py)

1. **VectorIndex Interface**:
   - Signature tests for all abstract methods
   - Type hint validation
   - Exception behavior (ValueError, KeyError)

2. **Config Validation**:
   - Missing nlist/nprobe defaults applied
   - Invalid nlist/nprobe rejected
   - Metric support (cosine, euclidean, dot_product)

## Open Questions & Future Work

### Resolved During Research
- ✅ Clustering algorithm: K-means (scikit-learn)
- ✅ Parameter names: nlist, nprobe
- ✅ Building strategy: Lazy with explicit override
- ✅ SDK changes: None needed (transparency achieved)

### Future Enhancements (Out of Scope)

1. **Product Quantization (IVFPQ)**:
   - Compress vectors within clusters
   - Reduce memory footprint
   - Trade-off: Lower recall, higher speedup

2. **Adaptive nprobe**:
   - Auto-tune nprobe based on query distribution
   - Maintain target recall percentage

3. **Incremental Clustering**:
   - Online centroid updates for high-write workloads
   - Complexity vs benefit needs analysis

4. **GPU Acceleration**:
   - Offload K-means and search to GPU
   - Requires CuPy or similar (breaks pure Python constraint)

## Conclusion

Research phase complete. All design decisions made:
- ✅ Clustering approach defined (scikit-learn K-means)
- ✅ Building strategy clarified (lazy + explicit)
- ✅ Parameter naming standardized (nlist, nprobe)
- ✅ Integration points identified (minimal changes)
- ✅ Testing strategy outlined (unit, integration, contract)
- ✅ Constitutional compliance verified

**Ready for Phase 1**: Data model and contracts generation.
