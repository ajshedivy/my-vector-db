# Feature Specification: IVFFLAT Index Implementation

**Feature Branch**: `001-ivfflat-index`
**Created**: 2025-11-07
**Status**: Draft
**Input**: User description: "develop IVF index implementation using the indexing base classes provided: @src/my_vector_db/indexes/base.py. Inverted File (IVF) - This is the most basic indexing technique. It splits the whole data into several clusters using techniques like K-means clustering. Each vector of the database is assigned to a specific cluster. This structured arrangement of vectors allows the user to make the search queries way faster. When a new query comes, the system doesn't traverse the whole dataset. Instead, it identifies the nearest or most similar clusters and searches for the specific document within those clusters. This implementation will be IVFFLAT, which means that each cluster will store the raw vectors without any compression or quantization. Only allowed dependencies are numpy and scikit learn"

## Clarifications

### Session 2025-11-07

- Q: Should probe count be configured at library creation time in index_config, or passed as a search-time parameter like k? → A: Probe count stored in library index_config (set at library creation, updatable via update_library)
- Q: What parameter names should be used in index_config for cluster count and probe count? → A: nlist (cluster count) and nprobe (probe count) - industry-standard IVF naming
- Q: What string value should be used for IVF in the IndexType enum? → A: "ivf" (lowercase, following existing convention: "flat", "hnsw")
- Q: When should clustering be triggered - automatic after adds, or explicit build_index call? → A: Clustering triggered by explicit build_index call OR lazily on first search if build_index not called (user choice)
- Q: Does SDK need IVF-specific methods or should it work transparently through existing methods? → A: No SDK changes needed - IVF works transparently through existing create_library (with index_type="ivf") and search methods

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Index Creation with Clustering (Priority: P1)

A developer using the vector database needs to create an IVF index that organizes vectors into clusters for faster search. The system should cluster the vectors either when explicitly requested via build_index call, or automatically on first search if not yet clustered, making subsequent searches more efficient than exhaustive search.

**Why this priority**: This is the foundation of IVF functionality. Without the ability to create and populate clusters, the index cannot provide any value. This story delivers the core clustering mechanism that makes IVF searches faster.

**Independent Test**: Can be fully tested by creating a library with IVF index type, adding vectors, and verifying clustering occurs either (a) after explicit build_index call, or (b) automatically on first search. Delivers a functional clustered index structure.

**Acceptance Scenarios**:

1. **Given** a developer using the SDK, **When** creating a library with index_type="ivf" and index_config={"nlist": 100, "nprobe": 10}, **Then** library is created with IVF index using existing create_library method without IVF-specific SDK changes
2. **Given** an empty library configured with IVF index type and nlist (cluster count), **When** vectors are added and build_index is called, **Then** all vectors are assigned to appropriate clusters based on similarity
3. **Given** a library with vectors but no explicit build_index call, **When** first search is performed, **Then** clustering is triggered automatically before search execution
4. **Given** a library with existing clustered vectors, **When** the IVF index rebuild is called, **Then** the clustering is recomputed and vectors are reassigned to optimal clusters
5. **Given** vectors of varying dimensions, **When** attempting to add to IVF index, **Then** only vectors matching the index dimension are accepted and stored

---

### User Story 2 - Fast Approximate Search (Priority: P2)

A developer querying the vector database needs search results returned faster than exhaustive FLAT index search, accepting that results may be approximate rather than exact. The system should identify the most relevant clusters and only search within those clusters instead of comparing against all vectors.

**Why this priority**: This delivers the primary value proposition of IVF - faster search through cluster-based pruning. Without this, IVF would offer no advantage over FLAT index. This builds on the clustering from P1.

**Independent Test**: Can be fully tested by performing searches on an IVF index and comparing execution time and recall against FLAT index on the same data. Delivers measurable performance improvement.

**Acceptance Scenarios**:

1. **Given** a query vector and IVF index with populated clusters, **When** search is performed with specified k neighbors, **Then** the most relevant clusters (based on library's configured nprobe) are identified and searched first
2. **Given** a search request with k=10, **When** relevant clusters contain fewer than 10 vectors total, **Then** system returns all available matches from those clusters
3. **Given** a library with configured nprobe, **When** search is performed, **Then** exactly that many nearest clusters are searched

---

### User Story 3 - Configurable Search Quality (Priority: P3)

A developer needs to control the trade-off between search speed and accuracy by configuring how many clusters are searched (nprobe parameter) at library creation or by updating the library configuration. More clusters searched means higher accuracy but slower speed, while fewer clusters means faster but potentially less accurate results.

**Why this priority**: This provides fine-tuning capability for advanced users to optimize the speed/accuracy trade-off for their specific use case. It builds on the basic search functionality from P2.

**Independent Test**: Can be fully tested by creating libraries with different nprobe settings, performing searches, and measuring recall percentage and query time. Can also test updating nprobe via update_library and observing changed search behavior. Delivers configurable performance tuning.

**Acceptance Scenarios**:

1. **Given** a library with IVF index configured with nlist=100 and nprobe=1, **When** search is performed, **Then** only 1 cluster is searched
2. **Given** a library with IVF index configured with nlist=100 and nprobe=10, **When** search is performed, **Then** 10 nearest clusters are searched and results from all 10 are combined
3. **Given** a library with nprobe configured greater than nlist, **When** search is performed, **Then** all clusters are searched (equivalent to exhaustive search)
4. **Given** a library with existing nprobe configuration, **When** library is updated with new nprobe value, **Then** subsequent searches use the new nprobe

---

### User Story 4 - Index Maintenance Operations (Priority: P4)

A developer needs to update or delete vectors from the IVF index while maintaining cluster organization. The system should support vector updates (reassigning to appropriate cluster if needed) and deletions without requiring full index rebuild.

**Why this priority**: This enables practical use of IVF in dynamic environments where data changes over time. While less critical than basic search functionality, it's necessary for production use. Depends on clustering from P1.

**Independent Test**: Can be fully tested by performing add/update/delete operations on an IVF index and verifying vectors are correctly maintained in clusters. Delivers a production-ready mutable index.

**Acceptance Scenarios**:

1. **Given** a vector exists in an IVF cluster, **When** that vector is updated with new values, **Then** the vector is reassigned to the appropriate cluster for the new values
2. **Given** a vector exists in an IVF cluster, **When** that vector is deleted, **Then** it is removed from its cluster and no longer appears in search results
3. **Given** vectors are added after initial clustering, **When** new vectors are inserted, **Then** each new vector is assigned to its nearest cluster without full reclustering

---

### Edge Cases

- What happens when nlist (cluster count) is greater than the number of vectors? (Should create as many clusters as possible up to the vector count)
- What happens when a cluster becomes empty after deletions? (Empty cluster should remain in structure but not be searched)
- How does system handle query vectors that are very dissimilar to all cluster centroids? (Should still search nearest clusters even if similarity is low)
- What happens when nprobe is set to 0 or negative? (Should default to nprobe=1 or return validation error)
- How does index behave with very small datasets (e.g., fewer than 10 vectors)? (Should still function but may not show performance benefit over FLAT)
- What happens when all vectors are identical or extremely similar? (Clustering may converge poorly; should handle gracefully)
- What happens when search is called on a library with no vectors? (Should return empty results without attempting clustering)
- What happens when vectors are added after clustering? (New vectors assigned to nearest existing cluster; full reclustering only on explicit rebuild)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST partition vectors into a configurable number of clusters (nlist parameter) using a clustering algorithm when triggered by explicit build_index call OR automatically on first search if not yet clustered
- **FR-002**: System MUST store each vector in its assigned cluster without compression or quantization (FLAT storage)
- **FR-003**: System MUST identify the nearest N clusters to a query vector based on cluster centroid similarity
- **FR-004**: System MUST search only within the identified nearest clusters and return top k results
- **FR-005**: System MUST support configurable nprobe (probe count) in library index_config to control how many clusters are searched (set at creation, updatable via library update)
- **FR-006**: System MUST support all distance metrics available in the base index class (cosine, euclidean, dot product)
- **FR-007**: System MUST allow vectors to be added to the index after initial clustering, assigning them to nearest cluster
- **FR-008**: System MUST allow vectors to be updated, reassigning them to appropriate cluster if needed
- **FR-009**: System MUST allow vectors to be deleted from their assigned clusters
- **FR-010**: System MUST implement the VectorIndex abstract interface defined in base.py
- **FR-011**: System MUST maintain cluster centroids and update them during index operations
- **FR-012**: System MUST handle edge cases gracefully (empty clusters, nprobe exceeding nlist, etc.)
- **FR-013**: System MUST add IndexType.IVF = "ivf" to the domain model enum, allowing users to specify index_type="ivf" when creating libraries
- **FR-014**: System MUST ensure SDK transparency - IVF index works through existing SDK methods (create_library, search, add_chunks, etc.) without requiring IVF-specific API changes. Users swap index types by changing index_type parameter only.

### Key Entities

- **IVF Index**: A vector index that organizes vectors into clusters for faster approximate nearest neighbor search. Contains cluster configuration (nlist, nprobe, metric), cluster centroids, and mapping of vectors to clusters.
- **Cluster**: A logical grouping of similar vectors. Each cluster has a centroid (representative point) and contains a subset of the total vectors. Clusters enable search space pruning.
- **Cluster Centroid**: The representative point for a cluster, used to determine which clusters are nearest to a query vector. Updated during clustering and potentially during index maintenance.
- **nlist**: Number of clusters parameter stored in library index_config. Determines how many clusters the vector space is partitioned into during index build.
- **nprobe**: Library-level configuration parameter stored in index_config that controls search quality vs speed trade-off. Determines how many nearest clusters to search for all queries to that library. Can be updated via library update operation.

### Assumptions

- **Default nlist**: If not specified in index_config at library creation, system will use a reasonable default based on dataset size (e.g., sqrt(n) clusters for n vectors)
- **Default nprobe**: If not specified in library index_config at creation, system will default to nprobe=1 (fastest, least accurate). Users can change this via library update
- **Clustering Algorithm**: K-means clustering will be used as the standard algorithm for partitioning vectors (scikit-learn implementation)
- **Lazy Building**: Clustering is triggered either by explicit build_index call OR automatically on first search if index not yet built. This allows flexibility without requiring users to manually trigger index builds.
- **Re-clustering Threshold**: Full re-clustering is only required on explicit rebuild; incremental adds/updates/deletes use nearest cluster assignment
- **Empty Cluster Handling**: Empty clusters are preserved in structure but skipped during search to avoid wasted computation
- **nlist Bounds**: Minimum 1 cluster, maximum equal to number of vectors (though maximum is impractical)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Search operations on IVF index complete in less time than equivalent FLAT index search on datasets with 10,000+ vectors
- **SC-002**: Search recall (percentage of true nearest neighbors found) exceeds 80% when nprobe is set to 10% of nlist
- **SC-003**: Index build operation successfully clusters vectors and assigns all vectors to clusters without errors
- **SC-004**: Vector maintenance operations (add, update, delete) complete without requiring full index rebuild
- **SC-005**: System handles edge cases (empty clusters, extreme probe counts, small datasets) without crashes or data corruption
- **SC-006**: Index can be configured with different nlist and nprobe values to demonstrate speed/accuracy trade-offs
- **SC-007**: All VectorIndex interface methods are implemented and pass the same test suite as FLAT index (compatibility)
- **SC-008**: Users can swap between index types (flat, ivf) by only changing index_type parameter in create_library - no other SDK code changes required
