# API Contract: IVFFLAT Index

**Feature**: 001-ivfflat-index
**Date**: 2025-11-07
**Phase**: 1 - Design

## Overview

This document specifies the API contract for IVF index integration. **No new endpoints are required** - IVF is fully supported through existing REST API endpoints by specifying `index_type="ivf"` in library creation.

## Endpoint Changes

### Summary

| Endpoint | Method | Change | IVF-Specific |
|----------|--------|--------|--------------|
| `/libraries` | POST | None | ✅ Accepts index_type="ivf" |
| `/libraries/{id}` | PUT | None | ✅ Can update index_config.nprobe |
| `/libraries/{id}/build-index` | POST | None | ✅ Triggers K-means clustering |
| `/libraries/{id}/query` | POST | None | ✅ Uses nprobe for search |
| All other endpoints | * | None | Work transparently with IVF |

**Key Principle**: SDK and API transparency - users swap index types by only changing the `index_type` parameter.

## Existing Endpoints with IVF Support

### 1. Create Library

**Endpoint**: `POST /libraries`

**Request Body** (existing schema with IVF example):

```json
{
  "name": "my_ivf_library",
  "index_type": "ivf",
  "index_config": {
    "nlist": 100,
    "nprobe": 10,
    "metric": "cosine"
  },
  "metadata": {}
}
```

**Response**: `201 Created`

```json
{
  "id": "uuid-string",
  "name": "my_ivf_library",
  "index_type": "ivf",
  "index_config": {
    "nlist": 100,
    "nprobe": 10,
    "metric": "cosine"
  },
  "document_ids": [],
  "metadata": {},
  "created_at": "2025-11-07T12:00:00Z",
  "updated_at": "2025-11-07T12:00:00Z"
}
```

**Validation**:
- `index_type` must be "flat", "hnsw", or "ivf"
- `nlist` (if provided) must be positive integer
- `nprobe` (if provided) must be positive integer
- `metric` (if provided) must be "cosine", "euclidean", or "dot_product"

**Errors**:
```json
{
  "detail": "Invalid index_type: must be 'flat', 'hnsw', or 'ivf'"
}
```

### 2. Update Library

**Endpoint**: `PUT /libraries/{library_id}`

**Request Body** (updating nprobe):

```json
{
  "index_config": {
    "nprobe": 20
  }
}
```

**Response**: `200 OK`

```json
{
  "id": "uuid-string",
  "name": "my_ivf_library",
  "index_type": "ivf",
  "index_config": {
    "nlist": 100,
    "nprobe": 20,
    "metric": "cosine"
  },
  "document_ids": [...],
  "metadata": {},
  "created_at": "2025-11-07T12:00:00Z",
  "updated_at": "2025-11-07T12:05:00Z"
}
```

**IVF-Specific Behavior**:
- Updating `nprobe` takes effect immediately on next search
- Updating `nlist` requires rebuild (user must call build-index)
- Updating `metric` requires rebuild

### 3. Build Index

**Endpoint**: `POST /libraries/{library_id}/build-index`

**Request Body**: None (empty body or `{}`)

**Response**: `200 OK`

```json
{
  "library_id": "uuid-string",
  "total_vectors": 10000,
  "dimension": 384,
  "index_type": "ivf",
  "index_config": {
    "nlist": 100,
    "nprobe": 10,
    "metric": "cosine"
  }
}
```

**IVF-Specific Behavior**:
- Triggers K-means clustering with nlist clusters
- Computes centroids
- Assigns all vectors to nearest clusters
- Returns actual nlist used (may be < requested if nlist > num_vectors)

**Errors**:
```json
{
  "detail": "Cannot build index: library has no vectors"
}
```

### 4. Search (Query)

**Endpoint**: `POST /libraries/{library_id}/query`

**Request Body** (existing schema):

```json
{
  "embedding": [0.1, 0.2, ..., 0.384],
  "k": 10,
  "filters": {
    "metadata_filters": {...}
  }
}
```

**Response**: `200 OK`

```json
{
  "results": [
    {
      "id": "chunk-uuid-1",
      "text": "Example text",
      "score": 0.95,
      "metadata": {...},
      "document_id": "doc-uuid",
      "created_at": "2025-11-07T12:00:00Z"
    },
    ...
  ],
  "total": 10
}
```

**IVF-Specific Behavior**:
- If index not built, triggers lazy build before search
- Identifies nprobe nearest clusters to query vector
- Searches only within those clusters
- Returns top k results across all searched clusters
- Filters applied post-clustering (consistent with FLAT)

**Performance Notes**:
- First search may have higher latency (lazy build)
- Subsequent searches fast (cluster-pruned)
- Result ordering by similarity (descending)

### 5. Add Chunks

**Endpoint**: `POST /documents/{document_id}/chunks`

**Request Body** (existing schema):

```json
{
  "text": "Example text",
  "embedding": [0.1, 0.2, ..., 0.384],
  "metadata": {}
}
```

**Response**: `201 Created` (chunk object)

**IVF-Specific Behavior**:
- If index built: assigns chunk to nearest existing cluster (no rebuild)
- If index not built: queues chunk for next clustering
- Transparent - no API changes

### 6. Bulk Add Chunks

**Endpoint**: `POST /documents/{document_id}/chunks/batch`

**Request Body** (existing schema):

```json
{
  "chunks": [
    {
      "text": "Text 1",
      "embedding": [0.1, ...],
      "metadata": {}
    },
    {
      "text": "Text 2",
      "embedding": [0.2, ...],
      "metadata": {}
    }
  ]
}
```

**Response**: `201 Created` (list of chunk objects)

**IVF-Specific Behavior**:
- Same as single add (assigns to nearest clusters if built)
- No automatic rebuild after bulk add
- Users can call build-index after bulk operations for reoptimization

### 7. Update/Delete Chunks

**Endpoints**:
- `PUT /chunks/{chunk_id}`
- `DELETE /chunks/{chunk_id}`

**IVF-Specific Behavior**:
- Update: removes from old cluster, adds to nearest new cluster
- Delete: removes from cluster
- Empty clusters preserved (not merged)
- Transparent - no API changes

## OpenAPI Schema Extensions

### IndexType Enum

**Before**:
```yaml
IndexType:
  type: string
  enum:
    - flat
    - hnsw
```

**After**:
```yaml
IndexType:
  type: string
  enum:
    - flat
    - hnsw
    - ivf  # NEW
```

### IndexConfig Schema (Dynamic)

**For IVF Index**:
```yaml
IndexConfig:
  type: object
  properties:
    nlist:
      type: integer
      minimum: 1
      description: Number of clusters for vector partitioning
      example: 100
    nprobe:
      type: integer
      minimum: 1
      description: Number of clusters to search
      example: 10
    metric:
      type: string
      enum: [cosine, euclidean, dot_product]
      description: Distance metric
      default: cosine
```

**OpenAPI Documentation Update**:

```yaml
/libraries:
  post:
    summary: Create a new library
    requestBody:
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/CreateLibraryRequest'
          examples:
            flat_index:
              summary: FLAT index library
              value:
                name: "flat_library"
                index_type: "flat"
                index_config:
                  metric: "cosine"
            ivf_index:
              summary: IVF index library
              value:
                name: "ivf_library"
                index_type: "ivf"
                index_config:
                  nlist: 100
                  nprobe: 10
                  metric: "cosine"
```

## SDK Contract (Zero Changes Required)

### Python SDK Usage

**Creating IVF Library**:
```python
from my_vector_db.sdk import VectorDBClient

client = VectorDBClient(base_url="http://localhost:8000")

# Create IVF library - uses existing create_library method
library = client.create_library(
    name="my_ivf_library",
    index_type="ivf",  # Only parameter that changes
    index_config={
        "nlist": 100,
        "nprobe": 10,
        "metric": "cosine"
    }
)
```

**All Other Operations Unchanged**:
```python
# Search - identical to FLAT index
results = client.search(
    library_id=library.id,
    embedding=[0.1, 0.2, ...],
    k=10
)

# Add chunks - identical to FLAT index
client.add_chunks(
    document_id=doc.id,
    chunks=[{"text": "...", "embedding": [...]}]
)

# Build index - identical to FLAT index
client.build_index(library_id=library.id)

# Update nprobe - uses existing update_library method
client.update_library(
    library_id=library.id,
    index_config={"nprobe": 20}
)
```

**SDK Transparency Verification**:
- ✅ No new SDK methods
- ✅ No new parameters on existing methods
- ✅ Users swap index types by changing one parameter
- ✅ All functionality accessible through existing API

## Error Responses

### IVF-Specific Errors

**Invalid nlist**:
```json
{
  "detail": "index_config.nlist must be a positive integer"
}
```

**Invalid nprobe**:
```json
{
  "detail": "index_config.nprobe must be a positive integer"
}
```

**nprobe > nlist (Warning, not error)**:
- Behavior: Clamp nprobe to nlist
- Response: Success with clamped value
- No error thrown (graceful handling)

**Build with no vectors**:
```json
{
  "detail": "Cannot build IVF index: library contains no vectors"
}
```

**Dimension mismatch** (inherited):
```json
{
  "detail": "Vector dimension 512 doesn't match index dimension 384"
}
```

## Backward Compatibility

### Existing Libraries

**Impact**: None - existing FLAT and HNSW libraries unaffected

**Migration Path** (if users want to convert):
1. Create new IVF library with same name_v2
2. Copy chunks from old library to new library
3. Build index on new library
4. Test search quality
5. Switch application to new library
6. Delete old library

**No automatic migration** - users explicitly opt-in to IVF

### API Versioning

**Not Required**: Changes are additive only
- New enum value (IndexType.IVF)
- New config parameters (nlist, nprobe)
- Existing endpoints unchanged
- Backward compatible

## Testing Contract

### Contract Test Requirements

**test_ivf_contract.py**:

```python
def test_create_ivf_library_via_api():
    """POST /libraries with index_type='ivf' creates IVF library"""
    response = client.post("/libraries", json={
        "name": "test_ivf",
        "index_type": "ivf",
        "index_config": {"nlist": 50, "nprobe": 5}
    })
    assert response.status_code == 201
    data = response.json()
    assert data["index_type"] == "ivf"
    assert data["index_config"]["nlist"] == 50
    assert data["index_config"]["nprobe"] == 5

def test_search_triggers_lazy_build():
    """POST /query on unbuilt IVF library triggers clustering"""
    # Create library and add vectors
    # ... setup code ...

    # Search without explicit build
    response = client.post(f"/libraries/{lib_id}/query", json={
        "embedding": [0.1] * 384,
        "k": 5
    })
    assert response.status_code == 200
    # Verify results returned (lazy build succeeded)
    assert len(response.json()["results"]) > 0

def test_update_nprobe_affects_search():
    """PUT /libraries updates nprobe and changes search behavior"""
    # Create IVF library with nprobe=1
    # Search and record results
    # Update nprobe=10
    response = client.put(f"/libraries/{lib_id}", json={
        "index_config": {"nprobe": 10}
    })
    assert response.status_code == 200
    # Search again - may get different (better) results
```

## Summary

**API Contract Status**:
- ✅ No new endpoints required
- ✅ Existing endpoints handle IVF transparently
- ✅ OpenAPI schema extended (IndexType enum + IVF examples)
- ✅ SDK contract unchanged (zero modifications)
- ✅ Backward compatible (additive changes only)
- ✅ Error handling defined
- ✅ Contract tests specified

**Compliance**:
- ✅ RESTful principles maintained
- ✅ FastAPI auto-documentation support
- ✅ Type safety via Pydantic validation
- ✅ Constitutional requirement (Principle V) satisfied

**Ready for quickstart guide generation**.
