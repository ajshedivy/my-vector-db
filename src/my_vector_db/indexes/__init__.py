"""
Vector index implementations for the Vector Database.

This package provides different vector index algorithms, each with different
tradeoffs between speed, memory, and accuracy:

- FlatIndex: Exact nearest neighbor search (baseline)
- HNSWIndex: Hierarchical Navigable Small World graphs (fast approximate)

All indexes implement the VectorIndex abstract base class.

Example:
    from my_vector_db.indexes import FlatIndex, HNSWIndex, VectorIndex

    index = HNSWIndex(dimension=384, config={"M": 16, "ef_construction": 200})
"""

from my_vector_db.indexes.base import VectorIndex
from my_vector_db.indexes.flat import FlatIndex
from my_vector_db.indexes.hnsw import HNSWIndex

__all__ = [
    "VectorIndex",
    "FlatIndex",
    "HNSWIndex",
]
