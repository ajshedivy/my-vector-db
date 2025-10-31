"""
Database storage layer for the Vector Database.

This package provides the in-memory storage implementation with persistence support.

Example:
    from my_vector_db.db import MyVectorDB

    db = MyVectorDB()
"""

from .my_vector_db import MyVectorDB

__all__ = ["MyVectorDB"]
