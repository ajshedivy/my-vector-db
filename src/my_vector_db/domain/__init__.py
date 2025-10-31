"""
Domain models for the Vector Database.

This package contains the core domain entities and value objects:
- Entities: Chunk, Document, Library
- Value Objects: BuildIndexResult
- Enums: IndexType, FilterOperator, LogicalOperator
- Filters: SearchFilters, SearchFiltersWithCallable, FilterGroup, MetadataFilter

Example:
    from my_vector_db.domain import Chunk, Document, Library, SearchFilters
"""

from my_vector_db.domain.models import (
    # Entities
    Chunk,
    Document,
    Library,
    # Value Objects
    BuildIndexResult,
    # Enums
    IndexType,
    FilterOperator,
    LogicalOperator,
    # Filter Models
    SearchFilters,
    SearchFiltersWithCallable,
    FilterGroup,
    MetadataFilter,
)

__all__ = [
    # Entities
    "Chunk",
    "Document",
    "Library",
    # Value Objects
    "BuildIndexResult",
    # Enums
    "IndexType",
    "FilterOperator",
    "LogicalOperator",
    # Filter Models
    "SearchFilters",
    "SearchFiltersWithCallable",
    "FilterGroup",
    "MetadataFilter",
]
