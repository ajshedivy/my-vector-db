"""
Filters package - Metadata filtering for vector search.

Provides filter evaluation logic for post-filtering search results.
"""

from my_vector_db.filters.evaluator import (
    evaluate_filter_group,
    evaluate_metadata_filter,
    evaluate_search_filters,
)

__all__ = [
    "evaluate_metadata_filter",
    "evaluate_filter_group",
    "evaluate_search_filters",
]
