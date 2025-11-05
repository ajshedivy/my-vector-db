"""
Filter Evaluator - Core logic for evaluating metadata filters.

This module provides pure functions to evaluate filters against chunks.
Uses post-filtering approach: chunks are filtered after vector search.

Design principles:
- Pure functions (no side effects)
- Early returns for readability
- Graceful handling of missing fields
- Type-safe with proper hints
- Pythonic and simple
"""

from typing import Any, Callable

from my_vector_db.domain.models import (
    Chunk,
    FilterGroup,
    FilterOperator,
    LogicalOperator,
    MetadataFilter,
    SearchFilters,
)


def evaluate_metadata_filter(chunk: Chunk, filter: MetadataFilter) -> bool:
    """
    Evaluate a single metadata filter against a chunk.

    Args:
        chunk: Chunk to evaluate
        filter: Filter condition to apply

    Returns:
        True if chunk passes the filter, False otherwise

    Note:
        Returns False if the field doesn't exist in chunk metadata.
        This is by design - missing fields fail filter conditions gracefully.

    Examples:
        >>> chunk = Chunk(
        ...     text="test",
        ...     embedding=[0.1],
        ...     metadata={"price": 50, "category": "tech"},
        ...     document_id=uuid4()
        ... )
        >>> filter = MetadataFilter(field="price", operator=FilterOperator.LESS_THAN, value=100)
        >>> evaluate_metadata_filter(chunk, filter)
        True
    """
    # Get field value from metadata
    value = chunk.metadata.get(filter.field)

    # If field doesn't exist, filter fails
    if value is None:
        return False

    # Evaluate based on operator
    match filter.operator:
        case FilterOperator.EQUALS:
            return value == filter.value

        case FilterOperator.NOT_EQUALS:
            return value != filter.value

        case FilterOperator.GREATER_THAN:
            return _safe_compare(value, filter.value, lambda a, b: a > b)

        case FilterOperator.GREATER_THAN_OR_EQUAL:
            return _safe_compare(value, filter.value, lambda a, b: a >= b)

        case FilterOperator.LESS_THAN:
            return _safe_compare(value, filter.value, lambda a, b: a < b)

        case FilterOperator.LESS_THAN_OR_EQUAL:
            return _safe_compare(value, filter.value, lambda a, b: a <= b)

        case FilterOperator.IN:
            # Value must be in the list
            if not isinstance(filter.value, list):
                return False
            return value in filter.value

        case FilterOperator.NOT_IN:
            # Value must not be in the list
            if not isinstance(filter.value, list):
                return False
            return value not in filter.value

        case FilterOperator.CONTAINS:
            # String contains substring
            if not isinstance(value, str) or not isinstance(filter.value, str):
                return False
            return filter.value in value

        case FilterOperator.NOT_CONTAINS:
            # String does not contain substring
            if not isinstance(value, str) or not isinstance(filter.value, str):
                return False
            return filter.value not in value

        case FilterOperator.STARTS_WITH:
            # String starts with prefix
            if not isinstance(value, str) or not isinstance(filter.value, str):
                return False
            return value.startswith(filter.value)

        case FilterOperator.ENDS_WITH:
            # String ends with suffix
            if not isinstance(value, str) or not isinstance(filter.value, str):
                return False
            return value.endswith(filter.value)

        case _:
            # Unknown operator - fail safe
            return False


def evaluate_filter_group(chunk: Chunk, group: FilterGroup) -> bool:
    """
    Evaluate a filter group (AND/OR logic) against a chunk.

    Supports nested filter groups for complex queries.

    Args:
        chunk: Chunk to evaluate
        group: Filter group with logical operator

    Returns:
        True if chunk passes the filter group, False otherwise

    Examples:
        >>> # AND logic: all filters must pass
        >>> group = FilterGroup(
        ...     operator=LogicalOperator.AND,
        ...     filters=[
        ...         MetadataFilter(field="price", operator=FilterOperator.LESS_THAN, value=100),
        ...         MetadataFilter(field="in_stock", operator=FilterOperator.EQUALS, value=True)
        ...     ]
        ... )
        >>> evaluate_filter_group(chunk, group)
        True  # Only if BOTH conditions are true

        >>> # OR logic: at least one filter must pass
        >>> group = FilterGroup(
        ...     operator=LogicalOperator.OR,
        ...     filters=[...]
        ... )
        >>> evaluate_filter_group(chunk, group)
        True  # If ANY condition is true
    """
    # Empty filter group passes (no constraints)
    if not group.filters:
        return True

    # Evaluate each filter (supports both MetadataFilter and nested FilterGroup)
    results = []
    for f in group.filters:
        if isinstance(f, MetadataFilter):
            results.append(evaluate_metadata_filter(chunk, f))
        elif isinstance(f, FilterGroup):
            # Recursive evaluation for nested groups
            results.append(evaluate_filter_group(chunk, f))
        else:
            # Unknown filter type - fail safe
            results.append(False)

    # Apply logical operator
    match group.operator:
        case LogicalOperator.AND:
            return all(results)
        case LogicalOperator.OR:
            return any(results)
        case _:
            # Unknown operator - fail safe
            return False


def evaluate_search_filters(chunk: Chunk, filters: SearchFilters) -> bool:
    """
    Evaluate complete SearchFilters against a chunk.

    PRIORITY SYSTEM:
    1. If custom_filter is provided, it takes COMPLETE PRECEDENCE over declarative filters
    2. Otherwise, apply declarative filters (metadata, time-based, document IDs)

    Declarative filter types:
    - Metadata filters (complex AND/OR logic)
    - Time-based filters (created_after, created_before)
    - Document ID filters

    All declarative conditions must pass (implicit AND across filter types).

    Args:
        chunk: Chunk to evaluate
        filters: Complete search filter specification

    Returns:
        True if chunk passes all filters, False otherwise

    Examples:
        # Declarative filters
        >>> filters = SearchFilters(
        ...     metadata=FilterGroup(
        ...         operator=LogicalOperator.AND,
        ...         filters=[
        ...             MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="tech")
        ...         ]
        ...     ),
        ...     created_after=datetime(2024, 1, 1),
        ...     document_ids=["doc-uuid-1", "doc-uuid-2"]
        ... )
        >>> evaluate_search_filters(chunk, filters)
        True  # Only if ALL conditions pass
    """
    # NOTE: This code path is for direct evaluator usage (advanced/developer use cases).
    # The SDK applies custom filters client-side via _apply_client_side_filter instead. For direct
    # evaluator usage, custom filters take precedence over declarative filters.
    # Use getattr to safely check for custom_filter (only exists on SearchFiltersWithCallable)
    custom_filter = getattr(filters, "custom_filter", None)
    if custom_filter is not None:
        try:
            return custom_filter(chunk)
        except Exception:
            # Custom filter raised exception - fail gracefully
            # This prevents user bugs from crashing the search
            return False

    # Declarative filters
    # Check time-based filters
    if filters.created_after is not None:
        if chunk.created_at < filters.created_after:
            return False

    if filters.created_before is not None:
        if chunk.created_at > filters.created_before:
            return False

    # Check document ID filter
    if filters.document_ids is not None:
        # Convert UUID to string for comparison
        if str(chunk.document_id) not in filters.document_ids:
            return False

    # Check metadata filters
    if filters.metadata is not None:
        if not evaluate_filter_group(chunk, filters.metadata):
            return False

    # All filters passed!
    return True


def _safe_compare(
    value: Any, compare_value: Any, comparator: Callable[[Any, Any], bool]
) -> bool:
    """
    Safely compare two values with a comparator function.

    Handles type mismatches gracefully by returning False.

    Args:
        value: Value from chunk metadata
        compare_value: Value from filter
        comparator: Comparison function (e.g., lambda a, b: a > b)

    Returns:
        Result of comparison, or False if types are incompatible
    """
    try:
        return comparator(value, compare_value)
    except (TypeError, ValueError):
        # Type mismatch or invalid comparison - fail gracefully
        return False
