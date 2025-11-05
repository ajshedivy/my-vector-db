"""
Core domain models for the Vector Database.

These Pydantic models represent the core entities:
- Chunk: A piece of text with its embedding and metadata
- Document: A collection of chunks with metadata
- Library: A collection of documents with an associated index
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FilterOperator(str, Enum):
    """Supported filter operators."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


class MetadataFilter(BaseModel):
    """Single metadata filter condition."""

    field: str = Field(..., description="Metadata field to filter on")
    operator: FilterOperator = Field(..., description="Filter operator")
    value: Union[str, int, float, bool, datetime, List[Any]] = Field(
        ..., description="Value to compare against"
    )

    @field_validator("value")
    @classmethod
    def validate_value_type(cls, v: Any, info) -> Any:
        """Validate value matches operator requirements."""
        operator = info.data.get("operator")

        if operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(v, list):
                raise ValueError(f"{operator} operator requires a list value")
        elif operator in [
            FilterOperator.CONTAINS,
            FilterOperator.NOT_CONTAINS,
            FilterOperator.STARTS_WITH,
            FilterOperator.ENDS_WITH,
        ]:
            if not isinstance(v, str):
                raise ValueError(f"{operator} operator requires a string value")

        return v


class LogicalOperator(str, Enum):
    """Logical operators for combining filters."""

    AND = "and"
    OR = "or"


class FilterGroup(BaseModel):
    """Group of filters combined with logical operators."""

    operator: LogicalOperator = Field(
        default=LogicalOperator.AND, description="Logical operator to combine filters"
    )
    filters: List[Union[MetadataFilter, "FilterGroup"]] = Field(
        default_factory=list, description="List of filters or nested filter groups"
    )

    @field_validator("filters")
    @classmethod
    def validate_filters_not_empty(cls, v: List) -> List:
        """Ensure filters list is not empty."""
        if not v:
            raise ValueError("filters list cannot be empty")
        return v


# Allow recursive type reference
FilterGroup.model_rebuild()


class SearchFilters(BaseModel):
    """
    Declarative filter specification for search queries (API-compatible).

    This base model supports declarative metadata filters and is safe for
    JSON serialization and OpenAPI schema generation.

    For SDK usage with custom Python filter functions, use SearchFiltersWithCallable.

    Examples:
        # Declarative metadata filters
        >>> filters = SearchFilters(
        ...     metadata=FilterGroup(
        ...         operator=LogicalOperator.AND,
        ...         filters=[
        ...             MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="tech")
        ...         ]
        ...     )
        ... )

        # Time-based filters
        >>> filters = SearchFilters(
        ...     created_after=datetime(2024, 1, 1),
        ...     created_before=datetime(2024, 12, 31)
        ... )

        # Document ID filters
        >>> filters = SearchFilters(
        ...     document_ids=["doc-id-1", "doc-id-2"]
        ... )
    """

    metadata: Optional[FilterGroup] = Field(
        default=None, description="Metadata filters to apply"
    )
    created_after: Optional[datetime] = Field(
        default=None, description="Filter chunks created after this date"
    )
    created_before: Optional[datetime] = Field(
        default=None, description="Filter chunks created before this date"
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None, description="Filter by specific document IDs"
    )

    @field_validator("created_after", "created_before")
    @classmethod
    def validate_dates(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Validate date ordering if both dates are provided."""
        if v and info.data.get("created_after") and info.data.get("created_before"):
            if info.data["created_after"] >= info.data["created_before"]:
                raise ValueError("created_after must be before created_before")
        return v


class SearchFiltersWithCallable(SearchFilters):
    """
    Extended filter specification with custom Python function support (SDK only).

    This model extends SearchFilters to add support for custom filter functions.
    The custom_filter field is excluded from serialization to maintain API compatibility.

    Note:
    - custom_filter takes precedence over all declarative filters when provided
    - Filter function receives SearchResult objects (from SDK) with fields:
      chunk_id, document_id, text, score, metadata

    Examples:
        # Custom filter function on search results (SDK/Python only)
        >>> filters = SearchFiltersWithCallable(
        ...     custom_filter=lambda result: result.metadata.get("rating", 0) > 50
        ... )

        # Filter using similarity score (only available in search results)
        >>> filters = SearchFiltersWithCallable(
        ...     custom_filter=lambda result: result.score > 0.85
        ... )

        # Combined
        >>> filters = SearchFiltersWithCallable(
        ...     metadata=FilterGroup(...),
        ...     custom_filter=lambda result: calculate_quality(result) > threshold
        ... )

        # Complex custom function
        >>> def quality_filter(result) -> bool:
        ...     # result has: chunk_id, document_id, text, score, metadata
        ...     rating_score = result.metadata.get("rating", 0) * 10
        ...     rating_score += result.metadata.get("views", 0) / 100
        ...     return rating_score >= 70 and result.score > 0.5
        >>> filters = SearchFiltersWithCallable(custom_filter=quality_filter)
    """

    # Type hint uses Any to avoid circular dependency with SDK layer
    # Actual objects passed are SearchResult instances from sdk.models
    custom_filter: Optional[Callable[[Any], bool]] = Field(
        default=None,
        exclude=True,  # Don't serialize - not API-safe
        description="Custom filter function (SDK/Python only, receives SearchResult object returns bool)",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IndexType(str, Enum):
    """
    Supported vector index types.

    Each index type offers different tradeoffs between accuracy, speed, and memory:
    - FLAT: Exact search, O(n) time, baseline for comparison
    - HNSW: Approximate search, graph-based, excellent for high-dimensional data
    """

    FLAT = "flat"
    HNSW = "hnsw"


class Chunk(BaseModel):
    """
    A chunk represents a piece of text with its vector embedding.

    Attributes:
        id: Unique identifier for the chunk
        text: The actual text content
        embedding: Vector representation of the text (list of floats)
        metadata: Additional metadata (e.g., source, page_number, etc.)
        document_id: ID of the parent document
        created_at: Timestamp when chunk was created
    """

    id: UUID = Field(default_factory=uuid4)
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: UUID
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class Document(BaseModel):
    """
    A document is a collection of chunks.

    Attributes:
        id: Unique identifier for the document
        name: Human-readable name for the document
        chunk_ids: List of chunk IDs belonging to this document
        metadata: Additional metadata (e.g., author, category, etc.)
        library_id: ID of the parent library
        created_at: Timestamp when document was created
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    chunk_ids: List[UUID] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    library_id: UUID
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class Library(BaseModel):
    """
    A library is a collection of documents with a vector index.

    Attributes:
        id: Unique identifier for the library
        name: Human-readable name for the library
        document_ids: List of document IDs in this library
        metadata: Additional metadata
        index_type: Type of vector index to use
        index_config: Configuration parameters for the index
        created_at: Timestamp when library was created
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    document_ids: List[UUID] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    index_type: IndexType = IndexType.FLAT
    index_config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class BuildIndexResult(BaseModel):
    """
    Result of a build index operation.

    Attributes:
        library_id: UUID of the library
        total_vectors: Number of vectors indexed
        dimension: Vector dimension
        index_type: Index type (flat, hnsw)
        index_config: Configuration parameters for the index
    """

    library_id: UUID = Field(..., description="Library ID")
    total_vectors: int = Field(..., description="Number of vectors indexed")
    dimension: int = Field(..., description="Vector dimension")
    index_type: IndexType = Field(..., description="Index type (flat, hnsw)")
    index_config: Dict[str, Any] = Field(
        default_factory=dict, description="Index configuration parameters"
    )

    model_config = ConfigDict(frozen=True)
