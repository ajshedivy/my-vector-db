"""
Filter Evaluator Tests

Comprehensive tests for metadata filtering logic.
Run with: pytest tests/test_filter_evaluator.py -v
"""

from datetime import datetime
from uuid import uuid4

import pytest

from my_vector_db.domain.models import (
    Chunk,
    FilterGroup,
    FilterOperator,
    LogicalOperator,
    MetadataFilter,
    SearchFilters,
    SearchFiltersWithCallable,
)
from my_vector_db.filters.evaluator import (
    evaluate_filter_group,
    evaluate_metadata_filter,
    evaluate_search_filters,
)

from my_vector_db.sdk.models import SearchResult


@pytest.fixture
def sample_chunk() -> Chunk:
    """Create a sample chunk for testing."""
    return Chunk(
        id=uuid4(),
        text="Sample chunk text",
        embedding=[0.1, 0.2, 0.3],
        metadata={
            "category": "technology",
            "price": 99.99,
            "in_stock": True,
            "views": 1500,
            "tags": "python machine-learning AI",
            "rating": 4.5,
        },
        document_id=uuid4(),
        created_at=datetime(2024, 6, 15, 12, 0, 0),
    )


class TestMetadataFilterOperators:
    """Tests for each filter operator."""

    def test_equals_operator_matches(self, sample_chunk: Chunk) -> None:
        """Test EQUALS operator with matching value."""
        filter = MetadataFilter(
            field="category", operator=FilterOperator.EQUALS, value="technology"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

    def test_equals_operator_no_match(self, sample_chunk: Chunk) -> None:
        """Test EQUALS operator with non-matching value."""
        filter = MetadataFilter(
            field="category", operator=FilterOperator.EQUALS, value="sports"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_not_equals_operator(self, sample_chunk: Chunk) -> None:
        """Test NOT_EQUALS operator."""
        filter = MetadataFilter(
            field="category", operator=FilterOperator.NOT_EQUALS, value="sports"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="category", operator=FilterOperator.NOT_EQUALS, value="technology"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_greater_than_operator(self, sample_chunk: Chunk) -> None:
        """Test GREATER_THAN operator."""
        filter = MetadataFilter(
            field="price", operator=FilterOperator.GREATER_THAN, value=50.0
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="price", operator=FilterOperator.GREATER_THAN, value=100.0
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_greater_than_or_equal_operator(self, sample_chunk: Chunk) -> None:
        """Test GREATER_THAN_OR_EQUAL operator."""
        filter = MetadataFilter(
            field="price", operator=FilterOperator.GREATER_THAN_OR_EQUAL, value=99.99
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="price", operator=FilterOperator.GREATER_THAN_OR_EQUAL, value=100.0
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_less_than_operator(self, sample_chunk: Chunk) -> None:
        """Test LESS_THAN operator."""
        filter = MetadataFilter(
            field="price", operator=FilterOperator.LESS_THAN, value=100.0
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="price", operator=FilterOperator.LESS_THAN, value=50.0
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_less_than_or_equal_operator(self, sample_chunk: Chunk) -> None:
        """Test LESS_THAN_OR_EQUAL operator."""
        filter = MetadataFilter(
            field="price", operator=FilterOperator.LESS_THAN_OR_EQUAL, value=99.99
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="price", operator=FilterOperator.LESS_THAN_OR_EQUAL, value=50.0
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_in_operator(self, sample_chunk: Chunk) -> None:
        """Test IN operator."""
        filter = MetadataFilter(
            field="category",
            operator=FilterOperator.IN,
            value=["technology", "sports", "news"],
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="category", operator=FilterOperator.IN, value=["sports", "news"]
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_not_in_operator(self, sample_chunk: Chunk) -> None:
        """Test NOT_IN operator."""
        filter = MetadataFilter(
            field="category", operator=FilterOperator.NOT_IN, value=["sports", "news"]
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="category",
            operator=FilterOperator.NOT_IN,
            value=["technology", "sports"],
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_contains_operator(self, sample_chunk: Chunk) -> None:
        """Test CONTAINS operator."""
        filter = MetadataFilter(
            field="tags", operator=FilterOperator.CONTAINS, value="python"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="tags", operator=FilterOperator.CONTAINS, value="javascript"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_not_contains_operator(self, sample_chunk: Chunk) -> None:
        """Test NOT_CONTAINS operator."""
        filter = MetadataFilter(
            field="tags", operator=FilterOperator.NOT_CONTAINS, value="javascript"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="tags", operator=FilterOperator.NOT_CONTAINS, value="python"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_starts_with_operator(self, sample_chunk: Chunk) -> None:
        """Test STARTS_WITH operator."""
        filter = MetadataFilter(
            field="tags", operator=FilterOperator.STARTS_WITH, value="python"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="tags", operator=FilterOperator.STARTS_WITH, value="AI"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_ends_with_operator(self, sample_chunk: Chunk) -> None:
        """Test ENDS_WITH operator."""
        filter = MetadataFilter(
            field="tags", operator=FilterOperator.ENDS_WITH, value="AI"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="tags", operator=FilterOperator.ENDS_WITH, value="python"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False


class TestFilterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_field_returns_false(self, sample_chunk: Chunk) -> None:
        """Test that filtering on missing field returns False."""
        filter = MetadataFilter(
            field="nonexistent_field", operator=FilterOperator.EQUALS, value="anything"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_comparison_with_incompatible_types(self, sample_chunk: Chunk) -> None:
        """Test comparison operators with incompatible types."""
        # Try to compare string with number
        filter = MetadataFilter(
            field="category", operator=FilterOperator.GREATER_THAN, value=100
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_string_operators_with_non_strings(self, sample_chunk: Chunk) -> None:
        """Test string operators with non-string values."""
        filter = MetadataFilter(
            field="price", operator=FilterOperator.CONTAINS, value="99"
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_in_operator_with_non_list(self, sample_chunk: Chunk) -> None:
        """Test IN operator validates that value must be a list."""
        # Pydantic should raise ValidationError for non-list value with IN operator
        with pytest.raises(ValueError, match="requires a list value"):
            MetadataFilter(
                field="category", operator=FilterOperator.IN, value="technology"
            )

    def test_boolean_comparison(self, sample_chunk: Chunk) -> None:
        """Test filtering on boolean values."""
        filter = MetadataFilter(
            field="in_stock", operator=FilterOperator.EQUALS, value=True
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        filter = MetadataFilter(
            field="in_stock", operator=FilterOperator.EQUALS, value=False
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is False

    def test_numeric_types(self, sample_chunk: Chunk) -> None:
        """Test filtering with different numeric types."""
        # Integer comparison
        filter = MetadataFilter(
            field="views", operator=FilterOperator.GREATER_THAN, value=1000
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True

        # Float comparison
        filter = MetadataFilter(
            field="rating", operator=FilterOperator.GREATER_THAN_OR_EQUAL, value=4.5
        )
        assert evaluate_metadata_filter(sample_chunk, filter) is True


class TestFilterGroups:
    """Tests for filter groups with AND/OR logic."""

    def test_and_group_all_pass(self, sample_chunk: Chunk) -> None:
        """Test AND group where all filters pass."""
        group = FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(
                    field="category", operator=FilterOperator.EQUALS, value="technology"
                ),
                MetadataFilter(
                    field="price", operator=FilterOperator.LESS_THAN, value=100.0
                ),
                MetadataFilter(
                    field="in_stock", operator=FilterOperator.EQUALS, value=True
                ),
            ],
        )
        assert evaluate_filter_group(sample_chunk, group) is True

    def test_and_group_one_fails(self, sample_chunk: Chunk) -> None:
        """Test AND group where one filter fails."""
        group = FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(
                    field="category", operator=FilterOperator.EQUALS, value="technology"
                ),
                MetadataFilter(
                    field="price", operator=FilterOperator.GREATER_THAN, value=100.0
                ),  # This fails
            ],
        )
        assert evaluate_filter_group(sample_chunk, group) is False

    def test_or_group_all_pass(self, sample_chunk: Chunk) -> None:
        """Test OR group where all filters pass."""
        group = FilterGroup(
            operator=LogicalOperator.OR,
            filters=[
                MetadataFilter(
                    field="category", operator=FilterOperator.EQUALS, value="technology"
                ),
                MetadataFilter(
                    field="in_stock", operator=FilterOperator.EQUALS, value=True
                ),
            ],
        )
        assert evaluate_filter_group(sample_chunk, group) is True

    def test_or_group_one_passes(self, sample_chunk: Chunk) -> None:
        """Test OR group where one filter passes."""
        group = FilterGroup(
            operator=LogicalOperator.OR,
            filters=[
                MetadataFilter(
                    field="category", operator=FilterOperator.EQUALS, value="sports"
                ),  # Fails
                MetadataFilter(
                    field="in_stock", operator=FilterOperator.EQUALS, value=True
                ),  # Passes
            ],
        )
        assert evaluate_filter_group(sample_chunk, group) is True

    def test_or_group_all_fail(self, sample_chunk: Chunk) -> None:
        """Test OR group where all filters fail."""
        group = FilterGroup(
            operator=LogicalOperator.OR,
            filters=[
                MetadataFilter(
                    field="category", operator=FilterOperator.EQUALS, value="sports"
                ),
                MetadataFilter(
                    field="price", operator=FilterOperator.GREATER_THAN, value=200.0
                ),
            ],
        )
        assert evaluate_filter_group(sample_chunk, group) is False

    def test_nested_filter_groups(self, sample_chunk: Chunk) -> None:
        """Test nested filter groups (groups within groups)."""
        # (category == "technology" AND price < 100) OR (in_stock == True AND views > 1000)
        group = FilterGroup(
            operator=LogicalOperator.OR,
            filters=[
                FilterGroup(
                    operator=LogicalOperator.AND,
                    filters=[
                        MetadataFilter(
                            field="category",
                            operator=FilterOperator.EQUALS,
                            value="technology",
                        ),
                        MetadataFilter(
                            field="price",
                            operator=FilterOperator.LESS_THAN,
                            value=100.0,
                        ),
                    ],
                ),
                FilterGroup(
                    operator=LogicalOperator.AND,
                    filters=[
                        MetadataFilter(
                            field="in_stock", operator=FilterOperator.EQUALS, value=True
                        ),
                        MetadataFilter(
                            field="views",
                            operator=FilterOperator.GREATER_THAN,
                            value=1000,
                        ),
                    ],
                ),
            ],
        )
        # Both nested groups should pass, so overall OR should pass
        assert evaluate_filter_group(sample_chunk, group) is True

    def test_empty_filter_group(self) -> None:
        """Test that empty filter groups are not allowed by Pydantic validation."""
        # Pydantic should raise ValidationError for empty filters list
        with pytest.raises(ValueError, match="filters list cannot be empty"):
            FilterGroup(operator=LogicalOperator.AND, filters=[])


class TestSearchFilters:
    """Tests for complete SearchFilters evaluation."""

    def test_metadata_filters_only(self, sample_chunk: Chunk) -> None:
        """Test search filters with only metadata filters."""
        filters = SearchFilters(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    MetadataFilter(
                        field="category",
                        operator=FilterOperator.EQUALS,
                        value="technology",
                    ),
                    MetadataFilter(
                        field="price", operator=FilterOperator.LESS_THAN, value=100.0
                    ),
                ],
            )
        )
        assert evaluate_search_filters(sample_chunk, filters) is True

    def test_created_after_filter(self, sample_chunk: Chunk) -> None:
        """Test created_after time filter."""
        # Chunk created at 2024-06-15
        filters = SearchFilters(created_after=datetime(2024, 1, 1))
        assert evaluate_search_filters(sample_chunk, filters) is True

        filters = SearchFilters(created_after=datetime(2024, 12, 31))
        assert evaluate_search_filters(sample_chunk, filters) is False

    def test_created_before_filter(self, sample_chunk: Chunk) -> None:
        """Test created_before time filter."""
        # Chunk created at 2024-06-15
        filters = SearchFilters(created_before=datetime(2024, 12, 31))
        assert evaluate_search_filters(sample_chunk, filters) is True

        filters = SearchFilters(created_before=datetime(2024, 1, 1))
        assert evaluate_search_filters(sample_chunk, filters) is False

    def test_document_ids_filter(self, sample_chunk: Chunk) -> None:
        """Test document_ids filter."""
        filters = SearchFilters(document_ids=[str(sample_chunk.document_id)])
        assert evaluate_search_filters(sample_chunk, filters) is True

        filters = SearchFilters(document_ids=[str(uuid4()), str(uuid4())])
        assert evaluate_search_filters(sample_chunk, filters) is False

    def test_combined_filters(self, sample_chunk: Chunk) -> None:
        """Test combination of all filter types."""
        filters = SearchFilters(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    MetadataFilter(
                        field="category",
                        operator=FilterOperator.EQUALS,
                        value="technology",
                    )
                ],
            ),
            created_after=datetime(2024, 1, 1),
            created_before=datetime(2024, 12, 31),
            document_ids=[str(sample_chunk.document_id)],
        )
        # All conditions should pass
        assert evaluate_search_filters(sample_chunk, filters) is True

    def test_combined_filters_one_fails(self, sample_chunk: Chunk) -> None:
        """Test combination where one filter fails (should return False)."""
        filters = SearchFilters(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    MetadataFilter(
                        field="category",
                        operator=FilterOperator.EQUALS,
                        value="technology",
                    )
                ],
            ),
            created_after=datetime(2024, 1, 1),
            document_ids=[str(uuid4())],  # Wrong document ID - fails
        )
        assert evaluate_search_filters(sample_chunk, filters) is False

    def test_no_filters(self, sample_chunk: Chunk) -> None:
        """Test empty SearchFilters passes everything."""
        filters = SearchFilters()
        assert evaluate_search_filters(sample_chunk, filters) is True


class TestCustomFilters:
    """Tests for custom filter functions."""

    def test_custom_filter_lambda(self, sample_chunk: Chunk) -> None:
        """Test custom filter with lambda function."""
        # Simple lambda filter
        filters = SearchFiltersWithCallable(
            custom_filter=lambda chunk: chunk.metadata.get("price", 0) < 100
        )
        assert evaluate_search_filters(sample_chunk, filters) is True

        # Lambda that fails
        filters = SearchFiltersWithCallable(
            custom_filter=lambda chunk: chunk.metadata.get("price", 0) > 200
        )
        assert evaluate_search_filters(sample_chunk, filters) is False

    def test_custom_filter_complex_function(self, sample_chunk: Chunk) -> None:
        """Test custom filter with complex function."""

        def quality_filter(result: SearchResult) -> bool:
            """Calculate quality score and filter."""
            metadata = result.metadata
            score = 0

            # Add points for various criteria
            score += metadata.get("rating", 0) * 10
            score += metadata.get("views", 0) / 100
            score += 5 if metadata.get("in_stock") else 0

            return score >= 50

        filters = SearchFiltersWithCallable(custom_filter=quality_filter)
        # sample_chunk: rating=4.5, views=1500, in_stock=True
        # score = 4.5*10 + 1500/100 + 5 = 45 + 15 + 5 = 65
        assert evaluate_search_filters(sample_chunk, filters) is True

    def test_custom_filter_exception_handling(self, sample_chunk: Chunk) -> None:
        """Test that exceptions in custom filter are handled gracefully."""

        def buggy_filter(result: SearchResult) -> bool:
            """Filter that raises exception."""
            # Intentionally access non-existent key without get()
            return result.metadata["nonexistent_key"] > 10

        filters = SearchFiltersWithCallable(custom_filter=buggy_filter)
        # Should return False instead of crashing
        assert evaluate_search_filters(sample_chunk, filters) is False

    def test_custom_filter_takes_precedence_over_metadata(
        self, sample_chunk: Chunk
    ) -> None:
        """Test that custom_filter takes precedence over declarative metadata filters."""
        # Create filters with BOTH custom and metadata
        filters = SearchFiltersWithCallable(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    # This would fail if evaluated
                    MetadataFilter(
                        field="category",
                        operator=FilterOperator.EQUALS,
                        value="WRONG_VALUE",
                    )
                ],
            ),
            # Custom filter returns True
            custom_filter=lambda result: True,
        )

        # Custom filter should win, metadata filter should be ignored
        assert evaluate_search_filters(sample_chunk, filters) is True

    def test_custom_filter_takes_precedence_over_time_filters(
        self, sample_chunk: Chunk
    ) -> None:
        """Test that custom_filter takes precedence over time filters."""
        # Time filter that would fail
        filters = SearchFiltersWithCallable(
            created_after=datetime(2025, 1, 1),  # Future date - would fail
            custom_filter=lambda chunk: True,  # Always passes
        )

        # Custom filter should win
        assert evaluate_search_filters(sample_chunk, filters) is True

    def test_custom_filter_takes_precedence_over_document_ids(
        self, sample_chunk: Chunk
    ) -> None:
        """Test that custom_filter takes precedence over document_ids filter."""
        # Document ID filter that would fail
        filters = SearchFiltersWithCallable(
            document_ids=["wrong-id-1", "wrong-id-2"],  # Would fail
            custom_filter=lambda chunk: True,  # Always passes
        )

        # Custom filter should win
        assert evaluate_search_filters(sample_chunk, filters) is True

    def test_custom_filter_access_to_full_chunk(self, sample_chunk: Chunk) -> None:
        """Test that custom filter has full access to chunk object."""

        def comprehensive_filter(result: SearchResult) -> bool:
            """Filter using all chunk properties."""
            # Can access text
            has_sample = "Sample" in result.text

            # Can access metadata
            correct_category = result.metadata.get("category") == "technology"

            # Can access embedding
            has_embedding = len(result.embedding) > 0

            # Can access created_at
            created_in_2024 = result.created_at.year == 2024

            return has_sample and correct_category and has_embedding and created_in_2024

        filters = SearchFiltersWithCallable(custom_filter=comprehensive_filter)
        assert evaluate_search_filters(sample_chunk, filters) is True

    def test_custom_filter_with_closure(self, sample_chunk: Chunk) -> None:
        """Test custom filter with closure capturing external variables."""
        min_price = 50
        max_price = 150

        def price_range_filter(result: SearchResult) -> bool:
            """Filter using closure variables."""
            price = result.metadata.get("price", 0)
            return min_price <= price <= max_price

        filters = SearchFiltersWithCallable(custom_filter=price_range_filter)
        # sample_chunk has price=99.99
        assert evaluate_search_filters(sample_chunk, filters) is True

        # Update closure variables
        min_price = 100
        filters = SearchFiltersWithCallable(
            custom_filter=lambda c: min_price <= c.metadata.get("price", 0) <= max_price
        )
        assert evaluate_search_filters(sample_chunk, filters) is False


class TestRealWorldScenarios:
    """Tests simulating real-world filtering scenarios."""

    def test_ecommerce_product_filter(self) -> None:
        """Test e-commerce product filtering scenario."""
        product = Chunk(
            id=uuid4(),
            text="iPhone 15 Pro",
            embedding=[0.1] * 10,
            metadata={
                "category": "electronics",
                "price": 999.99,
                "brand": "Apple",
                "in_stock": True,
                "rating": 4.7,
            },
            document_id=uuid4(),
            created_at=datetime.now(),
        )

        # Find products: (price < 1000 OR on_sale) AND in_stock AND brand in [Apple, Samsung]
        filters = SearchFilters(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    FilterGroup(
                        operator=LogicalOperator.OR,
                        filters=[
                            MetadataFilter(
                                field="price",
                                operator=FilterOperator.LESS_THAN,
                                value=1000.0,
                            ),
                        ],
                    ),
                    MetadataFilter(
                        field="in_stock", operator=FilterOperator.EQUALS, value=True
                    ),
                    MetadataFilter(
                        field="brand",
                        operator=FilterOperator.IN,
                        value=["Apple", "Samsung"],
                    ),
                ],
            )
        )

        assert evaluate_search_filters(product, filters) is True

    def test_document_search_filter(self) -> None:
        """Test document search with metadata filtering."""
        doc_chunk = Chunk(
            id=uuid4(),
            text="Python tutorial for beginners",
            embedding=[0.1] * 10,
            metadata={
                "language": "en",
                "department": "engineering",
                "doc_type": "tutorial",
                "difficulty": "beginner",
            },
            document_id=uuid4(),
            created_at=datetime(2024, 3, 15),
        )

        # Find: English docs from engineering, created in Q1 2024
        filters = SearchFilters(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    MetadataFilter(
                        field="language", operator=FilterOperator.EQUALS, value="en"
                    ),
                    MetadataFilter(
                        field="department",
                        operator=FilterOperator.EQUALS,
                        value="engineering",
                    ),
                ],
            ),
            created_after=datetime(2024, 1, 1),
            created_before=datetime(2024, 4, 1),
        )

        assert evaluate_search_filters(doc_chunk, filters) is True
