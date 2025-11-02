#!/usr/bin/env python3
"""
Example: Declarative Search Filters (Server-Side)

This script demonstrates server-side declarative filtering using metadata,
time-based filters, and complex logical operators.

Prerequisites:
1. API server running on http://localhost:8000
2. Vector DB SDK installed: pip install my-vector-db

What you'll learn:
- Metadata filtering with various operators (eq, gt, in, contains)
- Combining filters with AND/OR logic
- Time-based filtering
- Document ID filtering
"""

from my_vector_db.sdk import (
    VectorDBClient,
    SearchFilters,
    FilterGroup,
    MetadataFilter,
    FilterOperator,
    LogicalOperator,
)


def main():
    """Demonstrate declarative server-side search filters."""

    client = VectorDBClient(base_url="http://localhost:8000")

    print("=" * 70)
    print("Declarative Search Filters Example")
    print("=" * 70)

    # Create library and document
    library = client.create_library(name="research_papers", index_type="flat")
    document = client.create_document(library_id=library.id, name="papers_2024")
    print(f"\n✓ Created library and document\n")

    # Add research papers with metadata
    papers = [
        {
            "text": "Neural networks achieve state-of-the-art results",
            "embedding": [0.9, 0.8, 0.1],
            "metadata": {"category": "ai", "confidence": 0.95, "author": "Alice"},
        },
        {
            "text": "Deep learning transforms computer vision applications",
            "embedding": [0.85, 0.75, 0.15],
            "metadata": {"category": "ai", "confidence": 0.88, "author": "Bob"},
        },
        {
            "text": "Quantum computing solves complex optimization problems",
            "embedding": [0.5, 0.4, 0.6],
            "metadata": {"category": "quantum", "confidence": 0.92, "author": "Alice"},
        },
        {
            "text": "Blockchain enables decentralized applications",
            "embedding": [0.3, 0.2, 0.8],
            "metadata": {
                "category": "blockchain",
                "confidence": 0.75,
                "author": "Charlie",
            },
        },
    ]

    client.add_chunks(document_id=document.id, chunks=papers)
    print(f"✓ Added {len(papers)} papers\n")

    # Example 1: Simple metadata filter
    print("=" * 70)
    print("Filter 1: Papers in 'ai' category")
    print("=" * 70 + "\n")

    filters = SearchFilters(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(
                    field="category", operator=FilterOperator.EQUALS, value="ai"
                )
            ],
        )
    )

    results = client.search(
        library_id=library.id, embedding=[0.9, 0.8, 0.1], k=10, filters=filters
    )
    for result in results.results:
        print(f"- {result.text[:50]}... (category: {result.metadata['category']})")

    # Example 2: Numeric comparison filter
    print("\n" + "=" * 70)
    print("Filter 2: Papers with confidence > 0.9")
    print("=" * 70 + "\n")

    filters = SearchFilters(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(
                    field="confidence", operator=FilterOperator.GREATER_THAN, value=0.9
                )
            ],
        )
    )

    results = client.search(
        library_id=library.id, embedding=[0.9, 0.8, 0.1], k=10, filters=filters
    )
    for result in results.results:
        print(f"- {result.text[:50]}... (confidence: {result.metadata['confidence']})")

    # Example 3: Complex AND/OR logic
    print("\n" + "=" * 70)
    print("Filter 3: (category='ai' OR category='quantum') AND confidence > 0.85")
    print("=" * 70 + "\n")

    filters = SearchFilters(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                FilterGroup(
                    operator=LogicalOperator.OR,
                    filters=[
                        MetadataFilter(
                            field="category", operator=FilterOperator.EQUALS, value="ai"
                        ),
                        MetadataFilter(
                            field="category",
                            operator=FilterOperator.EQUALS,
                            value="quantum",
                        ),
                    ],
                ),
                MetadataFilter(
                    field="confidence", operator=FilterOperator.GREATER_THAN, value=0.85
                ),
            ],
        )
    )

    results = client.search(
        library_id=library.id, embedding=[0.9, 0.8, 0.1], k=10, filters=filters
    )
    for result in results.results:
        print(f"- {result.text[:50]}...")
        print(
            f"  Category: {result.metadata['category']}, Confidence: {result.metadata['confidence']}\n"
        )

    # Cleanup
    client.delete_library(library.id)
    print("✓ Cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
