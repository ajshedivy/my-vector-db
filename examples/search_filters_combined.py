#!/usr/bin/env python3
"""
Example: Combined Declarative + Custom Filters

This script demonstrates combining server-side declarative filters
with client-side custom filter functions for maximum flexibility.

Prerequisites:
1. API server running on http://localhost:8000
2. Vector DB SDK installed: pip install my-vector-db

What you'll learn:
- Combining declarative (server-side) and custom (client-side) filters
- Two-stage filtering workflow
- Performance optimization with combined filters
- When to use each filtering approach
"""

from my_vector_db.sdk import (
    VectorDBClient,
    SearchFilters,
    SearchFiltersWithCallable,
    FilterGroup,
    MetadataFilter,
    FilterOperator,
    LogicalOperator,
)


def main():
    """Demonstrate combined declarative and custom search filters."""

    client = VectorDBClient(base_url="http://localhost:8000")

    print("=" * 70)
    print("Combined Declarative + Custom Filters Example")
    print("=" * 70)

    # Create library and document
    library = client.create_library(name="combined_demo", index_type="flat")
    document = client.create_document(library_id=library.id, name="ml_papers")
    print(f"\n✓ Created library and document\n")

    # Add ML research papers
    papers = [
        {
            "text": "Transformer models revolutionize natural language processing",
            "embedding": [0.9, 0.8, 0.1],
            "metadata": {"category": "nlp", "confidence": 0.95, "year": 2024},
        },
        {
            "text": "Convolutional neural networks excel at image classification",
            "embedding": [0.85, 0.75, 0.15],
            "metadata": {"category": "vision", "confidence": 0.92, "year": 2024},
        },
        {
            "text": "Reinforcement learning agents master complex games",
            "embedding": [0.7, 0.6, 0.4],
            "metadata": {"category": "rl", "confidence": 0.88, "year": 2023},
        },
        {
            "text": "Graph neural networks process structured data efficiently",
            "embedding": [0.75, 0.65, 0.35],
            "metadata": {"category": "graph", "confidence": 0.90, "year": 2024},
        },
    ]

    client.add_chunks(document_id=document.id, chunks=papers)
    print(f"✓ Added {len(papers)} papers\n")

    # Example 1: Server-side filter only (for comparison)
    print("=" * 70)
    print("Baseline: Server-side filter only (confidence > 0.9)")
    print("=" * 70 + "\n")

    declarative_only = SearchFilters(
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
        library_id=library.id, embedding=[0.9, 0.8, 0.1], k=10, filters=declarative_only
    )

    print(f"Found {len(results.results)} results")
    for result in results.results:
        print(f"- {result.text[:55]}...")

    # Example 2: Combined filters (server + client)
    print("\n" + "=" * 70)
    print("Combined: confidence > 0.9 (server) + contains 'neural' (client)")
    print("=" * 70 + "\n")

    combined_filters = SearchFiltersWithCallable(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(
                    field="confidence", operator=FilterOperator.GREATER_THAN, value=0.9
                )
            ],
        ),
        custom_filter=lambda result: "neural" in result.text.lower(),
    )

    results = client.search(
        library_id=library.id,
        embedding=[0.9, 0.8, 0.1],
        k=10,
        combined_filters=combined_filters,
    )

    print(f"Found {len(results.results)} results")
    for result in results.results:
        print(f"- {result.text}")
        print(f"  Confidence: {result.metadata['confidence']}\n")

    # Example 3: Complex combined filtering
    print("=" * 70)
    print("Advanced: (year=2024 AND confidence>0.9) + keyword check")
    print("=" * 70 + "\n")

    advanced_filters = SearchFiltersWithCallable(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(
                    field="year", operator=FilterOperator.EQUALS, value=2024
                ),
                MetadataFilter(
                    field="confidence", operator=FilterOperator.GREATER_THAN, value=0.9
                ),
            ],
        ),
        custom_filter=lambda r: any(
            keyword in r.text.lower()
            for keyword in ["transformer", "neural", "network"]
        ),
    )

    results = client.search(
        library_id=library.id,
        embedding=[0.9, 0.8, 0.1],
        k=10,
        combined_filters=advanced_filters,
    )

    print(f"Found {len(results.results)} results")
    for result in results.results:
        print(f"- {result.text}")

    # Cleanup
    client.delete_library(library.id)
    print("\n✓ Cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
