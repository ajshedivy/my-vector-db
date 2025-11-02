#!/usr/bin/env python3
"""
Example: Custom Filter Functions (Client-Side)

This script demonstrates client-side filtering using custom Python functions
for complex text-based and business logic filtering.

Prerequisites:
1. API server running on http://localhost:8000
2. Vector DB SDK installed: pip install my-vector-db

What you'll learn:
- Using custom filter functions for text-based filtering
- Implementing complex business logic filters
- Client-side vs server-side filtering trade-offs
"""

from my_vector_db.sdk import VectorDBClient, SearchResult


def main():
    """Demonstrate custom client-side search filters."""

    client = VectorDBClient(base_url="http://localhost:8000")

    print("=" * 70)
    print("Custom Filter Functions Example")
    print("=" * 70)

    # Create library and document
    library = client.create_library(name="articles", index_type="flat")
    document = client.create_document(library_id=library.id, name="tech_articles")
    print(f"\n✓ Created library and document\n")

    # Add articles with varying content
    articles = [
        {
            "text": "Machine learning and neural networks transform AI applications",
            "embedding": [0.9, 0.8, 0.1],
            "metadata": {"category": "ai", "word_count": 8},
        },
        {
            "text": "Deep learning enables advanced pattern recognition",
            "embedding": [0.85, 0.75, 0.15],
            "metadata": {"category": "ai", "word_count": 6},
        },
        {
            "text": "Quantum computers solve optimization problems efficiently",
            "embedding": [0.5, 0.4, 0.6],
            "metadata": {"category": "quantum", "word_count": 6},
        },
        {
            "text": "Neural architecture search automates model design",
            "embedding": [0.88, 0.78, 0.12],
            "metadata": {"category": "ai", "word_count": 6},
        },
    ]

    client.add_chunks(document_id=document.id, chunks=articles)
    print(f"✓ Added {len(articles)} articles\n")

    # Example 1: Simple lambda filter
    print("=" * 70)
    print("Filter 1: Articles containing 'neural'")
    print("=" * 70 + "\n")

    results = client.search(
        library_id=library.id,
        embedding=[0.9, 0.8, 0.1],
        k=10,
        filter_function=lambda result: "neural" in result.text.lower(),
    )

    for result in results.results:
        print(f"- {result.text}")

    # Example 2: Complex filter function
    print("\n" + "=" * 70)
    print("Filter 2: AI articles with keywords 'learning' OR 'network'")
    print("=" * 70 + "\n")

    def ai_learning_filter(result: SearchResult) -> bool:
        """Filter for AI articles mentioning learning or networks."""
        if result.metadata.get("category") != "ai":
            return False

        text_lower = result.text.lower()
        keywords = ["learning", "network"]

        return any(keyword in text_lower for keyword in keywords)

    results = client.search(
        library_id=library.id,
        embedding=[0.9, 0.8, 0.1],
        k=10,
        filter_function=ai_learning_filter,
    )

    for result in results.results:
        print(f"- {result.text}")

    # Example 3: Business logic filter
    print("\n" + "=" * 70)
    print("Filter 3: Short articles (word_count < 7) with high relevance")
    print("=" * 70 + "\n")

    def short_relevant_filter(result: SearchResult) -> bool:
        """Filter for concise, relevant articles."""
        word_count = result.metadata.get("word_count", 0)
        is_short = word_count < 7
        is_relevant = result.score > 0.9  # High similarity score

        return is_short and is_relevant

    results = client.search(
        library_id=library.id,
        embedding=[0.9, 0.8, 0.1],
        k=10,
        filter_function=short_relevant_filter,
    )

    for result in results.results:
        print(f"- {result.text}")
        print(f"  Words: {result.metadata['word_count']}, Score: {result.score:.4f}\n")

    # Cleanup
    client.delete_library(library.id)
    print("✓ Cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
