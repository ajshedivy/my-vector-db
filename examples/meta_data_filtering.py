from my_vector_db import VectorDBClient
from my_vector_db.domain.models import (
    SearchFilters,
    FilterGroup,
    MetadataFilter,
    FilterOperator,
    LogicalOperator,
)

client = VectorDBClient(base_url="http://localhost:8000")

ai_articles = [
    # AI/ML articles (category: ai)
    {
        "text": "Machine learning models use neural networks for pattern recognition",
        "embedding": [0.9, 0.8, 0.1, 0.2, 0.3],
        "metadata": {
            "category": "ai",
            "topic": "machine learning",
            "confidence": 0.95,
            "author": "Alice",
        },
    },
    {
        "text": "Deep learning architectures enable complex AI applications",
        "embedding": [0.85, 0.75, 0.15, 0.25, 0.35],
        "metadata": {
            "category": "ai",
            "topic": "deep learning",
            "confidence": 0.88,
            "author": "Bob",
        },
    },
    {
        "text": "AI is super cool and has many applications in various industries",
        "embedding": [0.85, 0.75, 0.15, 0.25, 0.35],
        "metadata": {
            "category": "ai",
            "topic": "deep learning",
            "confidence": 0.79,
            "author": "Bob",
        },
    },
]

cloud_articles = [
    {
        "text": "Cloud infrastructure provides scalable computing resources",
        "embedding": [0.3, 0.2, 0.9, 0.8, 0.1],
        "metadata": {
            "category": "cloud",
            "topic": "infrastructure",
            "confidence": 0.92,
            "author": "Charlie",
        },
    },
    {
        "text": "Kubernetes orchestrates containerized applications in the cloud",
        "embedding": [0.35, 0.25, 0.85, 0.75, 0.15],
        "metadata": {
            "category": "cloud",
            "topic": "kubernetes",
            "confidence": 0.89,
            "author": "Alice",
        },
    },
]

library = client.create_library(
    name="tech_articles",
    index_type="flat",
    index_config={"metric": "cosine"},
)

document_ai = client.create_document(
    library_id=library.id, name="AI Articles", metadata={"category": "ai", "year": 2024}
)

document_cloud = client.create_document(
    library_id=library.id,
    name="Cloud Articles",
    metadata={"category": "cloud", "year": 2024},
)


def add_chunks_to_document(document, articles):
    for i, article in enumerate(articles):
        article["metadata"]["chunk_num"] = i + 1  # Add chunk number to metadata
        client.add_chunk(
            document_id=document.id,
            text=article["text"],
            embedding=article["embedding"],
            metadata=article["metadata"],
        )


def main():
    add_chunks_to_document(document_ai, ai_articles)
    add_chunks_to_document(document_cloud, cloud_articles)

    print(f"Added {len(ai_articles)} AI articles to document '{document_ai.name}'")
    print(
        f"Added {len(cloud_articles)} Cloud articles to document '{document_cloud.name}'"
    )

    # Query vector similar to AI/ML content
    query_vector = [0.88, 0.78, 0.12, 0.22, 0.32]

    results = client.search(library_id=library.id, embedding=query_vector, k=5)

    print(
        f"\n✓ Initial Search Results: {results.total} matches ({results.query_time_ms:.2f}ms)"
    )
    for i, result in enumerate(results.results, 1):
        print(f"\n  {i}. Score: {result.score:.4f}")
        print(f"     Text: {result.text[:60]}...")
        print(f"     Category: {result.metadata['category']}")
        print(f"     Confidence: {result.metadata['confidence']}")
        print(f"     Author: {result.metadata['author']}")

    # Create declarative filters using SearchFilters
    filters = SearchFilters(
        metadata=FilterGroup(
            operator=LogicalOperator.AND,
            filters=[
                MetadataFilter(
                    field="category", operator=FilterOperator.EQUALS, value="ai"
                ),
                MetadataFilter(
                    field="confidence",
                    operator=FilterOperator.GREATER_THAN,
                    value=0.8,
                ),
            ],
        )
    )

    print("\nFilter Configuration:")
    print("  Type: SearchFilters (declarative only)")
    print("  Rules:")
    print("    - category == 'ai' (server-side)")
    print("    - confidence > 0.8 (server-side)")
    print("  Over-fetch: NO (standard k results)")

    results = client.search(
        library_id=library.id, embedding=query_vector, k=5, filters=filters
    )

    print(f"\n✓ Results: {results.total} matches ({results.query_time_ms:.2f}ms)")
    print("\nMatching Articles:")
    for i, result in enumerate(results.results, 1):
        print(f"\n  {i}. Score: {result.score:.4f}")
        print(f"     Text: {result.text[:60]}...")
        print(f"     Category: {result.metadata['category']}")
        print(f"     Confidence: {result.metadata['confidence']}")
        print("     ✓ Passes server-side filters")

    client.delete_library(library_id=library.id)
    client.close()


if __name__ == "__main__":
    main()
