#!/usr/bin/env python3
"""
Example: Basic Vector Similarity Search

This script demonstrates the fundamentals of k-nearest neighbor (k-NN)
vector search using the Vector Database SDK.

Prerequisites:
1. API server running on http://localhost:8000
2. Vector DB SDK installed: pip install my-vector-db

What you'll learn:
- Creating a library and adding documents with embeddings
- Performing basic similarity search
- Interpreting search results and scores
- Adjusting the k parameter for different result counts
"""

from my_vector_db.sdk import VectorDBClient


def main():
    """Demonstrate basic vector similarity search."""

    # Initialize client
    client = VectorDBClient(base_url="http://localhost:8000")

    print("=" * 70)
    print("Basic Vector Similarity Search Example")
    print("=" * 70)

    # Create library with FLAT index (exact search)
    library = client.create_library(
        name="search_demo", index_type="flat", index_config={"metric": "cosine"}
    )
    print(f"\n✓ Created library: {library.name}")

    # Create document
    document = client.create_document(
        library_id=library.id,
        name="ai_concepts",
        metadata={"topic": "artificial_intelligence"},
    )
    print(f"✓ Created document: {document.name}")

    # Add sample chunks with embeddings
    chunks_data = [
        {
            "text": "Machine learning models learn patterns from data",
            "embedding": [0.9, 0.8, 0.1, 0.2, 0.3],
            "metadata": {"subtopic": "machine_learning"},
        },
        {
            "text": "Deep learning uses neural networks with multiple layers",
            "embedding": [0.85, 0.75, 0.15, 0.25, 0.35],
            "metadata": {"subtopic": "deep_learning"},
        },
        {
            "text": "Natural language processing analyzes human language",
            "embedding": [0.7, 0.6, 0.4, 0.3, 0.2],
            "metadata": {"subtopic": "nlp"},
        },
        {
            "text": "Computer vision enables machines to interpret images",
            "embedding": [0.5, 0.4, 0.6, 0.7, 0.8],
            "metadata": {"subtopic": "computer_vision"},
        },
    ]

    chunks = client.add_chunks(document_id=document.id, chunks=chunks_data)
    print(f"✓ Added {len(chunks)} chunks with embeddings\n")

    # Perform similarity search
    print("=" * 70)
    print("Searching for vectors similar to 'machine learning'")
    print("=" * 70)

    query_embedding = [0.88, 0.78, 0.12, 0.22, 0.32]  # Similar to ML embedding

    # Search with k=3 (top 3 results)
    results = client.search(library_id=library.id, embedding=query_embedding, k=3)

    print(f"\nFound {len(results.results)} results in {results.query_time_ms:.2f}ms\n")

    for i, result in enumerate(results.results, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Text: {result.text}")
        print(f"   Subtopic: {result.metadata.get('subtopic')}\n")

    # Demonstrate adjusting k parameter
    print("=" * 70)
    print("Searching with k=1 (most similar only)")
    print("=" * 70)

    results_k1 = client.search(library_id=library.id, embedding=query_embedding, k=1)

    print(f"\nTop result: {results_k1.results[0].text}")
    print(f"Score: {results_k1.results[0].score:.4f}\n")

    # Cleanup
    client.delete_library(library.id)
    print("✓ Cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
