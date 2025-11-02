#!/usr/bin/env python3
"""
Example: Batch Operations for Efficient Data Loading

This script demonstrates efficient batch operations for creating
multiple documents and chunks in single API calls.

Prerequisites:
1. API server running on http://localhost:8000
2. Vector DB SDK installed: pip install my-vector-db

What you'll learn:
- Batch adding chunks (vs individual operations)
- Performance comparison
- Using Chunk objects vs dictionaries
- Atomic batch operations
"""

import time
from my_vector_db.sdk import VectorDBClient
from my_vector_db.domain.models import Chunk


def main():
    """Demonstrate efficient batch operations."""

    client = VectorDBClient(base_url="http://localhost:8000")

    print("=" * 70)
    print("Batch Operations Example")
    print("=" * 70)

    # Create library and document
    library = client.create_library(name="batch_demo", index_type="flat")
    document = client.create_document(library_id=library.id, name="batch_doc")
    print(f"\n✓ Created library and document\n")

    # Example 1: Batch add with dictionaries (most common)
    print("=" * 70)
    print("Method 1: Batch add chunks using dictionaries")
    print("=" * 70 + "\n")

    chunk_dicts = [
        {
            "text": f"Sample text chunk number {i}",
            "embedding": [0.1 * i, 0.2 * i, 0.3 * i],
            "metadata": {"index": i, "source": "batch"},
        }
        for i in range(1, 6)
    ]

    start = time.time()
    chunks = client.add_chunks(document_id=document.id, chunks=chunk_dicts)
    duration = time.time() - start

    print(f"✓ Added {len(chunks)} chunks in {duration * 1000:.2f}ms")
    print(f"  Average: {(duration * 1000) / len(chunks):.2f}ms per chunk\n")

    # Example 2: Batch add with Chunk objects
    print("=" * 70)
    print("Method 2: Batch add chunks using Chunk objects")
    print("=" * 70 + "\n")

    chunk_objects = [
        Chunk(
            document_id=document.id,
            text=f"AI concept: {topic}",
            embedding=[0.9, 0.8, 0.1 * i],
            metadata={"topic": topic},
        )
        for i, topic in enumerate(["ML", "DL", "NLP", "CV"], 1)
    ]

    start = time.time()
    chunks = client.add_chunks(chunks=chunk_objects)
    duration = time.time() - start

    print(f"✓ Added {len(chunks)} chunks in {duration * 1000:.2f}ms\n")

    # Example 3: Performance comparison - batch vs individual
    print("=" * 70)
    print("Performance Comparison: Batch vs Individual Operations")
    print("=" * 70 + "\n")

    doc_individual = client.create_document(library_id=library.id, name="individual")
    doc_batch = client.create_document(library_id=library.id, name="batch")

    test_data = [
        {"text": f"Test {i}", "embedding": [0.1 * i, 0.2 * i, 0.3 * i]}
        for i in range(10)
    ]

    # Individual operations
    start = time.time()
    for data in test_data:
        client.add_chunk(
            document_id=doc_individual.id,
            text=data["text"],
            embedding=data["embedding"],
        )
    individual_time = time.time() - start

    # Batch operation
    start = time.time()
    client.add_chunks(document_id=doc_batch.id, chunks=test_data)
    batch_time = time.time() - start

    print(f"Individual operations (10 chunks): {individual_time * 1000:.2f}ms")
    print(f"Batch operation (10 chunks):       {batch_time * 1000:.2f}ms")
    print(f"Speedup: {individual_time / batch_time:.1f}x faster\n")

    # Example 4: Large batch operation
    print("=" * 70)
    print("Large Batch: 100 chunks in single operation")
    print("=" * 70 + "\n")

    doc_large = client.create_document(library_id=library.id, name="large_batch")

    large_batch = [
        {
            "text": f"Document {i}: Lorem ipsum dolor sit amet",
            "embedding": [0.01 * i, 0.02 * i, 0.03 * i],
            "metadata": {"batch_id": "large", "index": i},
        }
        for i in range(100)
    ]

    start = time.time()
    chunks = client.add_chunks(document_id=doc_large.id, chunks=large_batch)
    duration = time.time() - start

    print(f"✓ Added {len(chunks)} chunks in {duration * 1000:.2f}ms")
    print(f"  Throughput: {len(chunks) / duration:.0f} chunks/second\n")

    # Cleanup
    client.delete_library(library.id)
    print("✓ Cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
