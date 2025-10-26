"""
Example demonstrating batch operations with the Vector Database SDK.
"""

from my_vector_db.sdk import VectorDBClient, ServerConnectionError
from my_vector_db import Chunk


def main():
    """Batch operations example."""
    # Create client
    client = VectorDBClient(base_url="http://localhost:8000")

    try:
        # Create a library
        library = client.create_library(
            name="Batch Example Library",
            index_type="flat",
            index_config={"metric": "euclidean"},
        )
        print(f"Created library: {library.id}")

        # Create a document
        document = client.create_document(
            library_id=library.id,
            name="Batch Document",
        )
        print(f"Created document: {document.id}")

        # Batch add chunks - Method 1: Using Chunk objects
        chunks = [
            Chunk(
                document_id=document.id,
                text="The quick brown fox jumps over the lazy dog",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"source": "example", "position": 1},
            ),
            Chunk(
                document_id=document.id,
                text="The slow brown cat jumps over the lazy dog",
                embedding=[0.1, 0.7, 0.3, 0.4, 0.2],
                metadata={"source": "example", "position": 2},
            ),
            Chunk(
                document_id=document.id,
                text="A fast red bird flies through the clear blue sky",
                embedding=[0.9, 0.1, 0.5, 0.3, 0.7],
                metadata={"source": "example", "position": 3},
            ),
        ]

        created_chunks = client.add_chunks(chunks=chunks)

        result = client.build_index(library_id=library.id)
        print(f"Index built with {result.total_vectors} vectors")
        print(f"Dimension: {result.dimension}")
        print(f"Index type: {result.index_type}")
        print(f"Config: {result.index_config}")

        print(f"\nBatch added {len(created_chunks)} chunks (object style)")
        for chunk in created_chunks:
            print(f"  - {chunk.text[:50]}... (ID: {chunk.id})")

        # Batch add chunks - Method 2: Using dicts
        chunk_dicts = [
            {
                "text": "Python is a high-level programming language",
                "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
                "metadata": {"topic": "programming"},
            },
            {
                "text": "FastAPI is a modern web framework for Python",
                "embedding": [0.3, 0.4, 0.5, 0.6, 0.7],
                "metadata": {"topic": "web"},
            },
        ]

        more_chunks = client.add_chunks(chunks=chunk_dicts, document_id=document.id)
        print(f"\nBatch added {len(more_chunks)} chunks (dict style)")
        for chunk in more_chunks:
            print(f"  - {chunk.text[:50]}... (ID: {chunk.id})")

        # Perform similarity search
        results = client.search(
            library_id=library.id,
            embedding=[0.15, 0.25, 0.35, 0.45, 0.55],
            k=5,
        )
        print(
            f"\nSearch results: {results.total} matches ({results.query_time_ms:.2f}ms)"
        )
        for result in results.results:
            print(f"  Score: {result.score:.4f} - {result.text[:50]}...")

    except ServerConnectionError as e:
        print(f"Connection error: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
