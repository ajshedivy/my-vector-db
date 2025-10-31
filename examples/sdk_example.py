"""
Simple example demonstrating the Vector Database SDK.
"""

from my_vector_db.sdk import VectorDBClient, ServerConnectionError
from my_vector_db.sdk.models import SearchResult


def my_filter(result: SearchResult) -> bool:
    return "quick" in result.text.lower() and "fox" in result.text.lower()


def main():
    """Simple SDK usage example."""
    # Create client
    client = VectorDBClient(base_url="http://localhost:8000")

    try:
        # Create a library with FLAT index
        library = client.create_library(
            name="my_library",
            index_type="flat",
            index_config={"metric": "euclidean"},
        )
        print(f"Created library: {library.id}")
        print(library.index_config)

        # Create a document (requires library_id)
        document = client.create_document(
            library_id=library.id,
            name="my_document",
        )
        print(f"Created document: {document.id}")

        # add chunks to the document
        chunk = client.add_chunk(
            document_id=document.id,
            text="The quick brown fox jumps over the lazy dog",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        chunk2 = client.add_chunk(
            document_id=document.id,
            text="The slow brown cat jumps over the lazy dog",
            embedding=[0.1, 0.7, 0.3, 0.4, 0.2],
        )

        # Perform similarity search
        results = client.search(
            library_id=library.id,
            embedding=[0.15, 0.25, 0.35, 0.45, 0.55],
            k=5,
            filter_function=my_filter,
        )
        print(
            f"\nSearch results: {results.total} matches ({results.query_time_ms:.2f}ms)"
        )
        for result in results.results:
            print(f"  Score: {result.score:.4f} - {result.text[:50]}")
    except ServerConnectionError as e:
        print(f"Connection error: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
