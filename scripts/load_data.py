#!/usr/bin/env python3
"""
Load Test Data into Vector Database

This script loads pre-generated test data with embeddings into the vector database
via SDK

Usage:
    python load_data.py
"""

import json
from pathlib import Path
from typing import Any

from my_vector_db.sdk import VectorDBClient, ServerConnectionError

# API configuration
API_BASE_URL = "http://localhost:8000"
client = VectorDBClient(base_url=API_BASE_URL)


def load_test_data() -> dict[str, Any]:
    """Load pre-generated test data with embeddings."""
    test_data_path = Path(__file__).parent.parent / "data" / "test_data.json"
    with open(test_data_path) as f:
        return json.load(f)


def create_library_with_data(library_data: dict) -> str:
    """
    Create a library and populate it with documents and chunks.

    Args:
        library_data: Library data from test_data.json

    Returns:
        Library ID
    """
    print(f"\n{'=' * 70}")
    print(f"Creating library: {library_data['name']}")
    print(f"Index type: {library_data['index_type']}")
    print(f"{'=' * 70}")

    library = client.create_library(
        name=library_data["name"],
        index_type=library_data["index_type"],
        index_config=library_data.get("index_config", {}),
        metadata=library_data.get("metadata", {}),
    )

    print(f"✓ Created library: {library.id}")

    # Create documents and chunks
    for doc_data in library_data["documents"]:
        document = client.create_document(
            library_id=library.id,
            name=doc_data["name"],
            metadata=doc_data.get("metadata", {}),
        )
        print(f"  ✓ Created document: {document.name}")

        # Create chunks with pre-computed embeddings
        for chunk_data in doc_data["chunks"]:
            chunk = client.create_chunk(
                library_id=library.id,
                document_id=document.id,
                text=chunk_data["text"],
                embedding=chunk_data["embedding"],
                metadata=chunk_data.get("metadata", {}),
            )
            print(f"    ✓ Created chunk: {chunk.id}")

    total_chunks = sum(len(doc["chunks"]) for doc in library_data["documents"])
    print(
        f"\n✓ Library complete: {len(library_data['documents'])} documents, {total_chunks} chunks"
    )

    return library.id


def main():
    """Load all test data into the vector database."""
    print("=" * 70)
    print("Vector Database - Data Loader")
    print("=" * 70)
    print("\nLoading test data into the vector database...")
    print("Make sure the API server is running on http://localhost:8000\n")

    try:
        # Load test data
        print("Reading test data from file...")
        test_data = load_test_data()
        print(f"✓ Loaded {len(test_data['libraries'])} libraries from test_data.json")

        # Create all libraries
        library_ids = []
        for library_data in test_data["libraries"]:
            library_id = create_library_with_data(library_data)
            library_ids.append({"id": library_id, "name": library_data["name"]})

        # Summary
        print("\n" + "=" * 70)
        print("✓ Data Loading Complete!")
        print("=" * 70)
        print(f"\nSuccessfully loaded {len(library_ids)} libraries:\n")
        for lib in library_ids:
            print(f"  • {lib['name']}")
            print(f"    ID: {lib['id']}")

        print("\n" + "=" * 70)
        print("Next Steps:")
        print("=" * 70)
        print("1. Run app.py to test semantic search:")
        print("   python app.py")
        print("\n2. Run Agno example (requires: pip install agno):")
        print("   python examples/agno_example.py")
        print("\n3. Check API documentation:")
        print("   http://localhost:8000/docs")
        print()

    except ServerConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Please start the server first:")
        print("  python src/my_vector_db/main.py")
        print("  OR")
        print("  docker-compose up")
    except FileNotFoundError as e:
        print("\n❌ Error: Test data file not found")
        print(f"  {e}")
        print("\nMake sure you're running this from the project root directory.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
