#!/usr/bin/env python3
"""
Vector Database User Application

This interactive application demonstrates how to:
1. Connect to existing vector database libraries using the SDK
2. Generate embeddings using the Cohere API
3. Perform semantic searches to find relevant document chunks

Prerequisites:
- API server running on http://localhost:8000
- COHERE_API_KEY in .env file
- Data loaded (run load_data.py first)

Usage:
    python app.py
"""

import os
from typing import Any

import cohere
from dotenv import load_dotenv

from my_vector_db import VectorDBClient

# Load environment variables from .env file
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Validate API key
if not COHERE_API_KEY:
    raise ValueError(
        "COHERE_API_KEY not found in .env file. "
        "Please add your Cohere API key to the .env file."
    )

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# Initialize Vector DB SDK client
client = VectorDBClient(base_url=API_BASE_URL)


# ============================================================================
# Embedding Generation
# ============================================================================


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for the given text using Cohere API.

    Args:
        text: Input text to embed

    Returns:
        Embedding vector as a list of floats

    Note:
        Uses embed-english-light-v3.0 model which produces 384-dimensional
        embeddings, matching the test data format.
    """
    response = co.embed(
        texts=[text], model="embed-english-light-v3.0", input_type="search_query"
    )
    return response.embeddings[0]


# ============================================================================
# Search Functions
# ============================================================================


def search(library_id: str, query_text: str, k: int = 5) -> dict[str, Any]:
    """
    Search the library for relevant chunks based on a text query.

    This function:
    1. Generates an embedding for the query text using Cohere
    2. Sends the embedding to the vector database via SDK
    3. Returns the top k most relevant chunks

    Args:
        library_id: Library to search
        query_text: Natural language query
        k: Number of results to return (default: 5)

    Returns:
        Search results with scores and metadata
    """
    print(f"\n{'=' * 70}")
    print(f"Query: {query_text}")
    print(f"{'=' * 70}")
    print("Generating embedding with Cohere...")

    # Generate embedding for the query
    query_embedding = generate_embedding(query_text)
    print(f"✓ Generated {len(query_embedding)}-dimensional embedding")

    # Query the vector database using SDK
    print("Searching vector database...")
    result = client.search.query(library_id=library_id, embedding=query_embedding, k=k)

    print(f"✓ Query completed in {result.query_time_ms:.2f}ms")

    # Convert Pydantic models to dict for display
    return {
        "total": result.total,
        "query_time_ms": result.query_time_ms,
        "results": [
            {
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
                "chunk_id": str(r.chunk_id),
                "document_id": str(r.document_id),
            }
            for r in result.results
        ],
    }


def display_results(result: dict[str, Any], max_text_length: int = 200):
    """
    Display search results in a user-friendly format.

    Args:
        result: Search results from the API
        max_text_length: Maximum characters to display from each chunk
    """
    print(f"\nFound {result['total']} results:\n")

    for i, res in enumerate(result["results"], 1):
        print(f"{i}. Score: {res['score']:.4f}")

        # Truncate long text
        text = res["text"]
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        print(f"   Text: {text}")

        print(f"   Metadata: {res['metadata']}")
        print()


# ============================================================================
# Interactive Demo
# ============================================================================


def get_library_id(library_name: str) -> str:
    """Get library ID by name using SDK."""
    libraries = client.libraries.list()

    for lib in libraries:
        if lib.name == library_name:
            return str(lib.id)

    raise ValueError(f"Library '{library_name}' not found. Run load_data.py first.")


def run_demo():
    """Run an interactive demonstration of the vector database."""
    print("=" * 70)
    print("Vector Database User Application (using SDK)")
    print("=" * 70)
    print("\nThis app demonstrates semantic search using Cohere embeddings.")
    print("Make sure the API server is running on http://localhost:8000\n")

    try:
        # Connect to existing library
        library_name = "Python Programming Guide"
        print(f"Connecting to library: {library_name}...")

        try:
            library_id = get_library_id(library_name)
            print(f"✓ Connected to library: {library_id}")
        except ValueError as e:
            print(f"\n❌ {e}")
            print("\nPlease load data first:")
            print("  python load_data.py")
            return

        # Run example searches
        example_queries = [
            "How do I install Python?",
            "What are Python data types?",
            "How to define a function in Python?",
        ]

        print(f"\n{'=' * 70}")
        print("Running Example Searches")
        print(f"{'=' * 70}")

        for query_text in example_queries:
            result = search(library_id, query_text, k=5)
            display_results(result)
            print("-" * 70)

        # Interactive mode
        print(f"\n{'=' * 70}")
        print("Interactive Search Mode")
        print(f"{'=' * 70}")
        print("Enter your queries (or 'quit' to exit):\n")

        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break
                if not query:
                    continue

                result = search(library_id, query, k=5)
                display_results(result)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break

        print(f"\n{'=' * 70}")
        print("✓ Demo complete!")
        print(f"{'=' * 70}")

    except ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Please start the server first:")
        print("  python src/my_vector_db/main.py")
        print("  OR")
        print("  docker-compose up")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        # Close SDK client
        client.close()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    run_demo()
