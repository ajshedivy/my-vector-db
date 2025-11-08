#!/usr/bin/env python3
"""
Generate test data with real embeddings from Cohere API.

This script creates test libraries, documents, and chunks with embeddings
and saves them to a JSON file for use in testing.
"""

import json
import os
import cohere
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# Cohere API configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBEDDING_MODEL = "embed-english-light-v3.0"  # 384 dimensions


def generate_test_data() -> dict[str, Any]:
    """
    Generate comprehensive test data with multiple libraries, documents, and chunks.

    Returns:
        Dictionary containing test data structure
    """

    # Initialize Cohere client
    co = cohere.Client(COHERE_API_KEY)

    # Define test data structure
    test_data = {"libraries": []}

    # ========================================================================
    # Library 1: Python Programming Guide (Small dataset for basic tests)
    # ========================================================================

    python_chunks = [
        # Document 1: Getting Started
        "Python is a high-level, interpreted programming language known for its simplicity and readability.",
        "To install Python, download it from python.org and follow the installation instructions for your operating system.",
        "Variables in Python are created by assigning a value using the equals sign, like x = 5 or name = 'Alice'.",
        "Python supports multiple data types including integers, floats, strings, lists, tuples, and dictionaries.",
        "Functions in Python are defined using the def keyword followed by the function name and parameters.",
        # Document 2: Web Development
        "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
        "Flask is a lightweight WSGI web application framework that is easy to get started with and scales well.",
        "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design.",
        "Uvicorn is an ASGI web server implementation for Python, commonly used to run FastAPI applications.",
        "Pydantic provides data validation using Python type annotations, making it perfect for API development.",
        # Document 3: Data Science
        "NumPy is the fundamental package for numerical computing in Python, providing support for arrays and matrices.",
        "Pandas offers data structures and operations for manipulating numerical tables and time series data.",
        "Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.",
        "Scikit-learn provides simple and efficient tools for predictive data analysis and machine learning.",
        "Jupyter notebooks are an open-source web application for creating documents with live code and visualizations.",
    ]

    # ========================================================================
    # Library 2: Machine Learning Fundamentals (Medium dataset)
    # ========================================================================

    ml_chunks = [
        # Document 1: Basics
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Supervised learning uses labeled training data to learn the mapping between inputs and outputs.",
        "Unsupervised learning finds hidden patterns in data without pre-existing labels or categories.",
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
        # Document 2: Algorithms
        "Linear regression is a supervised learning algorithm that models relationships between variables.",
        "Decision trees are tree-structured models that make predictions by learning simple decision rules.",
        "Random forests combine multiple decision trees to improve accuracy and reduce overfitting.",
        "Support vector machines find the optimal hyperplane that separates different classes in the data.",
        "K-means clustering partitions data into K distinct clusters based on feature similarity.",
        # Document 3: Neural Networks
        "Neural networks consist of interconnected nodes organized in layers that process information.",
        "Backpropagation is the algorithm used to train neural networks by computing gradients efficiently.",
        "Convolutional neural networks are designed for processing grid-like data such as images.",
        "Recurrent neural networks process sequential data by maintaining internal state across time steps.",
        "Transformers use self-attention mechanisms to process sequential data in parallel for better performance.",
    ]

    # ========================================================================
    # Library 3: Vector Search & Embeddings (Technical/Advanced)
    # ========================================================================

    vector_chunks = [
        # Document 1: Embeddings
        "Vector embeddings are numerical representations of data that capture semantic meaning in high-dimensional space.",
        "Word embeddings like Word2Vec map words to vectors where similar words have similar representations.",
        "Sentence embeddings encode entire sentences into fixed-size vectors for semantic similarity comparison.",
        "Embedding models are trained to place semantically similar items close together in vector space.",
        "Dimensionality reduction techniques like PCA can compress embeddings while preserving important information.",
        # Document 2: Search Algorithms
        "Approximate nearest neighbor search trades perfect accuracy for significant speed improvements in large datasets.",
        "HNSW (Hierarchical Navigable Small World) uses a graph-based approach for efficient approximate search.",
        "Ball trees partition vector space into nested hyperspheres for efficient exact nearest neighbor search.",
        "KD-trees recursively divide space along coordinate axes, working well in low-dimensional spaces.",
        "Locality-sensitive hashing groups similar vectors into buckets to speed up similarity search.",
        # Document 3: Applications
        "Semantic search finds documents based on meaning rather than exact keyword matches.",
        "Recommendation systems use embeddings to find similar items or users for personalized suggestions.",
        "Image similarity search encodes images as vectors to find visually similar content.",
        "Duplicate detection identifies near-identical content by measuring embedding similarity.",
        "Clustering algorithms group embeddings to discover patterns and categorize data automatically.",
    ]

    print("Generating embeddings from Cohere API...")
    print(
        f"Total chunks to embed: {len(python_chunks) + len(ml_chunks) + len(vector_chunks)}"
    )

    # Generate embeddings for all chunks
    all_texts = python_chunks + ml_chunks + vector_chunks

    print("Calling Cohere API...")
    response = co.embed(
        texts=all_texts, model=EMBEDDING_MODEL, input_type="search_document"
    )
    embeddings = response.embeddings
    print(
        f"✓ Generated {len(embeddings)} embeddings ({len(embeddings[0])} dimensions each)"
    )

    # Split embeddings back into groups
    python_embeddings = embeddings[: len(python_chunks)]
    ml_embeddings = embeddings[len(python_chunks) : len(python_chunks) + len(ml_chunks)]
    vector_embeddings = embeddings[len(python_chunks) + len(ml_chunks) :]

    # ========================================================================
    # Build Library 1: Python Programming
    # ========================================================================

    library1 = {
        "name": "Python Programming Guide",
        "index_type": "flat",
        "metadata": {
            "description": "A comprehensive guide to Python programming",
            "language": "en",
            "difficulty": "beginner-to-intermediate",
        },
        "documents": [
            {
                "name": "Getting Started with Python",
                "metadata": {
                    "author": "Alice Johnson",
                    "category": "basics",
                    "date": "2024-01-15",
                },
                "chunks": [
                    {
                        "text": python_chunks[i],
                        "embedding": python_embeddings[i],
                        "metadata": {"section": "basics", "page": i + 1},
                    }
                    for i in range(5)
                ],
            },
            {
                "name": "Web Development with Python",
                "metadata": {
                    "author": "Bob Smith",
                    "category": "web-dev",
                    "date": "2024-02-10",
                },
                "chunks": [
                    {
                        "text": python_chunks[i],
                        "embedding": python_embeddings[i],
                        "metadata": {"section": "web", "page": i - 4},
                    }
                    for i in range(5, 10)
                ],
            },
            {
                "name": "Data Science Tools",
                "metadata": {
                    "author": "Charlie Davis",
                    "category": "data-science",
                    "date": "2024-03-05",
                },
                "chunks": [
                    {
                        "text": python_chunks[i],
                        "embedding": python_embeddings[i],
                        "metadata": {"section": "data-science", "page": i - 9},
                    }
                    for i in range(10, 15)
                ],
            },
        ],
    }

    # ========================================================================
    # Build Library 2: Machine Learning
    # ========================================================================

    library2 = {
        "name": "Machine Learning Fundamentals",
        "index_type": "ivf",
        "index_config": {"nlist": 5, "nprobe": 2, "metric": "cosine"},
        "metadata": {
            "description": "Core concepts in machine learning and AI",
            "language": "en",
            "difficulty": "intermediate-to-advanced",
        },
        "documents": [
            {
                "name": "ML Basics and Concepts",
                "metadata": {
                    "author": "Dr. Emily Chen",
                    "category": "fundamentals",
                    "date": "2024-01-20",
                },
                "chunks": [
                    {
                        "text": ml_chunks[i],
                        "embedding": ml_embeddings[i],
                        "metadata": {"topic": "basics", "page": i + 1},
                    }
                    for i in range(5)
                ],
            },
            {
                "name": "Common ML Algorithms",
                "metadata": {
                    "author": "Dr. Emily Chen",
                    "category": "algorithms",
                    "date": "2024-02-15",
                },
                "chunks": [
                    {
                        "text": ml_chunks[i],
                        "embedding": ml_embeddings[i],
                        "metadata": {"topic": "algorithms", "page": i - 4},
                    }
                    for i in range(5, 10)
                ],
            },
            {
                "name": "Neural Networks Deep Dive",
                "metadata": {
                    "author": "Frank Wilson",
                    "category": "deep-learning",
                    "date": "2024-03-10",
                },
                "chunks": [
                    {
                        "text": ml_chunks[i],
                        "embedding": ml_embeddings[i],
                        "metadata": {"topic": "neural-nets", "page": i - 9},
                    }
                    for i in range(10, 15)
                ],
            },
        ],
    }

    # ========================================================================
    # Build Library 3: Vector Search
    # ========================================================================

    library3 = {
        "name": "Vector Search and Embeddings",
        "index_type": "flat",
        "index_config": {"leaf_size": 40, "metric": "cosine"},
        "metadata": {
            "description": "Advanced concepts in vector search and embeddings",
            "language": "en",
            "difficulty": "advanced",
        },
        "documents": [
            {
                "name": "Understanding Embeddings",
                "metadata": {
                    "author": "Grace Lee",
                    "category": "embeddings",
                    "date": "2024-02-01",
                },
                "chunks": [
                    {
                        "text": vector_chunks[i],
                        "embedding": vector_embeddings[i],
                        "metadata": {"concept": "embeddings", "page": i + 1},
                    }
                    for i in range(5)
                ],
            },
            {
                "name": "Search Algorithms Explained",
                "metadata": {
                    "author": "Henry Martinez",
                    "category": "algorithms",
                    "date": "2024-02-20",
                },
                "chunks": [
                    {
                        "text": vector_chunks[i],
                        "embedding": vector_embeddings[i],
                        "metadata": {"concept": "search", "page": i - 4},
                    }
                    for i in range(5, 10)
                ],
            },
            {
                "name": "Real-World Applications",
                "metadata": {
                    "author": "Isabel Garcia",
                    "category": "applications",
                    "date": "2024-03-01",
                },
                "chunks": [
                    {
                        "text": vector_chunks[i],
                        "embedding": vector_embeddings[i],
                        "metadata": {"concept": "applications", "page": i - 9},
                    }
                    for i in range(10, 15)
                ],
            },
        ],
    }

    test_data["libraries"] = [library1, library2, library3]

    # ========================================================================
    # Generate sample queries with embeddings
    # ========================================================================

    query_texts = [
        "How do I get started with Python programming?",
        "What are the best Python frameworks for building web APIs?",
        "Explain how neural networks work and how they learn",
        "What is the difference between supervised and unsupervised learning?",
        "How does HNSW algorithm work for vector search?",
        "What are vector embeddings and why are they useful?",
    ]

    print("\nGenerating query embeddings...")
    query_response = co.embed(
        texts=query_texts, model=EMBEDDING_MODEL, input_type="search_query"
    )
    query_embeddings = query_response.embeddings

    test_data["sample_queries"] = [
        {
            "text": text,
            "embedding": embedding,
            "expected_library": [
                "Python Programming Guide",
                "Python Programming Guide",
                "Machine Learning Fundamentals",
                "Machine Learning Fundamentals",
                "Vector Search and Embeddings",
                "Vector Search and Embeddings",
            ][i],
        }
        for i, (text, embedding) in enumerate(zip(query_texts, query_embeddings))
    ]

    print(f"✓ Generated {len(query_embeddings)} query embeddings")

    return test_data


def main():
    """Main function to generate and save test data."""

    print("=" * 70)
    print("Generating Test Data with Cohere Embeddings")
    print("=" * 70)

    # Generate test data
    test_data = generate_test_data()

    # Calculate statistics
    total_chunks = sum(
        len(chunk)
        for library in test_data["libraries"]
        for doc in library["documents"]
        for chunk in doc["chunks"]
    )

    print("\n" + "=" * 70)
    print("Test Data Summary")
    print("=" * 70)
    print(f"Libraries: {len(test_data['libraries'])}")
    print(
        f"Total documents: {sum(len(lib['documents']) for lib in test_data['libraries'])}"
    )
    print(f"Total chunks: {total_chunks}")
    print(f"Sample queries: {len(test_data['sample_queries'])}")
    print(
        f"Embedding dimensions: {len(test_data['libraries'][0]['documents'][0]['chunks'][0]['embedding'])}"
    )

    # Save to JSON file
    output_path = Path(__file__).parent.parent / "data" / "test_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(test_data, f, indent=2)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully ({file_size_mb:.2f} MB)")

    print("\n" + "=" * 70)
    print("✓ Test data generation complete!")
    print("=" * 70)
    print("\nUse this file in your tests:")
    print("  with open('tests/fixtures/test_data.json') as f:")
    print("      data = json.load(f)")


if __name__ == "__main__":
    main()
