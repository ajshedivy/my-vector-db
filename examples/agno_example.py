#!/usr/bin/env python3
"""
Example: Using MyVectorDB with Agno Framework

This script demonstrates how to use the custom vector database
with Agno's Knowledge and Agent classes for semantic search.

Prerequisites:
1. API server running on http://localhost:8000
2. Knowledge base already populated (python scripts/load_data.py first)
3. Agno installed: pip install agno
4. Anthropic API key in environment (for agent LLM)

Note: This example assumes you've already loaded data using scripts/load_data.py
      which creates the "Python Programming Guide" library.
"""

from textwrap import dedent
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from my_vector_db.db import MyVectorDB
from agno.models.anthropic import Claude
from agno.knowledge.embedder.cohere import CohereEmbedder
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv

load_dotenv()


def print_db_info(vector_db: MyVectorDB) -> None:
    chunk_count = vector_db.get_count()
    print(f"✓ Connected to library: {vector_db.library_name}")
    print(f"  Chunks in database: {chunk_count}")

    if chunk_count == 0:
        print("\n⚠ Warning: No data found in library!")
        print("Run scripts/load_data.py first to load Python programming content.")
        return

    print("\n" + "=" * 70)
    print("Querying Agent with Python Knowledge")
    print("=" * 70)

    # Ask questions about Python programming
    questions = [
        "How do I install Python?",
        "What are the basic data types in Python?",
        "How do I define a function?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 70)


embedder = CohereEmbedder(
    id="embed-english-light-v3.0",  # 384 dimensions, matches test data
    input_type="search_document",
)

vector_db = MyVectorDB(
    api_base_url="http://localhost:8000",
    library_name="Python Programming Guide",
    embedder=embedder,
    index_type="flat",
)

contents_db = SqliteDb(db_file="tmp/data.db")

knowledge_base = Knowledge(
    name="My Awsome Python Knowledge Base",
    vector_db=vector_db,
    max_results=4,
    contents_db=contents_db,
)


def main():
    """Demonstrate Agno integration with MyVectorDB."""

    print_db_info(vector_db)

    knowledge_base.add_content(
        name="why python is great",
        text_content=dedent(
            """\
            Python is super cool because it has a simple syntax, a large standard library, 
            and a vibrant ecosystem of third-party packages. It's great for beginners and experts alike, 
            and can be used for web development, data science, machine learning, automation, and more!"""
        ),
        skip_if_exists=True,
    )

    agent = Agent(
        name="PythonTutor",
        knowledge=knowledge_base,
        model=Claude(id="claude-sonnet-4-5"),
        search_knowledge=True,
        read_chat_history=True,
        markdown=True,
        debug_mode=True,
    )

    agent.cli_app(stream=True, markdown=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
