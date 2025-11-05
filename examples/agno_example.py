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
from utils import print_db_info

from dotenv import load_dotenv

load_dotenv()

# optional table for storing knowledge metadata
contents_db = SqliteDb(db_file="tmp/data.db")

embedder = CohereEmbedder(
    id="embed-english-light-v3.0",  # 384 dimensions, matches test data
    input_type="search_document",
)

vector_db = MyVectorDB(
    api_base_url="http://localhost:8000",
    library_name="Python Programming Guide",
    embedder=embedder,
)

knowledge_base = Knowledge(
    name="My Awsome Python Knowledge Base",
    vector_db=vector_db,
    max_results=4,
    contents_db=contents_db,
)

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


def main():
    """Demonstrate Agno integration with MyVectorDB."""

    print_db_info(vector_db)

    agent.cli_app(stream=True, markdown=True, stream_events=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
