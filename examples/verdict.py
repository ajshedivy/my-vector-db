#!/usr/bin/env python3
"""
Example: Using MyVectorDB with Agno Framework

This script demonstrates how to use the custom vector database
with Agno's Knowledge and Agent classes for semantic search.

Prerequisites:
1. API server running on http://localhost:8000
2. The Verdict text file in data/the_verdict.txt
3. Agno installed: pip install agno
4. Anthropic API key in environment (for agent LLM)
"""

from textwrap import dedent
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from my_vector_db.db import MyVectorDB
from agno.models.anthropic import Claude
from agno.knowledge.reader.text_reader import TextReader
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.embedder.cohere import CohereEmbedder
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

vector_db = MyVectorDB(
    api_base_url="http://localhost:8000",
    library_name="the verdict",
    index_type="flat",
)

knowledge_base = Knowledge(name="The verdict", vector_db=vector_db, max_results=4)

embedder = CohereEmbedder(id="embed-english-light-v3.0")


def main():
    """Demonstrate Agno integration with MyVectorDB."""

    questions = [
        "What is the main theme of 'The Verdict'?",
        "Who are the main characters in 'The Verdict'?",
        "What is the setting of the story?",
        "How does the story end?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 70)

    txt_reader = TextReader(
        chunking_strategy=SemanticChunking(
            embedder=embedder, chunk_size=500, similarity_threshold=0.7
        )
    )
    knowledge_base.add_content(
        name="the verdict by Edith Wharton",
        path="data/the_verdict.txt",
        reader=txt_reader,
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
