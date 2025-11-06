from contextlib import asynccontextmanager
import os
from typing import AsyncIterator, Optional
from uuid import UUID
from my_vector_db.domain.models import Chunk
from my_vector_db.sdk import VectorDBClient
import cohere
import anyio

from fastmcp import FastMCP, Context
from dotenv import load_dotenv


class MyVectorDbContext:
    def __init__(self, client: VectorDBClient, cohere_client: cohere.Client) -> None:
        self.client = client
        self.cohere_client = cohere_client
        self._library_cache: dict[str, str] = {}  # name -> UUID mapping
        self._document_cache: dict[str, str] = {}  # name -> UUID mapping

    def generate_embedding(self, text: str) -> list[float]:
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
        response = self.cohere_client.embed(
            texts=[text], model="embed-english-light-v3.0", input_type="search_query"
        )
        embeddings = response.embeddings
        if isinstance(embeddings, list) and len(embeddings) > 0:
            return embeddings[0]
        raise ValueError("No embeddings returned from Cohere API")

    def resolve_library_id(self, library_name_or_id: str) -> str:
        """
        Resolve a library name or UUID to a UUID.

        Args:
            library_name_or_id: Library name (e.g., "My Library") or UUID
                              (case-insensitive, whitespace trimmed)

        Returns:
            Library UUID as string

        Raises:
            ValueError: If library not found
        """
        # Try parsing as UUID first
        try:
            UUID(library_name_or_id)
            return library_name_or_id
        except (ValueError, AttributeError):
            pass

        # Normalize the lookup key (lowercase, strip whitespace)
        normalized_name = library_name_or_id.strip().lower()

        # Look up by normalized name in cache
        if normalized_name in self._library_cache:
            return self._library_cache[normalized_name]

        # Refresh cache and try again
        libraries = self.client.list_libraries()
        self._library_cache = {
            lib.name.strip().lower(): str(lib.id) for lib in libraries
        }

        if normalized_name in self._library_cache:
            return self._library_cache[normalized_name]

        raise ValueError(f"Library '{library_name_or_id}' not found")

    def resolve_document_id(self, document_name_or_id: str) -> str:
        """
        Resolve a document name or UUID to a UUID.

        Args:
            document_name_or_id: Document name or UUID
                                (case-insensitive, whitespace trimmed)

        Returns:
            Document UUID as string

        Raises:
            ValueError: If document not found
        """
        # Try parsing as UUID first
        try:
            UUID(document_name_or_id)
            return document_name_or_id
        except (ValueError, AttributeError):
            pass

        # Normalize the lookup key (lowercase, strip whitespace)
        normalized_name = document_name_or_id.strip().lower()

        # Look up by normalized name in cache
        if normalized_name in self._document_cache:
            return self._document_cache[normalized_name]

        # Refresh cache by checking all libraries
        libraries = self.client.list_libraries()
        for library in libraries:
            documents = self.client.list_documents(library.id)
            for doc in documents:
                self._document_cache[doc.name.strip().lower()] = str(doc.id)

        if normalized_name in self._document_cache:
            return self._document_cache[normalized_name]

        raise ValueError(f"Document '{document_name_or_id}' not found")


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[MyVectorDbContext]:
    """Manage application lifecycle for MyVectorDb connector."""
    load_dotenv()

    connector = VectorDBClient(
        base_url=os.getenv("VECTORDB_BASE_URL", "http://localhost:8000")
    )
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is required")
    cohere_client = cohere.Client(cohere_api_key)

    try:
        yield MyVectorDbContext(connector, cohere_client)
    finally:
        pass


mcp = FastMCP(name="MyVectorDb", lifespan=server_lifespan)


@mcp.tool()
async def search(
    library_name: str, query_text: str, k: int = 5, ctx: Optional[Context] = None
) -> str:
    """Search for chunks in the specified library using semantic vector search.

    Args:
        library_name: Name or UUID of the library to search in
        query_text: Natural language query to search for
        k: Number of results to return (default: 5)

    Returns:
        Formatted search results with chunk text and metadata
    """
    if ctx is None:
        raise ValueError("Context is required")
    context: MyVectorDbContext = ctx.request_context.lifespan_context

    # Resolve library name to UUID
    library_id = context.resolve_library_id(library_name)

    # Generate embedding for the query
    embedding = context.generate_embedding(query_text)

    # Perform search (wrap sync call in async)
    search_results = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
        context.client.search, library_id, embedding, k
    )

    # Format output
    output = f"Vector search results for '{library_name}':\n\n"
    for result in search_results.results:
        output += f"Score: {result.score:.4f}\n"
        output += f"Text: {result.text}\n"
        output += f"Metadata: {result.metadata}\n"
        output += f"Document ID: {result.document_id}\n"
        output += f"Chunk ID: {result.chunk_id}\n\n"

    return output


@mcp.tool()
async def list_documents(library_name: str, ctx: Optional[Context] = None) -> str:
    """List all documents in the specified library.

    Args:
        library_name: Name or UUID of the library

    Returns:
        Formatted list of documents with their details
    """
    if ctx is None:
        raise ValueError("Context is required")
    context: MyVectorDbContext = ctx.request_context.lifespan_context

    # Resolve library name to UUID
    library_id = context.resolve_library_id(library_name)

    # List documents (wrap sync call in async)
    documents = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
        context.client.list_documents, library_id
    )

    # Format output
    output = f"Documents in library '{library_name}':\n\n"
    for doc in documents:
        output += f"Name: {doc.name}\n"
        output += f"ID: {doc.id}\n"
        output += f"Chunks: {len(doc.chunk_ids)}\n"
        output += f"Created: {doc.created_at}\n"
        output += f"Metadata: {doc.metadata}\n\n"

    return output


@mcp.tool()
async def list_libraries(ctx: Optional[Context] = None) -> str:
    """List all libraries in the vector database.

    Returns:
        Formatted list of all available libraries
    """
    if ctx is None:
        raise ValueError("Context is required")
    context: MyVectorDbContext = ctx.request_context.lifespan_context

    # List libraries (wrap sync call in async)
    libraries = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
        context.client.list_libraries
    )

    # Format output
    output = "Available libraries:\n\n"
    for lib in libraries:
        output += f"Name: {lib.name}\n"
        output += f"ID: {lib.id}\n"
        output += f"Index Type: {lib.index_type}\n"
        output += f"Created: {lib.created_at}\n"
        output += f"Metadata: {lib.metadata}\n\n"

    return output


@mcp.tool()
async def list_chunks(document_name: str, ctx: Optional[Context] = None) -> str:
    """List all chunks in the specified document.

    Args:
        document_name: Name or UUID of the document

    Returns:
        Formatted list of chunks with their content
    """
    if ctx is None:
        raise ValueError("Context is required")
    context: MyVectorDbContext = ctx.request_context.lifespan_context

    # Resolve document name to UUID
    document_id = context.resolve_document_id(document_name)

    # List chunks (wrap sync call in async)
    chunks = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
        context.client.list_chunks, document_id
    )

    # Format output
    output = f"Chunks in document '{document_name}':\n\n"
    for chunk in chunks:
        output += f"ID: {chunk.id}\n"
        output += f"Text: {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}\n"
        output += f"Metadata: {chunk.metadata}\n"
        output += f"Created: {chunk.created_at}\n\n"

    return output


@mcp.tool()
async def get_library(library_name: str, ctx: Optional[Context] = None) -> str:
    """Get detailed information about a specific library.

    Args:
        library_name: Name or UUID of the library

    Returns:
        Detailed library information
    """
    if ctx is None:
        raise ValueError("Context is required")
    context: MyVectorDbContext = ctx.request_context.lifespan_context

    # Resolve library name to UUID
    library_id = context.resolve_library_id(library_name)

    # Get library details (wrap sync call in async)
    library = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
        context.client.get_library, library_id
    )

    # Format output
    output = f"Library: {library.name}\n"
    output += f"ID: {library.id}\n"
    output += f"Index Type: {library.index_type}\n"
    output += f"Total Documents: {len(library.document_ids)}\n"
    output += f"Created: {library.created_at}\n"
    output += f"Updated: {library.updated_at}\n"
    output += f"Metadata: {library.metadata}\n"
    output += f"Index Config: {library.index_config}\n"

    return output


@mcp.tool()
async def get_document(document_name: str, ctx: Optional[Context] = None) -> str:
    """Get detailed information about a specific document.

    Args:
        document_name: Name or UUID of the document

    Returns:
        Detailed document information
    """
    if ctx is None:
        raise ValueError("Context is required")
    context: MyVectorDbContext = ctx.request_context.lifespan_context

    # Resolve document name to UUID
    document_id = context.resolve_document_id(document_name)

    # Get document details (wrap sync call in async)
    document = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
        context.client.get_document, document_id
    )

    # Format output
    output = f"Document: {document.name}\n"
    output += f"ID: {document.id}\n"
    output += f"Library ID: {document.library_id}\n"
    output += f"Total Chunks: {len(document.chunk_ids)}\n"
    output += f"Created: {document.created_at}\n"
    output += f"Updated: {document.updated_at}\n"
    output += f"Metadata: {document.metadata}\n"

    return output

@mcp.tool()
async def get_chunk(chunk_id: str, ctx: Optional[Context] = None) -> str:
    """Get detailed information about a specific chunk.

    Args:
        chunk_id: UUID of the chunk

    Returns:
        Detailed chunk information
    """
    if ctx is None:
        raise ValueError("Context is required")
    context: MyVectorDbContext = ctx.request_context.lifespan_context

    # Get chunk details (wrap sync call in async)
    chunk: Chunk = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
        context.client.get_chunk, chunk_id
    )

    # Format output
    output = f"Chunk ID: {chunk.id}\n"
    output += f"Text: {chunk.text}\n"
    output += f"Document ID: {chunk.document_id}\n"
    output += f"Metadata: {chunk.metadata}\n"
    output += f"Created: {chunk.created_at}\n"

    return output


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run MyVectorDb MCP server")
    parser.add_argument(
        "--host", default="http://localhost:8000", help="Vector DB host URL"
    )
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport")
    parser.add_argument("--http", action="store_true", help="Use HTTP transport")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    mcp.config = {  # type: ignore[attr-defined]
        "VECTORDB_BASE_URL": os.getenv("VECTORDB_BASE_URL", args.host),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY", ""),
    }

    if args.http:
        mcp.run(transport="http", host="127.0.0.1", port=8001)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
