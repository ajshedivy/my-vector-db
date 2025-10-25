"""
My Vector Database SDK Client

This module provides a simplified, flat API for interacting with the Vector Database.

Design principles applied:
- SOLID: Single responsibility, dependency inversion
- Pythonic: Flat is better than nested, explicit is better than implicit
- Type safety: Full static typing with Pydantic models
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

import httpx

from my_vector_db.domain.models import SearchFilters
from my_vector_db.sdk.errors import handle_errors
from my_vector_db.sdk.models import (
    Chunk,
    ChunkCreate,
    ChunkUpdate,
    Document,
    DocumentCreate,
    DocumentUpdate,
    IndexType,
    Library,
    LibraryCreate,
    LibraryUpdate,
    SearchQuery,
    SearchResponse,
)


class VectorDBClient:
    """
    Main client for interacting with the Vector Database API.

    Provides a flat, easy-to-discover API for all CRUD operations on libraries,
    documents, chunks, and search functionality.

    Example:
        >>> # Create client
        >>> client = VectorDBClient(base_url="http://localhost:8000")
        >>>
        >>> # Create library
        >>> library = client.create_library(name="my_library", index_type="hnsw")
        >>>
        >>> # Create document
        >>> document = client.create_document(
        ...     library_id=library.id,
        ...     name="my_document"
        ... )
        >>>
        >>> # Add chunk
        >>> chunk = client.create_chunk(
        ...     document_id=document.id,
        ...     text="Example text",
        ...     embedding=[0.1, 0.2, 0.3, ...]
        ... )
        >>>
        >>> # Search
        >>> results = client.search(
        ...     library_id=library.id,
        ...     embedding=[0.1, 0.2, 0.3, ...],
        ...     k=10
        ... )
        >>>
        >>> # Always close when done
        >>> client.close()
        >>>
        >>> # Or use context manager
        >>> with VectorDBClient(base_url="http://localhost:8000") as client:
        ...     library = client.create_library(name="my_library")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Vector Database client.

        Args:
            base_url: Base URL of the Vector Database API
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Configure HTTP client
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(timeout=self.timeout, headers=headers)

    # ========================================================================
    # Private HTTP Helper Methods
    # ========================================================================
    # These methods handle all HTTP communication and are decorated with
    # @handle_errors to automatically convert httpx errors to SDK exceptions.
    # This keeps public method bodies clean and focused on business logic.

    @handle_errors
    def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        """
        Internal GET request handler with automatic error handling.

        Args:
            path: API endpoint path (e.g., "/libraries")
            **kwargs: Additional arguments for httpx request

        Returns:
            Parsed JSON response as dictionary

        Raises:
            SDK exceptions via @handle_errors decorator
        """
        url = f"{self.base_url}{path}"
        return self._client.get(url, **kwargs)

    @handle_errors
    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        """
        Internal POST request handler with automatic error handling.

        Args:
            path: API endpoint path (e.g., "/libraries")
            **kwargs: Additional arguments for httpx request (json, data, etc.)

        Returns:
            Parsed JSON response as dictionary

        Raises:
            SDK exceptions via @handle_errors decorator
        """
        url = f"{self.base_url}{path}"
        return self._client.post(url, **kwargs)

    @handle_errors
    def _put(self, path: str, **kwargs: Any) -> httpx.Response:
        """
        Internal PUT request handler with automatic error handling.

        Args:
            path: API endpoint path (e.g., "/libraries/{id}")
            **kwargs: Additional arguments for httpx request (json, data, etc.)

        Returns:
            Parsed JSON response as dictionary

        Raises:
            SDK exceptions via @handle_errors decorator
        """
        url = f"{self.base_url}{path}"
        return self._client.put(url, **kwargs)

    @handle_errors
    def _delete(self, path: str, **kwargs: Any) -> httpx.Response:
        """
        Internal DELETE request handler with automatic error handling.

        Args:
            path: API endpoint path (e.g., "/libraries/{id}")
            **kwargs: Additional arguments for httpx request

        Returns:
            Parsed JSON response as dictionary (usually empty)

        Raises:
            SDK exceptions via @handle_errors decorator
        """
        url = f"{self.base_url}{path}"
        return self._client.delete(url, **kwargs)

    # ========================================================================
    # Library Operations
    # ========================================================================

    def create_library(
        self,
        name: str,
        index_type: str = "flat",
        metadata: Optional[Dict[str, Any]] = None,
        index_config: Optional[Dict[str, Any]] = None,
    ) -> Library:
        """
        Create a new library.

        Args:
            name: Library name
            index_type: Type of vector index ("flat", "hnsw")
            metadata: Optional metadata dictionary
            index_config: Optional index configuration

        Returns:
            Created Library instance

        Raises:
            ValidationError: If request validation fails
            ConnectionError: If cannot connect to the API
            TimeoutError: If request times out
            VectorDBError: For other errors

        Example:
            >>> library = client.create_library(
            ...     name="my_library",
            ...     index_type="hnsw",
            ...     metadata={"category": "research"}
            ... )
        """
        data = LibraryCreate(
            name=name,
            index_type=IndexType(index_type),
            metadata=metadata or {},
            index_config=index_config or {},
        )
        response_data = self._post("/libraries", json=data.model_dump())
        return Library(**response_data)

    def get_library(self, library_id: Union[UUID, str]) -> Library:
        """
        Retrieve a library by ID.

        Args:
            library_id: UUID of the library

        Returns:
            Library instance

        Raises:
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Example:
            >>> library = client.get_library(library_id="uuid-here")
        """
        response_data = self._get(f"/libraries/{library_id}")
        return Library(**response_data)

    def list_libraries(self) -> List[Library]:
        """
        List all libraries.

        Returns:
            List of Library instances

        Raises:
            VectorDBError: For errors

        Example:
            >>> libraries = client.list_libraries()
            >>> for lib in libraries:
            ...     print(f"{lib.name}: {lib.id}")
        """
        response_data = self._get("/libraries")
        return [Library(**lib) for lib in response_data]

    def update_library(
        self,
        library_id: Union[UUID, str],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index_type: Optional[str] = None,
        index_config: Optional[Dict[str, Any]] = None,
    ) -> Library:
        """
        Update an existing library.

        Args:
            library_id: UUID of the library
            name: Optional new name
            metadata: Optional new metadata
            index_type: Optional new index type
            index_config: Optional new index configuration

        Returns:
            Updated Library instance

        Raises:
            NotFoundError: If library doesn't exist
            ValidationError: If update validation fails
            VectorDBError: For other errors

        Example:
            >>> library = client.update_library(
            ...     library_id="uuid-here",
            ...     name="Updated Library Name"
            ... )
        """
        data = LibraryUpdate(
            name=name,
            metadata=metadata,
            index_type=index_type,
            index_config=index_config,
        )
        response_data = self._put(f"/libraries/{library_id}", json=data.model_dump())
        return Library(**response_data)

    def delete_library(self, library_id: Union[UUID, str]) -> None:
        """
        Delete a library and all its documents and chunks.

        Args:
            library_id: UUID of the library

        Raises:
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Example:
            >>> client.delete_library(library_id="uuid-here")
        """
        self._delete(f"/libraries/{library_id}")

    def build_index(self, library_id: Union[UUID, str]) -> Dict[str, Any]:
        """
        Explicitly build/rebuild the vector index for a library.

        Args:
            library_id: UUID of the library

        Returns:
            Build status information

        Raises:
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Example:
            >>> result = client.build_index(library_id="uuid-here")
            >>> print(f"Index built with {result['total_vectors']} vectors")
        """
        raise NotImplementedError

    # ========================================================================
    # Document Operations
    # ========================================================================

    def create_document(
        self,
        library_id: Union[UUID, str],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Create a new document in a library.

        Args:
            library_id: UUID of the parent library
            name: Document name
            metadata: Optional metadata dictionary

        Returns:
            Created Document instance

        Raises:
            ValidationError: If request validation fails
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Example:
            >>> document = client.create_document(
            ...     library_id=library.id,
            ...     name="Research Paper",
            ...     metadata={"author": "John Doe"}
            ... )
        """
        data = DocumentCreate(
            library_id=UUID(str(library_id)),
            name=name,
            metadata=metadata or {},
        )

        response = self._post(
            f"/libraries/{library_id}/documents",
            json=data.model_dump(mode="json"),
        )
        return Document(**response)

    def get_document(
        self, library_id: Union[UUID, str], document_id: Union[UUID, str]
    ) -> Document:
        """
        Retrieve a document by ID.

        Args:
            library_id: UUID of the parent library
            document_id: UUID of the document

        Returns:
            Document instance

        Raises:
            NotFoundError: If library or document doesn't exist
            VectorDBError: For other errors

        Example:
            >>> document = client.get_document(
            ...     library_id=library.id,
            ...     document_id="doc-uuid"
            ... )
        """
        response = self._get(f"/libraries/{library_id}/documents/{document_id}")
        return Document(**response)

    def list_documents(self, library_id: Union[UUID, str]) -> List[Document]:
        """
        List all documents in a library.

        Args:
            library_id: UUID of the library

        Returns:
            List of Document instances

        Raises:
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Example:
            >>> documents = client.list_documents(library_id=library.id)
            >>> for doc in documents:
            ...     print(f"{doc.name}: {len(doc.chunk_ids)} chunks")
        """
        response = self._get(f"/libraries/{library_id}/documents")
        return [Document(**doc) for doc in response]

    def update_document(
        self,
        library_id: Union[UUID, str],
        document_id: Union[UUID, str],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Update an existing document.

        Args:
            library_id: UUID of the parent library
            document_id: UUID of the document
            name: Optional new name
            metadata: Optional new metadata

        Returns:
            Updated Document instance

        Raises:
            NotFoundError: If library or document doesn't exist
            ValidationError: If update validation fails
            VectorDBError: For other errors

        Example:
            >>> document = client.update_document(
            ...     library_id=library.id,
            ...     document_id=doc.id,
            ...     metadata={"status": "reviewed"}
            ... )
        """
        data = DocumentUpdate(name=name, metadata=metadata)

        response = self._put(
            f"/libraries/{library_id}/documents/{document_id}",
            json=data.model_dump(exclude_none=True),
        )
        return Document(**response)

    def delete_document(
        self, library_id: Union[UUID, str], document_id: Union[UUID, str]
    ) -> None:
        """
        Delete a document and all its chunks.

        Args:
            library_id: UUID of the parent library
            document_id: UUID of the document

        Raises:
            NotFoundError: If library or document doesn't exist
            VectorDBError: For other errors

        Example:
            >>> client.delete_document(library_id=library.id, document_id=doc.id)
        """
        self._delete(f"/libraries/{library_id}/documents/{document_id}")

    # ========================================================================
    # Chunk Operations
    # ========================================================================

    def create_chunk(
        self,
        library_id: Union[UUID, str],
        document_id: Union[UUID, str],
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Create a new chunk in a document.

        Note: library_id is not required here as chunks are accessed via document_id.
        The API will automatically determine the library from the document.

        Args:
            document_id: UUID of the parent document
            text: Text content of the chunk
            embedding: Vector embedding of the text
            metadata: Optional metadata dictionary

        Returns:
            Created Chunk instance

        Raises:
            ValidationError: If request validation fails
            NotFoundError: If document doesn't exist
            VectorDBError: For other errors

        Example:
            >>> chunk = client.create_chunk(
            ...     document_id=document.id,
            ...     text="This is a chunk of text",
            ...     embedding=[0.1, 0.2, 0.3, ...],
            ...     metadata={"page": 1}
            ... )
        """
        data = ChunkCreate(
            document_id=UUID(str(document_id)),
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )

        response = self._post(
            f"/libraries/{library_id}/documents/{document_id}/chunks",
            json=data.model_dump(mode="json"),
        )
        return Chunk(**response)

    def get_chunk(
        self,
        library_id: Union[UUID, str],
        document_id: Union[UUID, str],
        chunk_id: Union[UUID, str],
    ) -> Chunk:
        """
        Retrieve a chunk by ID.

        Args:
            library_id: UUID of the parent library
            document_id: UUID of the parent document
            chunk_id: UUID of the chunk

        Returns:
            Chunk instance

        Raises:
            NotFoundError: If library, document, or chunk doesn't exist
            VectorDBError: For other errors

        Example:
            >>> chunk = client.get_chunk(
            ...     library_id=library.id,
            ...     document_id=document.id,
            ...     chunk_id="chunk-uuid"
            ... )
        """
        response = self._get(
            f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}"
        )
        return Chunk(**response)

    def list_chunks(
        self, library_id: Union[UUID, str], document_id: Union[UUID, str]
    ) -> List[Chunk]:
        """
        List all chunks in a document.

        Args:
            library_id: UUID of the parent library
            document_id: UUID of the document

        Returns:
            List of Chunk instances

        Raises:
            NotFoundError: If library or document doesn't exist
            VectorDBError: For other errors

        Example:
            >>> chunks = client.list_chunks(
            ...     library_id=library.id,
            ...     document_id=document.id
            ... )
            >>> for chunk in chunks:
            ...     print(f"{chunk.text[:50]}...")
        """
        response = self._get(f"/libraries/{library_id}/documents/{document_id}/chunks")
        return [Chunk(**chunk) for chunk in response]

    def update_chunk(
        self,
        library_id: Union[UUID, str],
        document_id: Union[UUID, str],
        chunk_id: Union[UUID, str],
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Update an existing chunk.

        Args:
            library_id: UUID of the parent library
            document_id: UUID of the parent document
            chunk_id: UUID of the chunk
            text: Optional new text
            embedding: Optional new embedding
            metadata: Optional new metadata

        Returns:
            Updated Chunk instance

        Raises:
            NotFoundError: If library, document, or chunk doesn't exist
            ValidationError: If update validation fails
            VectorDBError: For other errors

        Example:
            >>> chunk = client.update_chunk(
            ...     library_id=library.id,
            ...     document_id=document.id,
            ...     chunk_id=chunk.id,
            ...     text="Updated text content"
            ... )
        """
        data = ChunkUpdate(text=text, embedding=embedding, metadata=metadata)

        response = self._put(
            f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}",
            json=data.model_dump(exclude_none=True),
        )
        return Chunk(**response)

    def delete_chunk(
        self,
        library_id: Union[UUID, str],
        document_id: Union[UUID, str],
        chunk_id: Union[UUID, str],
    ) -> None:
        """
        Delete a chunk.

        Args:
            library_id: UUID of the parent library
            document_id: UUID of the parent document
            chunk_id: UUID of the chunk

        Raises:
            NotFoundError: If library, document, or chunk doesn't exist
            VectorDBError: For other errors

        Example:
            >>> client.delete_chunk(
            ...     library_id=library.id,
            ...     document_id=document.id,
            ...     chunk_id=chunk.id
            ... )
        """
        self._delete(
            f"/libraries/{library_id}/documents/{document_id}/chunks/{chunk_id}"
        )

    # ========================================================================
    # Search Operations
    # ========================================================================

    def search(
        self,
        library_id: Union[UUID, str],
        embedding: List[float],
        k: int = 10,
        filters: Optional[Union[SearchFilters, Dict[str, Any], Callable]] = None,
    ) -> SearchResponse:
        """
        Perform k-nearest neighbor vector search in a library.

        Args:
            library_id: UUID of the library to search in
            embedding: Query vector embedding
            k: Number of nearest neighbors to return (1-1000)
            filters: Search filters. Can be:
                    - SearchFilters object (declarative or custom)
                    - Dict (converted to SearchFilters with validation)
                    - Callable (wrapped in SearchFilters as custom_filter)

        Returns:
            SearchResponse with matching chunks and query time

        Raises:
            ValidationError: If request validation fails
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Note:
            Custom filter functions are applied CLIENT-SIDE after fetching from the API.
            The SDK over-fetches (k*3) results and filters them locally. This means:
            - Custom filters work seamlessly with REST API
            - More network transfer but enables text/custom filtering
            - Filter functions receive Chunk objects with text and metadata
              (embedding and created_at fields are not available for client-side filters)

        Examples:
            # Using dict (declarative filters only)
            >>> results = client.search(
            ...     library_id=library.id,
            ...     embedding=[0.1, 0.2, 0.3, ...],
            ...     k=5,
            ...     filters={
            ...         "metadata": {
            ...             "operator": "and",
            ...             "filters": [
            ...                 {"field": "category", "operator": "eq", "value": "tech"}
            ...             ]
            ...         }
            ...     }
            ... )

            # Using SearchFilters object (supports custom functions)
            >>> from my_vector_db.domain.models import SearchFilters, FilterGroup, MetadataFilter
            >>> filters = SearchFilters(
            ...     metadata=FilterGroup(
            ...         operator=LogicalOperator.AND,
            ...         filters=[
            ...             MetadataFilter(field="category", operator=FilterOperator.EQUALS, value="tech")
            ...         ]
            ...     )
            ... )
            >>> results = client.search(library_id=library.id, embedding=vec, k=5, filters=filters)

            # Using custom filter function (SDK only - not via REST API)
            >>> filters = SearchFilters(
            ...     custom_filter=lambda chunk: chunk.metadata.get("score", 0) > 50
            ... )
            >>> results = client.search(library_id=library.id, embedding=vec, k=10, filters=filters)

            # Or pass the callable directly (convenience shortcut)
            >>> results = client.search(
            ...     library_id=library.id,
            ...     embedding=vec,
            ...     k=10,
            ...     filters=lambda chunk: chunk.metadata.get("score", 0) > 50
            ... )
        """
        # Convert filters to SearchFilters if needed
        custom_filter_func = None
        if filters is not None:
            if isinstance(filters, dict):
                # Dict -> SearchFilters (with validation)
                filters = SearchFilters(**filters)
            elif callable(filters):
                # Callable -> SearchFilters with custom_filter
                filters = SearchFilters(custom_filter=filters)
            # else: already SearchFilters, use as-is

            # Extract custom filter for client-side filtering
            if filters.custom_filter is not None:
                custom_filter_func = filters.custom_filter

        # Determine fetch size (over-fetch if client-side filtering needed)
        fetch_k = k * 3 if custom_filter_func else k

        # Create search query (filters is now SearchFilters or None)
        # Note: custom_filter is excluded during serialization (exclude=True)
        data = SearchQuery(
            embedding=embedding,
            k=fetch_k,
            filters=filters,
        )

        response = self._post(
            f"/libraries/{library_id}/query", json=data.model_dump(mode="json")
        )
        search_response = SearchResponse(**response)

        # Apply client-side filtering if custom filter was provided
        if custom_filter_func:
            search_response = self._apply_client_side_filter(
                search_response, custom_filter_func, k
            )

        return search_response

    def _apply_client_side_filter(
        self,
        response: SearchResponse,
        filter_func: Callable,
        k: int,
    ) -> SearchResponse:
        """
        Apply custom filter function client-side to search results.

        This enables custom filter functions to work with the REST API by:
        1. Over-fetching results from the API (k*3)
        2. Applying the custom filter client-side
        3. Returning top k results that pass the filter

        Args:
            response: SearchResponse with over-fetched results
            filter_func: Custom filter function (chunk) -> bool
            k: Original requested number of results

        Returns:
            New SearchResponse with filtered results (up to k items)
        """
        from datetime import datetime

        filtered_results = []

        for result in response.results:
            # Create a temporary Chunk object from SearchResult
            # Note: embedding and created_at are not available, set to defaults
            chunk = Chunk(
                id=result.chunk_id,
                text=result.text,
                embedding=[],  # Not available in SearchResult
                metadata=result.metadata,
                document_id=result.document_id,
                created_at=datetime.now(),  # Not available in SearchResult
            )

            # Apply custom filter
            try:
                if filter_func(chunk):
                    filtered_results.append(result)
            except Exception:
                # Fail gracefully if filter raises exception
                continue

            # Stop if we have enough results
            if len(filtered_results) >= k:
                break

        # Return new SearchResponse with filtered results
        return SearchResponse(
            results=filtered_results,
            total=len(filtered_results),
            query_time_ms=response.query_time_ms,  # Keep original query time
        )

    # ========================================================================
    # Context Manager and Cleanup
    # ========================================================================

    def close(self) -> None:
        """
        Close the HTTP client and release resources.

        Example:
            >>> client = VectorDBClient()
            >>> try:
            ...     # Use client
            ...     pass
            ... finally:
            ...     client.close()
        """
        self._client.close()

    def __enter__(self) -> "VectorDBClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit - close HTTP client."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the client."""
        return f"VectorDBClient(base_url='{self.base_url}')"
