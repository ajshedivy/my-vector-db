"""
My Vector Database SDK Client

This module provides a simplified, flat API for interacting with the Vector Database.

Design principles applied:
- SOLID: Single responsibility, dependency inversion
- Pythonic: Flat is better than nested, explicit is better than implicit
- Type safety: Full static typing with Pydantic models
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

import httpx

from my_vector_db.domain.models import (
    BuildIndexResult,
    SearchFilters,
    SearchFiltersWithCallable,
)
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
    SearchResult,
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

    def get_health_status(self) -> Dict[str, Any]:
        """
        Retrieve health status of the Vector Database service.

        Returns:
            Dictionary with health status information

        Raises:
            ConnectionError: If cannot connect to the API
            TimeoutError: If request times out
            VectorDBError: For other errors

        Example:
            >>> status = client.get_health_status()
            >>> print(status)
        """
        return self._get("/health")

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
        library: Union[Library, UUID, str],
        *,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index_type: Optional[IndexType] = None,
        index_config: Optional[Dict[str, Any]] = None,
    ) -> Library:
        """
        Update an existing library.

        Pass either a Library object (fetch-modify-update) or a library ID with fields to update.
        When passing a Library object, fields can be overridden via keyword arguments.

        Args:
            library: Library object OR library ID (UUID/string)
            name: Override/set name (optional)
            metadata: Override/set metadata (optional)
            index_type: Override/set index type (optional)
            index_config: Override/set index configuration (optional)

        Returns:
            Updated Library instance

        Raises:
            NotFoundError: If library doesn't exist
            ValidationError: If update validation fails
            VectorDBError: For other errors

        Examples:
            # Object-based update (fetch-modify-update)
            >>> library = client.get_library(library_id)
            >>> library.name = "Updated Name"
            >>> library = client.update_library(library)

            # ID-based update (specific fields)
            >>> library = client.update_library(library_id, name="New Name")

            # Hybrid: fetch object, override one field
            >>> library = client.get_library(library_id)
            >>> library = client.update_library(library, metadata={"new": "data"})
        """
        if isinstance(library, Library):
            # Object-based: use fields from library, allow kwargs to override
            library_id = library.id
            data = LibraryUpdate(
                name=name if name is not None else library.name,
                metadata=metadata if metadata is not None else library.metadata,
                index_type=index_type if index_type is not None else library.index_type,
                index_config=(
                    index_config if index_config is not None else library.index_config
                ),
            )
        else:
            # ID-based: must provide at least one field
            library_id = UUID(str(library))

            # Build update data from provided fields only
            if all(v is None for v in [name, metadata, index_type, index_config]):
                raise ValueError(
                    "When updating by ID, must provide at least one field: "
                    "'name', 'metadata', 'index_type', or 'index_config'"
                )

            data = LibraryUpdate(
                name=name,
                metadata=metadata,
                index_type=index_type,
                index_config=index_config,
            )

        response_data = self._put(
            f"/libraries/{library_id}",
            json=data.model_dump(exclude_none=True),
        )
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

    def build_index(self, library_id: Union[UUID, str]) -> BuildIndexResult:
        """
        Explicitly build/rebuild the vector index for a library.

        For HNSW indexes, this should be called after adding/updating chunks
        to optimize search performance. FLAT indexes automatically update,
        but this can still be used to validate the index.

        Args:
            library_id: UUID of the library

        Returns:
            BuildIndexResult with build information:
                - library_id: UUID of the library
                - total_vectors: Number of vectors indexed
                - dimension: Vector dimension
                - index_type: Index type (flat, hnsw)
                - index_config: Index configuration parameters

        Raises:
            NotFoundError: If library doesn't exist
            ValidationError: If no chunks or invalid dimensions
            VectorDBError: For other errors

        Example:
            >>> result = client.build_index(library_id="uuid-here")
            >>> print(f"Index built with {result.total_vectors} vectors")
            >>> print(f"Dimension: {result.dimension}")
            >>> print(f"Index type: {result.index_type}")
            >>> print(f"Config: {result.index_config}")
        """
        response = self._post(f"/libraries/{library_id}/index/build")
        return BuildIndexResult(**response)

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

    def get_document(self, document_id: Union[UUID, str]) -> Document:
        """
        Retrieve a document by ID.

        Args:
            document_id: UUID of the document

        Returns:
            Document instance

        Raises:
            NotFoundError: If document doesn't exist
            VectorDBError: For other errors

        Example:
            >>> document = client.get_document(document_id="doc-uuid")
        """
        response = self._get(f"/documents/{document_id}")
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
        document: Union[Document, UUID, str],
        *,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Update an existing document.

        Pass either a Document object (fetch-modify-update) or a document ID with fields to update.
        When passing a Document object, fields can be overridden via keyword arguments.

        Args:
            document: Document object OR document ID (UUID/string)
            name: Override/set name (optional)
            metadata: Override/set metadata (optional)

        Returns:
            Updated Document instance

        Raises:
            NotFoundError: If document doesn't exist
            ValidationError: If update validation fails
            VectorDBError: For other errors

        Examples:
            # Object-based update (fetch-modify-update)
            >>> document = client.get_document(document_id)
            >>> document.name = "Updated Name"
            >>> document = client.update_document(document)

            # ID-based update (specific fields)
            >>> document = client.update_document(document_id, name="New Name")

            # Hybrid: fetch object, override one field
            >>> document = client.get_document(document_id)
            >>> document = client.update_document(document, metadata={"new": "data"})
        """
        if isinstance(document, Document):
            # Object-based: use fields from document, allow kwargs to override
            document_id = document.id
            data = DocumentUpdate(
                name=name if name is not None else document.name,
                metadata=metadata if metadata is not None else document.metadata,
            )
        else:
            # ID-based: must provide at least one field
            document_id = UUID(str(document))

            # Build update data from provided fields only
            if all(v is None for v in [name, metadata]):
                raise ValueError(
                    "When updating by ID, must provide at least one field: "
                    "'name' or 'metadata'"
                )

            data = DocumentUpdate(
                name=name,
                metadata=metadata,
            )

        response = self._put(
            f"/documents/{document_id}",
            json=data.model_dump(exclude_none=True),
        )
        return Document(**response)

    def delete_document(self, document_id: Union[UUID, str]) -> None:
        """
        Delete a document and all its chunks.

        Args:
            document_id: UUID of the document

        Raises:
            NotFoundError: If document doesn't exist
            VectorDBError: For other errors

        Example:
            >>> client.delete_document(document_id=doc.id)
        """
        self._delete(f"/documents/{document_id}")

    # ========================================================================
    # Chunk Operations
    # ========================================================================

    def add_chunk(
        self,
        *,
        chunk: Optional[Chunk] = None,
        document_id: Optional[Union[UUID, str]] = None,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Add a new chunk to a document.

        Supports two calling styles:
        1. Object style: add_chunk(chunk=Chunk(...)) - document_id inferred from chunk
        2. Primitive style: add_chunk(document_id=..., text="...", embedding=[...])

        Note: All parameters must be passed as keyword arguments.

        Args:
            chunk: Chunk object (use this OR document_id+text+embedding, not both)
            document_id: UUID of the parent document (required if using primitive style)
            text: Text content of the chunk (required if not using chunk object)
            embedding: Vector embedding (required if not using chunk object)
            metadata: Optional metadata dictionary

        Returns:
            Created Chunk instance

        Raises:
            ValidationError: If neither chunk nor text+embedding provided
            NotFoundError: If document doesn't exist
            ValueError: If parameters are invalid or document_id cannot be determined
            VectorDBError: For other errors

        Examples:
            # Object style (document_id inferred from chunk)
            >>> chunk_obj = Chunk(
            ...     document_id=document.id,
            ...     text="Hello world",
            ...     embedding=[0.1, 0.2, 0.3],
            ...     metadata={"page": 1}
            ... )
            >>> created = client.add_chunk(chunk=chunk_obj)

            # Primitive style (document_id must be provided)
            >>> created = client.add_chunk(
            ...     document_id=document.id,
            ...     text="Hello world",
            ...     embedding=[0.1, 0.2, 0.3],
            ...     metadata={"page": 1}
            ... )
        """
        # Determine document_id and create ChunkCreate object
        resolved_document_id = document_id

        if chunk is not None:
            # Object style - extract document_id from chunk
            resolved_document_id = chunk.document_id
            data = ChunkCreate(
                document_id=UUID(str(chunk.document_id)),
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
            )
        elif text is not None and embedding is not None:
            # Primitive style - document_id must be provided
            if resolved_document_id is None:
                raise ValueError(
                    "document_id must be provided when using primitive style (text + embedding)"
                )
            data = ChunkCreate(
                document_id=UUID(str(resolved_document_id)),
                text=text,
                embedding=embedding,
                metadata=metadata or {},
            )
        else:
            raise ValueError(
                "Must provide either 'chunk' object OR both 'text' and 'embedding'"
            )

        # Ensure we have a document_id for the API call
        if resolved_document_id is None:
            raise ValueError("Could not determine document_id from chunk or parameters")

        response = self._post(
            f"/documents/{resolved_document_id}/chunks",
            json=data.model_dump(mode="json"),
        )
        return Chunk(**response)

    def create_chunk(
        self,
        document_id: Union[UUID, str],
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Create a new chunk in a document.

        DEPRECATED: Use add_chunk() instead. This method will be removed in a future version.

        Args:
            document_id: UUID of the parent document
            text: Text content of the chunk
            embedding: Vector embedding of the text
            metadata: Optional metadata dictionary

        Returns:
            Created Chunk instance
        """
        warnings.warn(
            "create_chunk() is deprecated and will be removed in a future version. "
            "Use add_chunk() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_chunk(
            document_id=document_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )

    def get_chunk(self, chunk_id: Union[UUID, str]) -> Chunk:
        """
        Retrieve a chunk by ID.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            Chunk instance

        Raises:
            NotFoundError: If chunk doesn't exist
            VectorDBError: For other errors

        Example:
            >>> chunk = client.get_chunk(chunk_id="chunk-uuid")
        """
        response = self._get(f"/chunks/{chunk_id}")
        return Chunk(**response)

    def list_chunks(self, document_id: Union[UUID, str]) -> List[Chunk]:
        """
        List all chunks in a document.

        Args:
            document_id: UUID of the document

        Returns:
            List of Chunk instances

        Raises:
            NotFoundError: If document doesn't exist
            VectorDBError: For other errors

        Example:
            >>> chunks = client.list_chunks(document_id=document.id)
            >>> for chunk in chunks:
            ...     print(f"{chunk.text[:50]}...")
        """
        response = self._get(f"/documents/{document_id}/chunks")
        return [Chunk(**chunk) for chunk in response]

    def list_all_chunks(self, library_id: Union[UUID, str]) -> List[Chunk]:
        """
        List all chunks across all documents in a library.

        Args:
            library_id: UUID of the library

        Returns:
            List of Chunk instances from all documents

        Raises:
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Example:
            >>> chunks = client.list_all_chunks(library_id=library.id)
            >>> print(f"Total chunks: {len(chunks)}")
            >>> for chunk in chunks:
            ...     print(f"Doc {chunk.document_id}: {chunk.text[:50]}...")
        """
        documents = self.list_documents(library_id=library_id)
        all_chunks = []
        for document in documents:
            chunks = self.list_chunks(document_id=document.id)
            all_chunks.extend(chunks)
        return all_chunks

    def update_chunk(
        self,
        chunk: Union[Chunk, UUID, str],
        *,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Update an existing chunk.

        Pass either a Chunk object (fetch-modify-update) or a chunk ID with fields to update.
        When passing a Chunk object, fields can be overridden via keyword arguments.

        Args:
            chunk: Chunk object OR chunk ID (UUID/string)
            text: Override/set text (optional)
            embedding: Override/set embedding (optional)
            metadata: Override/set metadata (optional)

        Returns:
            Updated Chunk instance

        Raises:
            NotFoundError: If chunk doesn't exist
            ValidationError: If update validation fails
            VectorDBError: For other errors

        Examples:
            # Object-based update (fetch-modify-update)
            >>> chunk = client.get_chunk(chunk_id)
            >>> chunk.text = "Updated text"
            >>> chunk = client.update_chunk(chunk)

            # ID-based update (specific fields)
            >>> chunk = client.update_chunk(chunk_id, text="New text")

            # Hybrid: fetch object, override one field
            >>> chunk = client.get_chunk(chunk_id)
            >>> chunk = client.update_chunk(chunk, metadata={"new": "data"})
        """
        if isinstance(chunk, Chunk):
            # Object-based: use fields from chunk, allow kwargs to override
            chunk_id = str(chunk.id)
            data = ChunkUpdate(
                text=text if text is not None else chunk.text,
                embedding=embedding if embedding is not None else chunk.embedding,
                metadata=metadata if metadata is not None else chunk.metadata,
            )
        else:
            # ID-based: must provide at least one field
            chunk_id = str(chunk)

            # Build update data from provided fields only
            if all(v is None for v in [text, embedding, metadata]):
                raise ValueError(
                    "When updating by ID, must provide at least one field: "
                    "'text', 'embedding', or 'metadata'"
                )

            data = ChunkUpdate(
                text=text,
                embedding=embedding,
                metadata=metadata,
            )

        response = self._put(
            f"/chunks/{chunk_id}",
            json=data.model_dump(exclude_none=True),
        )
        return Chunk(**response)

    def delete_chunk(self, chunk_id: Union[UUID, str]) -> None:
        """
        Delete a chunk.

        Args:
            chunk_id: UUID of the chunk

        Raises:
            NotFoundError: If chunk doesn't exist
            VectorDBError: For other errors

        Example:
            >>> client.delete_chunk(chunk_id=chunk.id)
        """
        self._delete(f"/chunks/{chunk_id}")

    def add_chunks(
        self,
        *,
        chunks: List[Union[Chunk, Dict[str, Any]]],
        document_id: Optional[Union[UUID, str]] = None,
    ) -> List[Chunk]:
        """
        Add multiple chunks to a document in a single request.

        This is more efficient than adding chunks one by one as it only
        invalidates the vector index once.

        Note: All parameters must be passed as keyword arguments.

        Args:
            chunks: List of Chunk objects or dicts with {text, embedding, metadata}
            document_id: UUID of the parent document (optional if chunks are Chunk objects with document_id)

        Returns:
            List of created Chunk instances

        Raises:
            ValidationError: If any chunk validation fails
            NotFoundError: If document doesn't exist
            ValueError: If document_id cannot be determined
            VectorDBError: For other errors

        Examples:
            # Using Chunk objects (document_id inferred from chunks)
            >>> chunks = [
            ...     Chunk(
            ...         document_id=document.id,
            ...         text="First chunk",
            ...         embedding=[0.1, 0.2, 0.3],
            ...         metadata={"page": 1}
            ...     ),
            ...     Chunk(
            ...         document_id=document.id,
            ...         text="Second chunk",
            ...         embedding=[0.4, 0.5, 0.6],
            ...         metadata={"page": 2}
            ...     )
            ... ]
            >>> created = client.add_chunks(chunks=chunks)

            # Using dicts (document_id must be provided)
            >>> chunks = [
            ...     {"text": "First", "embedding": [0.1, 0.2], "metadata": {}},
            ...     {"text": "Second", "embedding": [0.3, 0.4], "metadata": {}}
            ... ]
            >>> created = client.add_chunks(chunks=chunks, document_id=document.id)
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        # Validate that document_id is provided for dict chunks
        if document_id is None and any(isinstance(c, dict) for c in chunks):
            raise ValueError("document_id must be provided when using dict chunks")

        # Convert chunks to ChunkCreate objects and extract document_id from first chunk
        chunk_creates = []
        resolved_document_id = document_id

        for chunk in chunks:
            if isinstance(chunk, Chunk):
                # Chunk object - use its document_id
                chunk_doc_id = chunk.document_id

                # If document_id not yet resolved, use this chunk's document_id
                if resolved_document_id is None:
                    resolved_document_id = chunk_doc_id

                chunk_creates.append(
                    ChunkCreate(
                        document_id=UUID(str(chunk_doc_id)),
                        text=chunk.text,
                        embedding=chunk.embedding,
                        metadata=chunk.metadata,
                    )
                )
            elif isinstance(chunk, dict):
                # Dict - validate required fields
                if "text" not in chunk or "embedding" not in chunk:
                    raise ValueError(
                        "Each chunk dict must have 'text' and 'embedding' fields"
                    )
                chunk_creates.append(
                    ChunkCreate(
                        document_id=UUID(str(resolved_document_id)),
                        text=chunk["text"],
                        embedding=chunk["embedding"],
                        metadata=chunk.get("metadata", {}),
                    )
                )
            else:
                raise ValueError(
                    f"Chunks must be Chunk objects or dicts, got {type(chunk)}"
                )

        # Ensure we have a document_id for the API call
        if resolved_document_id is None:
            raise ValueError(
                "Could not determine document_id from chunks or parameters"
            )

        # Call batch API endpoint
        response = self._post(
            f"/documents/{resolved_document_id}/chunks/batch",
            json={"chunks": [c.model_dump(mode="json") for c in chunk_creates]},
        )

        # Convert response to Chunk objects
        return [Chunk(**chunk) for chunk in response["chunks"]]

    # ========================================================================
    # Search Operations
    # ========================================================================

    def search(
        self,
        library_id: Union[UUID, str],
        embedding: List[float],
        k: int = 10,
        filters: Optional[Union[SearchFilters, Dict[str, Any]]] = None,
        filter_function: Optional[Callable[[SearchResult], bool]] = None,
        combined_filters: Optional[SearchFiltersWithCallable] = None,
    ) -> SearchResponse:
        """
        Perform k-nearest neighbor vector search in a library.

        Args:
            library_id: UUID of the library to search in
            embedding: Query vector embedding
            k: Number of nearest neighbors to return (1-1000)
            filters: Declarative search filters applied server-side. Can be:
                    - SearchFilters object (structured filters for metadata, time, document IDs)
                    - Dict (converted to SearchFilters with validation)
            filter_function: Custom filter function applied client-side.
                    - Callable[[SearchResult], bool] (function that receives SearchResult objects)
            combined_filters: Combined declarative and custom filters.
                    - SearchFiltersWithCallable (includes both metadata filters and custom_filter function)

        Returns:
            SearchResponse with matching chunks and query time

        Raises:
            ValidationError: If request validation fails or multiple filter parameters provided
            NotFoundError: If library doesn't exist
            VectorDBError: For other errors

        Note:
            - Only ONE of filters, filter_function, or combined_filters can be specified
            - Declarative filters (filters param) are applied SERVER-SIDE for optimal performance
            - Custom filter functions (filter_function param) are applied CLIENT-SIDE after fetching
            - Combined filters (combined_filters param) apply declarative server-side then custom client-side
            - When using client-side filtering, the SDK over-fetches (k*3) results and filters locally
            - Filter functions receive SearchResult objects with these fields:
              * chunk_id: UUID
              * document_id: UUID
              * text: str
              * score: float
              * metadata: Dict[str, Any]

        Examples:
            # Declarative filters only (server-side)
            >>> results = client.search(
            ...     library_id=library.id,
            ...     embedding=[0.1, 0.2, 0.3, ...],
            ...     k=5,
            ...     filters={"metadata": {"operator": "and", "filters": [{"field": "category", "operator": "eq", "value": "tech"}]}}
            ... )

            # Custom filter function only (client-side)
            >>> results = client.search(
            ...     library_id=library.id,
            ...     embedding=vec,
            ...     k=10,
            ...     filter_function=lambda result: result.score > 0.8 and "important" in result.text
            ... )

            # Combined: declarative + custom
            >>> from my_vector_db import SearchFiltersWithCallable, FilterGroup, MetadataFilter
            >>> results = client.search(
            ...     library_id=library.id,
            ...     embedding=vec,
            ...     k=10,
            ...     combined_filters=SearchFiltersWithCallable(
            ...         metadata=FilterGroup(...),
            ...         custom_filter=lambda result: "machine learning" in result.text.lower()
            ...     )
            ... )
        """
        provided_filters = sum(
            [
                filters is not None,
                filter_function is not None,
                combined_filters is not None,
            ]
        )

        if provided_filters > 1:
            raise ValueError(
                "Only one of 'filters', 'filter_function', or 'combined_filters' can be specified"
            )

        declarative_filters = None
        custom_filter_func = None

        if filters is not None:
            if isinstance(filters, dict):
                declarative_filters = SearchFilters(**filters)
            elif isinstance(filters, SearchFilters):
                declarative_filters = filters
            else:
                raise ValueError(
                    f"filters must be SearchFilters or Dict, got {type(filters)}"
                )

        elif filter_function is not None:
            if not callable(filter_function):
                raise ValueError(
                    f"filter_function must be Callable, got {type(filter_function)}"
                )
            custom_filter_func = filter_function

        elif combined_filters is not None:
            if not isinstance(combined_filters, SearchFiltersWithCallable):
                raise ValueError(
                    f"combined_filters must be SearchFiltersWithCallable, got {type(combined_filters)}"
                )

            declarative_filters = SearchFilters(
                metadata=combined_filters.metadata,
                created_after=combined_filters.created_after,
                created_before=combined_filters.created_before,
                document_ids=combined_filters.document_ids,
            )
            custom_filter_func = combined_filters.custom_filter

        fetch_k = k * 3 if custom_filter_func else k

        data = SearchQuery(
            embedding=embedding,
            k=fetch_k,
            filters=declarative_filters,
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
        filter_func: Callable[[SearchResult], bool],
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
            filter_func: Custom filter function that accepts SearchResult -> bool
                        SearchResult has: chunk_id, text, metadata, score, document_id
            k: Original requested number of results

        Returns:
            New SearchResponse with filtered results (up to k items)
        """
        filtered_results = []

        for result in response.results:
            # Apply custom filter directly on SearchResult
            try:
                if filter_func(result):
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
    # Admin / Persistence Methods
    # ========================================================================

    def save_snapshot(self) -> Dict[str, Any]:
        """
        Manually trigger a database snapshot save.

        This saves the current database state to disk immediately,
        regardless of the automatic save threshold configured on the server.

        Returns:
            Dictionary with save status and statistics

        Raises:
            ServiceUnavailableError: If persistence is not enabled on the server
            VectorDBError: For other server errors

        Example:
            >>> result = client.save_snapshot()
            >>> print(f"Saved snapshot with {result['stats']['chunks']} chunks")
        """
        response = self._post("/admin/snapshot/save")
        return response

    def restore_snapshot(self) -> Dict[str, Any]:
        """
        Restore database state from the latest snapshot.

        WARNING: This will replace ALL current data with the snapshot data.
        Any data created after the snapshot was taken will be lost.

        Returns:
            Dictionary with restore status and restored counts

        Raises:
            NotFoundError: If no snapshot file exists
            ServiceUnavailableError: If persistence is not enabled
            VectorDBError: For other errors

        Example:
            >>> result = client.restore_snapshot()
            >>> print(f"Restored {result['stats']['libraries']} libraries")
        """
        response = self._post("/admin/snapshot/restore")
        return response

    def get_persistence_status(self) -> Dict[str, Any]:
        """
        Get current persistence status and statistics.

        Returns information about:
        - Whether persistence is enabled
        - Snapshot file existence and metadata
        - Number of operations since last save
        - Current database statistics

        Returns:
            Dictionary with persistence status details

        Example:
            >>> status = client.get_persistence_status()
            >>> if status['enabled']:
            ...     print(f"Operations since save: {status['operations_since_save']}")
            ...     print(f"Save threshold: {status['save_threshold']}")
        """
        response = self._get("/admin/persistence/status")
        return response

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
