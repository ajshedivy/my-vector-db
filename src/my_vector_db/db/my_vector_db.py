"""
MyVectorDB implementation for Agno framework.

This module provides an Agno-compatible wrapper around the custom vector database REST API,
allowing Agno agents to use the vector database for knowledge storage and retrieval.
"""

from hashlib import md5
from typing import Any, Dict, List, Optional

from my_vector_db.sdk.models import SearchResponse
from my_vector_db.sdk import VectorDBClient, VectorDBError, NotFoundError

from agno.knowledge.document import Document
from agno.knowledge.embedder import Embedder
from agno.utils.log import log_debug, log_info, logger
from agno.vectordb.base import VectorDb
from agno.vectordb.search import SearchType


class MyVectorDB(VectorDb):
    """
    MyVectorDB class for managing vector operations with custom REST API.

    This implementation wraps a FastAPI-based vector database, providing
    an Agno-compatible interface for document storage and semantic search.

    Args:
        api_base_url: Base URL of the vector database API (default: http://localhost:8000)
        library_name: Name of the library to create/use
        index_type: Type of index to use ('flat' or 'hnsw')
        embedder: The embedder to use when embedding documents
        name: Optional name for the vector database instance
        description: Optional description
        id: Optional custom ID
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        library_name: Optional[str] = None,
        index_type: str = "flat",
        embedder: Optional[Embedder] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        id: Optional[str] = None,
    ):
        # Generate ID if not provided
        if id is None:
            from agno.utils.string import generate_id

            library_identifier = library_name or "default_library"
            seed = f"{api_base_url}#{library_identifier}"
            id = generate_id(seed)

        # Initialize base class
        super().__init__(id=id, name=name, description=description)

        # API configuration
        self.api_base_url = api_base_url.rstrip("/")
        self.client = VectorDBClient(base_url=self.api_base_url)

        # Library configuration
        self.library_name = library_name or "agno_knowledge_base"
        self.index_type = index_type
        self.library_id: Optional[str] = None
        self.document_id: Optional[str] = None  # We'll use one document per library

        # Embedder for embedding document contents
        if embedder is None:
            from agno.knowledge.embedder.cohere import CohereEmbedder

            embedder = CohereEmbedder(
                id="embed-english-light-v3.0",  # 384 dimensions, matches test data
                input_type="search_document",
            )
            log_info("Embedder not provided, using CohereEmbedder as default.")

        self.embedder: Embedder = embedder
        self.dimensions: Optional[int] = self.embedder.dimensions

        if self.dimensions is None:
            raise ValueError("Embedder.dimensions must be set.")

        log_debug(f"Initialized MyVectorDB with library: '{self.library_name}'")

    def create(self) -> None:
        """Create the library if it does not exist."""
        if not self.exists():
            log_info(f"Creating library: {self.library_name}")

            try:
                library = self.client.create_library(
                    name=self.library_name,
                    index_type=self.index_type,
                    index_config={"metric": "cosine"},
                    metadata={"description": self.description or "Agno Knowledge Base"},
                )
                self.library_id = library.id

                # Create a default document to hold chunks
                document = self.client.create_document(
                    library_id=library.id,
                    name=f"{self.library_name}_documents",
                    metadata={"type": "agno_knowledge"},
                )
                self.document_id = document.id

                log_info(
                    f"Created library: {self.library_id}, document: {self.document_id}"
                )
            except VectorDBError as e:
                logger.error(f"Error creating library: {e}")
                raise

    async def async_create(self) -> None:
        """Create the library asynchronously."""
        # For simplicity, use sync create (could be enhanced with httpx.AsyncClient)
        self.create()

    def _ensure_library_exists(self) -> None:
        """Ensure library and document are initialized."""
        if not self.library_id or not self.document_id:
            if self.exists():
                libraries = self.client.list_libraries()
                for lib in libraries:
                    if lib.name == self.library_name:
                        self.library_id = lib.id
                        if lib.document_ids:
                            self.document_id = lib.document_ids[0]
                        break
            else:
                # Create new library
                self.create()

    def doc_exists(self, document: Document) -> bool:
        """Check if a document exists in the database."""
        try:
            self._ensure_library_exists()

            # Generate doc ID from content
            cleaned_content = document.content.replace("\x00", "\ufffd")
            doc_id = md5(cleaned_content.encode()).hexdigest()

            # Check if chunk with this ID exists
            return self.id_exists(doc_id)
        except Exception as e:
            log_debug(f"Error checking doc existence: {e}")
            return False

    async def async_doc_exists(self, document: Document) -> bool:
        """Asynchronously check if document exists."""
        return self.doc_exists(document)

    def insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert documents into the database.

        Args:
            content_hash: Hash of the content being inserted
            documents: List of documents to insert
            filters: Optional filters to add as metadata
        """
        if len(documents) <= 0:
            log_info("No documents to insert")
            return

        self._ensure_library_exists()
        log_debug(f"Inserting {len(documents)} documents")

        inserted_count = 0
        for document in documents:
            # Skip if document already exists
            if self.doc_exists(document):
                log_debug(f"Document already exists: {document.name}")
                continue

            # Add filters to metadata if provided
            if filters:
                meta_data = document.meta_data.copy() if document.meta_data else {}
                meta_data.update(filters)
                document.meta_data = meta_data

            # Embed the document
            document.embed(embedder=self.embedder)

            # Prepare chunk data
            cleaned_content = document.content.replace("\x00", "\ufffd")
            doc_id = md5(cleaned_content.encode()).hexdigest()

            chunk_metadata = {
                "name": document.name,
                "meta_data": document.meta_data,
                "usage": document.usage,
                "content_id": document.content_id,
                "content_hash": content_hash,
                "doc_id": doc_id,
            }

            try:
                # Insert chunk via API
                chunk = self.client.create_chunk(
                    library_id=self.library_id,
                    document_id=self.document_id,
                    text=cleaned_content,
                    embedding=document.embedding,
                    metadata=chunk_metadata,
                )

                inserted_count += 1
                log_debug(
                    f"Inserted document: {document.name} with chunk ID: {chunk.id}"
                )
            except VectorDBError as e:
                logger.error(f"Error inserting document '{document.name}': {e}")

        log_info(f"Successfully inserted {inserted_count} documents")

    async def async_insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Asynchronously insert documents."""
        # For simplicity, use sync insert (could be enhanced with httpx.AsyncClient)
        self.insert(content_hash, documents, filters)

    def upsert_available(self) -> bool:
        """Check if upsert is available."""
        return True

    def upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert documents (delete existing with same content_hash, then insert)."""
        if self.content_hash_exists(content_hash):
            self._delete_by_content_hash(content_hash)
        self.insert(content_hash=content_hash, documents=documents, filters=filters)

    async def async_upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Asynchronously upsert documents."""
        self.upsert(content_hash, documents, filters)

    def _get_query_embedding(self, query: str) -> Optional[list[float]]:
        """
        Generate embedding for a query using the appropriate input type.

        For Cohere embeddings, this uses 'search_query' input type which is
        optimized for retrieval tasks.
        """
        # Check if this is a CohereEmbedder
        from agno.knowledge.embedder.cohere import CohereEmbedder

        if isinstance(self.embedder, CohereEmbedder):
            # Temporarily set input_type to search_query for retrieval
            original_input_type = self.embedder.input_type
            self.embedder.input_type = "search_query"
            embedding = self.embedder.get_embedding(query)
            self.embedder.input_type = original_input_type
            return embedding
        else:
            # For other embedders, use default behavior
            return self.embedder.get_embedding(query)

    def search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for documents matching the query.

        Args:
            query: Query string to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            List of matching documents
        """
        self._ensure_library_exists()

        try:
            # Generate query embedding with appropriate input type
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                logger.error(f"Error getting embedding for query: {query}")
                return []

            # Search via API
            result: SearchResponse = self.client.search(
                library_id=self.library_id,
                embedding=query_embedding,
                k=limit,
                filters=filters,
            )

            # Convert results to Document objects
            documents = []
            for search_result in result.results:
                metadata = search_result.metadata

                # Apply filters if provided (note: filters should already be applied by API)
                if filters:
                    match = True
                    for key, value in filters.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                # Create Document object
                doc = Document(
                    name=metadata.get("name", "unknown"),
                    meta_data=metadata.get("meta_data", {}),
                    content=search_result.text,
                    embedder=self.embedder,
                    embedding=None,  # Embedding not returned in search results
                    usage=metadata.get("usage"),
                    content_id=metadata.get("content_id"),
                )
                documents.append(doc)

            log_info(f"Found {len(documents)} documents")
            return documents

        except VectorDBError as e:
            logger.error(f"Error searching: {e}")
            return []

    async def async_search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Asynchronously search for documents."""
        return self.search(query, limit, filters)

    def drop(self) -> None:
        """Delete the library."""
        if self.exists():
            try:
                log_debug(f"Deleting library: {self.library_name}")
                _ = self.client.delete_library(library_id=self.library_id)
                self.library_id = None
                self.document_id = None
                log_info(f"Deleted library: {self.library_name}")
            except VectorDBError as e:
                logger.error(f"Error deleting library: {e}")

    async def async_drop(self) -> None:
        """Asynchronously delete the library."""
        self.drop()

    def exists(self) -> bool:
        """Check if the library exists."""
        try:
            libraries = self.client.list_libraries()

            for lib in libraries:
                if lib.name == self.library_name:
                    return True
            return False
        except VectorDBError:
            return False

    async def async_exists(self) -> bool:
        """Asynchronously check if library exists."""
        return self.exists()

    def get_count(self) -> int:
        """Get the number of chunks in the library."""
        try:
            self._ensure_library_exists()

            # Get library details to count chunks
            library = self.client.get_library(library_id=self.library_id)

            # Count chunks across all documents
            total_chunks = 0
            for doc_id in library.document_ids:
                document = self.client.get_document(
                    library_id=self.library_id, document_id=doc_id
                )
                total_chunks += len(document.chunk_ids)

            return total_chunks
        except VectorDBError as e:
            logger.error(f"Error getting count: {e}")
            return 0

    async def async_get_count(self) -> int:
        """Asynchronously get the count."""
        return self.get_count()

    def optimize(self) -> None:
        """Optimize the database (no-op for this implementation)."""
        pass

    def delete(self) -> bool:
        """Delete the database (alias for drop)."""
        self.drop()
        return True

    def name_exists(self, name: str) -> bool:
        """Check if a document with the given name exists."""
        try:
            self._ensure_library_exists()

            # Search for chunks with matching name in metadata
            # This requires getting all chunks and checking metadata
            # For simplicity, we'll return False (could be enhanced)
            return False
        except Exception as e:
            logger.error(f"Error checking name existence: {e}")
            return False

    async def async_name_exists(self, name: str) -> bool:
        """Asynchronously check if name exists."""
        return self.name_exists(name)

    def id_exists(self, id: str) -> bool:
        """Check if a chunk with the given ID exists."""
        try:
            self._ensure_library_exists()

            # Try to get the chunk
            _ = self.client.get_chunk(
                library_id=self.library_id, document_id=self.document_id, chunk_id=id
            )
            return True
        except NotFoundError:
            # Expected case - chunk doesn't exist
            return False
        except VectorDBError as e:
            # Unexpected error - log and return False
            logger.error(f"Error checking chunk existence: {e}")
            return False

    def content_hash_exists(self, content_hash: str) -> bool:
        """Check if documents with the given content hash exist."""
        # For simplicity, assume it doesn't exist
        # Could be enhanced by searching metadata
        return False

    def delete_by_id(self, id: str) -> bool:
        """Delete a chunk by ID."""
        try:
            self._ensure_library_exists()

            _ = self.client.delete_chunk(
                library_id=self.library_id, document_id=self.document_id, chunk_id=id
            )
            log_info(f"Deleted chunk with id: {id}")
            return True
        except VectorDBError as e:
            logger.error(f"Error deleting chunk by id '{id}': {e}")
            return False

    def delete_by_name(self, name: str) -> bool:
        """Delete chunks by name (not implemented)."""
        raise NotImplementedError(
            "delete_by_name is not implemented for VectorDB backend"
        )

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete chunks by metadata (not implemented)."""
        raise NotImplementedError(
            "delete_by_metadata is not implemented for VectorDB backend"
        )

    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete chunks by content_id (not implemented)."""
        raise NotImplementedError(
            "delete_by_content_id is not implemented for VectorDB backend"
        )

    def _delete_by_content_hash(self, content_hash: str) -> bool:
        """Delete chunks by content hash (not implemented)."""
        raise NotImplementedError(
            "_delete_by_content_hash is not implemented for VectorDB backend"
        )

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for chunks with given content_id."""
        raise NotImplementedError(
            "update_metadata is not implemented for VectorDB backend"
        )

    def get_supported_search_types(self) -> List[str]:
        """Get supported search types."""
        return [SearchType.vector]

    def __del__(self):
        """Cleanup: close HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass
