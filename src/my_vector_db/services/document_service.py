"""
Document Service - Business logic for document and chunk operations.

This service handles CRUD operations for documents and chunks.
Index invalidation is automatic when chunks are modified.
"""

from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import UUID

from my_vector_db.domain.models import Chunk, Document
from my_vector_db.storage import VectorStorage

if TYPE_CHECKING:
    from my_vector_db.services.library_service import LibraryService


class DocumentService:
    """
    Service for managing documents and chunks.

    This service provides business logic for document/chunk CRUD operations.
    Automatically invalidates library indexes when chunks are modified.
    """

    def __init__(
        self, storage: VectorStorage, library_service: Optional["LibraryService"] = None
    ) -> None:
        """
        Initialize the document service.

        Args:
            storage: The storage instance to use
            library_service: Optional library service for index invalidation
        """
        self._storage = storage
        self._library_service = library_service

    # ========================================================================
    # Document Operations
    # ========================================================================

    def create_document(
        self, library_id: UUID, name: str, metadata: Optional[Dict] = None
    ) -> Document:
        """
        Create a new document in a library.

        Args:
            library_id: Parent library ID
            name: Document name
            metadata: Optional metadata

        Returns:
            The created document

        Raises:
            KeyError: If library doesn't exist (raised by storage layer)
        """
        # Create document model
        document = Document(name=name, library_id=library_id, metadata=metadata or {})

        # Store in storage (validates library exists)
        self._storage.create_document(document)
        return document

    def get_document(self, document_id: UUID) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: The document's unique identifier

        Returns:
            The document if found, None otherwise
        """
        return self._storage.get_document(document_id)

    def update_document(
        self,
        document_id: UUID,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Document:
        """
        Update a document.

        Args:
            document_id: The document's unique identifier
            name: New name (optional)
            metadata: New metadata (optional)

        Returns:
            The updated document

        Raises:
            KeyError: If document doesn't exist
        """
        # Get existing document
        document = self._storage.get_document(document_id)
        if not document:
            raise KeyError(f"Document with ID {document_id} not found")

        # Update fields that are not None
        if name is not None:
            document.name = name
        if metadata is not None:
            document.metadata = metadata

        # Save to storage
        self._storage.update_document(document_id, document)
        return document

    def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document and all its chunks.

        The library's index is automatically invalidated and will be rebuilt
        on the next query.

        Args:
            document_id: The document's unique identifier

        Returns:
            True if deleted, False if not found
        """
        # Get document to find library_id before deleting
        document = self._storage.get_document(document_id)
        if document and self._library_service:
            library_id = document.library_id
            # Delete from storage (cascades to chunks)
            result = self._storage.delete_document(document_id)
            # Invalidate index after deletion
            self._library_service.invalidate_index(library_id)
            return result

        # Delete from storage (no invalidation if library_service not set)
        return self._storage.delete_document(document_id)

    def list_documents(self, library_id: UUID) -> List[Document]:
        """
        Get all documents in a library.

        Args:
            library_id: The library's unique identifier

        Returns:
            List of documents
        """
        return self._storage.list_documents_by_library(library_id)

    # ========================================================================
    # Chunk Operations
    # ========================================================================

    def create_chunk(
        self,
        document_id: UUID,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> Chunk:
        """
        Create a new chunk in a document.

        The library's index is automatically invalidated and will be rebuilt
        on the next query.

        Args:
            document_id: Parent document ID
            text: Chunk text
            embedding: Vector embedding
            metadata: Optional metadata

        Returns:
            The created chunk

        Raises:
            KeyError: If document doesn't exist
        """
        # Verify document exists and get library_id
        document = self._storage.get_document(document_id)
        if not document:
            raise KeyError(f"Document with ID {document_id} not found")

        # Create chunk model
        chunk = Chunk(
            document_id=document_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )

        # Store in storage
        self._storage.create_chunk(chunk)

        # Invalidate index so it rebuilds on next query
        if self._library_service:
            self._library_service.invalidate_index(document.library_id)

        return chunk

    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        """
        Get a chunk by ID.

        Args:
            chunk_id: The chunk's unique identifier

        Returns:
            The chunk if found, None otherwise
        """
        return self._storage.get_chunk(chunk_id)

    def update_chunk(
        self,
        chunk_id: UUID,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> Chunk:
        """
        Update a chunk.

        If embedding is updated, the library's index is automatically invalidated
        and will be rebuilt on the next query.

        Args:
            chunk_id: The chunk's unique identifier
            text: New text (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)

        Returns:
            The updated chunk

        Raises:
            KeyError: If chunk doesn't exist
        """
        # Get existing chunk
        chunk = self._storage.get_chunk(chunk_id)
        if not chunk:
            raise KeyError(f"Chunk with ID {chunk_id} not found")

        # Track if embedding changed
        embedding_changed = False

        # Update fields that are not None
        if text is not None:
            chunk.text = text
        if embedding is not None:
            chunk.embedding = embedding
            embedding_changed = True
        if metadata is not None:
            chunk.metadata = metadata

        # Save to storage
        self._storage.update_chunk(chunk_id, chunk)

        # Invalidate index if embedding changed
        if embedding_changed and self._library_service:
            document = self._storage.get_document(chunk.document_id)
            if document:
                self._library_service.invalidate_index(document.library_id)

        return chunk

    def delete_chunk(self, chunk_id: UUID) -> bool:
        """
        Delete a chunk.

        The library's index is automatically invalidated and will be rebuilt
        on the next query.

        Args:
            chunk_id: The chunk's unique identifier

        Returns:
            True if deleted, False if not found
        """
        # Get chunk and document to find library_id before deleting
        chunk = self._storage.get_chunk(chunk_id)
        if chunk and self._library_service:
            document = self._storage.get_document(chunk.document_id)
            if document:
                library_id = document.library_id
                # Delete from storage
                result = self._storage.delete_chunk(chunk_id)
                # Invalidate index after deletion
                self._library_service.invalidate_index(library_id)
                return result

        # Delete from storage (no invalidation if library_service not set)
        return self._storage.delete_chunk(chunk_id)

    def list_chunks(self, document_id: UUID) -> List[Chunk]:
        """
        Get all chunks in a document.

        Args:
            document_id: The document's unique identifier

        Returns:
            List of chunks
        """
        return self._storage.list_chunks_by_document(document_id)

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def create_documents_batch(
        self, library_id: UUID, documents: List[Document]
    ) -> List[Document]:
        """
        Create multiple documents in a single operation.

        This is more efficient than creating documents one by one.

        Args:
            library_id: Parent library ID
            documents: List of documents to create (IDs will be generated if not set)

        Returns:
            List of created documents

        Raises:
            KeyError: If library doesn't exist
        """
        # Ensure all documents have the correct library_id
        for document in documents:
            document.library_id = library_id

        # Store in batch (validates library exists)
        self._storage.create_documents_batch(documents)
        return documents

    def create_chunks_batch(
        self, document_id: UUID, chunks: List[Chunk]
    ) -> List[Chunk]:
        """
        Create multiple chunks in a single operation.

        The library's index is automatically invalidated and will be rebuilt
        on the next query.

        This is more efficient than creating chunks one by one, especially
        when adding many chunks to a document.

        Args:
            document_id: Parent document ID
            chunks: List of chunks to create

        Returns:
            List of created chunks

        Raises:
            KeyError: If document doesn't exist
        """
        # Verify document exists and get library_id
        document = self._storage.get_document(document_id)
        if not document:
            raise KeyError(f"Document with ID {document_id} not found")

        # Ensure all chunks have the correct document_id
        for chunk in chunks:
            chunk.document_id = document_id

        # Store in batch
        self._storage.create_chunks_batch(chunks)

        # Invalidate index once after all chunks are added
        if self._library_service:
            self._library_service.invalidate_index(document.library_id)

        return chunks
