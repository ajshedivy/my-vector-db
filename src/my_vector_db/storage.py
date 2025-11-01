"""
In-memory storage layer with thread-safe operations.

This module provides thread-safe CRUD operations for libraries, documents, and chunks.
Uses threading.RLock for synchronization to prevent data races.

Design Choices:
- RLock (Reentrant Lock): Allows the same thread to acquire the lock multiple times.
  This is useful for nested operations (e.g., creating a document also updates the library).
- Read/Write pattern: Multiple readers can access data concurrently, but writes are exclusive.
- In-memory storage: Fast access, suitable for development and moderate-scale deployments.
  For production, this could be extended to support persistence to disk.
"""

from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional
from uuid import UUID

from my_vector_db.domain.models import Chunk, Document, Library
from my_vector_db.serialization import deserialize_from_json, serialize_to_json


class VectorStorage:
    """
    Thread-safe in-memory storage for vector database entities.

    This class manages the storage and retrieval of libraries, documents, and chunks
    with proper synchronization to prevent data races in concurrent environments.
    """

    def __init__(self) -> None:
        """Initialize the storage with empty dictionaries and a reentrant lock."""
        self._libraries: Dict[UUID, Library] = {}
        self._documents: Dict[UUID, Document] = {}
        self._chunks: Dict[UUID, Chunk] = {}
        self._lock = RLock()

        # Persistence settings
        self._persistence_enabled = False
        self._snapshot_path: Optional[Path] = None
        self._operation_counter = 0
        self._save_every = -1

    def enable_persistence(self, data_dir: str, save_every: int = -1) -> None:
        """
        Enable periodic snapshot persistence.

        Args:
            data_dir: Directory to store snapshot files
            save_every: Save snapshot after this many operations (default: -1 = disabled)
        """
        with self._lock:
            self._persistence_enabled = True
            self._save_every = save_every

            # Ensure data directory exists
            data_path = Path(data_dir)
            data_path.mkdir(parents=True, exist_ok=True)

            self._snapshot_path = data_path / "snapshot.json"

    def _maybe_save_snapshot(self) -> None:
        """
        Increment operation counter and save snapshot if threshold reached.
        """
        if not self._persistence_enabled or self._save_every < 0:
            return

        self._operation_counter += 1
        if self._operation_counter >= self._save_every:
            self.save_snapshot()
            self._operation_counter = 0

    def save_snapshot(self) -> None:
        """
        Save current storage state to snapshot file.

        Uses serialization functions with atomic write pattern.
        Thread-safe: acquires lock before reading data.
        """
        if not self._persistence_enabled or not self._snapshot_path:
            return

        with self._lock:
            libraries = list(self._libraries.values())
            documents = list(self._documents.values())
            chunks = list(self._chunks.values())

        # Serialize outside of lock (I/O doesn't need lock)
        serialize_to_json(libraries, documents, chunks, self._snapshot_path)

    def load_snapshot(self) -> bool:
        """
        Load storage state from snapshot file.

        Returns:
            True if snapshot was loaded successfully, False otherwise

        Raises:
            ValueError: If snapshot is corrupted or incompatible
        """
        if not self._persistence_enabled or not self._snapshot_path:
            return False

        if not self._snapshot_path.exists():
            return False

        try:
            # Deserialize from file
            libraries, documents, chunks = deserialize_from_json(self._snapshot_path)

            # Populate storage with lock
            with self._lock:
                self._libraries.clear()
                self._documents.clear()
                self._chunks.clear()

                for library in libraries:
                    self._libraries[library.id] = library

                for document in documents:
                    self._documents[document.id] = document

                for chunk in chunks:
                    self._chunks[chunk.id] = chunk

            return True
        except Exception as e:
            # Re-raise with context
            raise ValueError(f"Failed to load snapshot: {e}") from e

    def snapshot_exists(self) -> bool:
        """Check if a snapshot file exists."""
        return bool(
            self._persistence_enabled
            and self._snapshot_path
            and self._snapshot_path.exists()
        )

    # ========================================================================
    # Library Operations
    # ========================================================================

    def create_library(self, library: Library) -> Library:
        """
        Create a new library.

        Args:
            library: The library to store

        Returns:
            The created library

        Raises:
            ValueError: If library with this ID already exists
        """
        with self._lock:
            if library.id in self._libraries:
                raise ValueError(f"Library with ID {library.id} already exists")

            library.updated_at = library.created_at
            self._libraries[library.id] = library
            self._maybe_save_snapshot()
            return library

    def get_library(self, library_id: UUID) -> Optional[Library]:
        """
        Retrieve a library by ID.

        Args:
            library_id: The library's unique identifier

        Returns:
            The library if found, None otherwise
        """
        with self._lock:
            return self._libraries.get(library_id)

    def update_library(self, library_id: UUID, library: Library) -> Library:
        """
        Update an existing library.

        Args:
            library_id: The library's unique identifier
            library: The updated library data

        Returns:
            The updated library

        Raises:
            KeyError: If library doesn't exist
        """
        with self._lock:
            if library_id not in self._libraries:
                raise KeyError(f"Library with ID {library_id} not found")

            library.updated_at = datetime.now()
            self._libraries[library_id] = library

            # Persistence handled by _maybe_save_snapshot()
            self._maybe_save_snapshot()

            return library

    def delete_library(self, library_id: UUID) -> bool:
        """
        Delete a library and all its documents/chunks.

        Args:
            library_id: The library's unique identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            library = self._libraries.get(library_id)
            if not library:
                return False

            # Cascade delete: remove all documents (which will remove all chunks)
            for document_id in list(library.document_ids):
                self.delete_document(document_id)

                # Persistence handled by _maybe_save_snapshot()
                self._maybe_save_snapshot()

            # Now delete the library itself
            del self._libraries[library_id]
            return True

    def list_libraries(self) -> List[Library]:
        """
        Get all libraries.

        Returns:
            List of all libraries
        """
        with self._lock:
            return list(self._libraries.values())

    # ========================================================================
    # Document Operations
    # ========================================================================

    def create_document(self, document: Document) -> Document:
        """
        Create a new document.

        Args:
            document: The document to store

        Returns:
            The created document

        Raises:
            ValueError: If document with this ID already exists
            KeyError: If parent library doesn't exist
        """
        with self._lock:
            if document.id in self._documents:
                raise ValueError(f"Document with ID {document.id} already exists")

            # Verify parent library exists
            library = self._libraries.get(document.library_id)
            if not library:
                raise KeyError(f"Library with ID {document.library_id} not found")

            # Store document
            document.updated_at = document.created_at
            self._documents[document.id] = document

            # Update parent library's document_ids
            if document.id not in library.document_ids:
                library.document_ids.append(document.id)

                # Persistence handled by _maybe_save_snapshot()
                self._maybe_save_snapshot()

            return document

    def get_document(self, document_id: UUID) -> Optional[Document]:
        """
        Retrieve a document by ID.

        Args:
            document_id: The document's unique identifier

        Returns:
            The document if found, None otherwise
        """
        with self._lock:
            return self._documents.get(document_id)

    def update_document(self, document_id: UUID, document: Document) -> Document:
        """
        Update an existing document.

        Args:
            document_id: The document's unique identifier
            document: The updated document data

        Returns:
            The updated document

        Raises:
            KeyError: If document doesn't exist
        """
        with self._lock:
            if document_id not in self._documents:
                raise KeyError(f"Document with ID {document_id} not found")

            document.updated_at = datetime.now()
            self._documents[document_id] = document

            # Persistence handled by _maybe_save_snapshot()
            self._maybe_save_snapshot()

            return document

    def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document's unique identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            document = self._documents.get(document_id)
            if not document:
                return False

            # Cascade delete: remove all chunks
            for chunk_id in list(document.chunk_ids):
                self.delete_chunk(chunk_id)

            # Remove document from parent library's document_ids
            library = self._libraries.get(document.library_id)
            if library and document_id in library.document_ids:
                library.document_ids.remove(document_id)

            # Delete the document itself
            del self._documents[document_id]

            # Persistence handled by _maybe_save_snapshot()
            self._maybe_save_snapshot()
            return True

    def list_documents_by_library(self, library_id: UUID) -> List[Document]:
        """
        Get all documents in a library.

        Args:
            library_id: The library's unique identifier

        Returns:
            List of documents in the library
        """
        with self._lock:
            library = self._libraries.get(library_id)
            if not library:
                return []

            return [
                self._documents[doc_id]
                for doc_id in library.document_ids
                if doc_id in self._documents
            ]

    # ========================================================================
    # Chunk Operations
    # ========================================================================

    def create_chunk(self, chunk: Chunk) -> Chunk:
        """
        Create a new chunk.

        Args:
            chunk: The chunk to store

        Returns:
            The created chunk

        Raises:
            ValueError: If chunk with this ID already exists
            KeyError: If parent document doesn't exist
        """
        with self._lock:
            if chunk.id in self._chunks:
                raise ValueError(f"Chunk with ID {chunk.id} already exists")

            # Verify parent document exists
            document = self._documents.get(chunk.document_id)
            if not document:
                raise KeyError(f"Document with ID {chunk.document_id} not found")

            # Store chunk
            chunk.updated_at = chunk.created_at
            self._chunks[chunk.id] = chunk

            # Update parent document's chunk_ids
            if chunk.id not in document.chunk_ids:
                document.chunk_ids.append(chunk.id)

            # Persistence handled by _maybe_save_snapshot()
            self._maybe_save_snapshot()

            return chunk

    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        """
        Retrieve a chunk by ID.

        Args:
            chunk_id: The chunk's unique identifier

        Returns:
            The chunk if found, None otherwise
        """
        with self._lock:
            return self._chunks.get(chunk_id)

    def update_chunk(self, chunk_id: UUID, chunk: Chunk) -> Chunk:
        """
        Update an existing chunk.

        Args:
            chunk_id: The chunk's unique identifier
            chunk: The updated chunk data

        Returns:
            The updated chunk

        Raises:
            KeyError: If chunk doesn't exist
        """
        with self._lock:
            if chunk_id not in self._chunks:
                raise KeyError(f"Chunk with ID {chunk_id} not found")

            chunk.updated_at = datetime.now()
            self._chunks[chunk_id] = chunk

            # Persistence handled by _maybe_save_snapshot()
            self._maybe_save_snapshot()

            return chunk

    def delete_chunk(self, chunk_id: UUID) -> bool:
        """
        Delete a chunk.

        Args:
            chunk_id: The chunk's unique identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                return False

            # Remove chunk from parent document's chunk_ids
            document = self._documents.get(chunk.document_id)
            if document and chunk_id in document.chunk_ids:
                document.chunk_ids.remove(chunk_id)

            # Delete the chunk itself
            del self._chunks[chunk_id]
            # Persistence handled by _maybe_save_snapshot()
            self._maybe_save_snapshot()
            return True

    def list_chunks_by_document(self, document_id: UUID) -> List[Chunk]:
        """
        Get all chunks in a document.

        Args:
            document_id: The document's unique identifier

        Returns:
            List of chunks in the document
        """
        with self._lock:
            document = self._documents.get(document_id)
            if not document:
                return []

            return [
                self._chunks[chunk_id]
                for chunk_id in document.chunk_ids
                if chunk_id in self._chunks
            ]

    def get_all_chunks_by_library(self, library_id: UUID) -> List[Chunk]:
        """
        Get all chunks in a library (across all documents).

        This is useful for building the vector index.

        Args:
            library_id: The library's unique identifier

        Returns:
            List of all chunks in the library
        """
        with self._lock:
            library = self._libraries.get(library_id)
            if not library:
                return []

            all_chunks: List[Chunk] = []
            for document_id in library.document_ids:
                document = self._documents.get(document_id)
                if document:
                    for chunk_id in document.chunk_ids:
                        chunk = self._chunks.get(chunk_id)
                        if chunk:
                            all_chunks.append(chunk)

            return all_chunks

    def create_chunks_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Create multiple chunks in a single transaction.

        This is more efficient than creating chunks one by one, especially
        when adding many chunks to a document.

        Args:
            chunks: List of chunks to store

        Returns:
            List of created chunks

        Raises:
            ValueError: If any chunk with the same ID already exists
            KeyError: If parent document doesn't exist for any chunk
        """
        with self._lock:
            # Validate all chunks first (fail-fast before modifying anything)
            for chunk in chunks:
                if chunk.id in self._chunks:
                    raise ValueError(f"Chunk with ID {chunk.id} already exists")

                document = self._documents.get(chunk.document_id)
                if not document:
                    raise KeyError(f"Document with ID {chunk.document_id} not found")

            # All validation passed, now create all chunks atomically
            created_chunks: List[Chunk] = []
            for chunk in chunks:
                # Store chunk
                chunk.updated_at = chunk.created_at
                self._chunks[chunk.id] = chunk

                # Update parent document's chunk_ids
                document = self._documents[chunk.document_id]
                if chunk.id not in document.chunk_ids:
                    document.chunk_ids.append(chunk.id)

                created_chunks.append(chunk)

                # Persistence handled by _maybe_save_snapshot()
                self._maybe_save_snapshot()

            return created_chunks

    def create_documents_batch(self, documents: List[Document]) -> List[Document]:
        """
        Create multiple documents in a single transaction.

        Args:
            documents: List of documents to store

        Returns:
            List of created documents

        Raises:
            ValueError: If any document with the same ID already exists
            KeyError: If parent library doesn't exist for any document
        """
        with self._lock:
            # Validate all documents first (fail-fast before modifying anything)
            for document in documents:
                if document.id in self._documents:
                    raise ValueError(f"Document with ID {document.id} already exists")

                library = self._libraries.get(document.library_id)
                if not library:
                    raise KeyError(f"Library with ID {document.library_id} not found")

            # All validation passed, now create all documents atomically
            created_documents: List[Document] = []
            for document in documents:
                # Store document
                document.updated_at = document.created_at
                self._documents[document.id] = document

                # Update parent library's document_ids
                library = self._libraries[document.library_id]
                if document.id not in library.document_ids:
                    library.document_ids.append(document.id)

                created_documents.append(document)

                # Persistence handled by _maybe_save_snapshot()
                self._maybe_save_snapshot()

            return created_documents


# Singleton instance for the application
storage = VectorStorage()
