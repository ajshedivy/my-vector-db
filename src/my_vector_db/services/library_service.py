"""
Library Service - Business logic for library operations.

This service handles CRUD operations for libraries and manages the vector indexes
associated with each library. It acts as an intermediary between the API layer
and the storage/index layers.
"""

from typing import Dict, List, Optional, Set
from uuid import UUID

from my_vector_db.domain.models import BuildIndexResult, IndexType, Library
from my_vector_db.indexes.base import VectorIndex
from my_vector_db.indexes.flat import FlatIndex
from my_vector_db.indexes.hnsw import HNSWIndex
from my_vector_db.storage import VectorStorage


class LibraryService:
    """
    Service for managing libraries and their vector indexes.

    This service provides business logic for library operations and manages
    the creation and lifecycle of vector indexes.
    """

    def __init__(self, storage: VectorStorage) -> None:
        """
        Initialize the library service.

        Args:
            storage: The storage instance to use
        """
        self._storage = storage
        self._indexes: Dict[UUID, VectorIndex] = {}
        self._dirty_indexes: Set[UUID] = set()

    def create_library(
        self,
        name: str,
        metadata: Optional[Dict] = None,
        index_type: IndexType = IndexType.FLAT,
        index_config: Optional[Dict] = None,
    ) -> Library:
        """
        Create a new library.

        Args:
            name: Library name
            metadata: Optional metadata
            index_type: Type of index
            index_config: Configuration for the index

        Returns:
            The created library
        """
        # Create library model (Pydantic validates index_type automatically)
        library = Library(
            name=name,
            metadata=metadata or {},
            index_type=index_type,
            index_config=index_config or {},
        )

        # Store in storage
        self._storage.create_library(library)

        return library

    def get_library(self, library_id: UUID) -> Optional[Library]:
        """
        Get a library by ID.

        Args:
            library_id: The library's unique identifier

        Returns:
            The library if found, None otherwise
        """
        return self._storage.get_library(library_id)

    def update_library(
        self,
        library_id: UUID,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        index_type: Optional[IndexType] = None,
        index_config: Optional[Dict] = None,
    ) -> Library:
        """
        Update a library.

        Args:
            library_id: The library's unique identifier
            name: New name (optional)
            metadata: New metadata (optional)
            index_type: New index type (optional, requires rebuild)
            index_config: New index config (optional, requires rebuild)

        Returns:
            The updated library

        Raises:
            KeyError: If library doesn't exist
        """
        # Get existing library
        library = self._storage.get_library(library_id)
        if not library:
            raise KeyError(f"Library with ID {library_id} not found")

        # Track if index needs rebuilding
        index_changed = False

        # Update fields that are provided
        if name is not None:
            library.name = name
        if metadata is not None:
            library.metadata = metadata
        if index_type is not None and index_type != library.index_type:
            library.index_type = index_type
            index_changed = True
        if index_config is not None:
            library.index_config = index_config
            index_changed = True

        # Save to storage
        self._storage.update_library(library_id, library)

        # If index configuration changed, clear cached index and dirty flag
        if index_changed:
            if library_id in self._indexes:
                del self._indexes[library_id]
            self._dirty_indexes.discard(library_id)

        return library

    def delete_library(self, library_id: UUID) -> bool:
        """
        Delete a library and all its data.

        Args:
            library_id: The library's unique identifier

        Returns:
            True if deleted, False if not found
        """
        # Remove from active indexes and dirty tracking
        if library_id in self._indexes:
            del self._indexes[library_id]
        self._dirty_indexes.discard(library_id)

        # Remove from storage (cascades to documents/chunks)
        return self._storage.delete_library(library_id)

    def list_libraries(self) -> List[Library]:
        """
        Get all libraries.

        Returns:
            List of all libraries
        """
        return self._storage.list_libraries()

    def build_index(self, library_id: UUID) -> BuildIndexResult:
        """
        Build or rebuild the vector index for a library.

        This loads all chunks from the library and adds them to the index.
        Vector dimension is auto-detected from the first chunk.

        Args:
            library_id: The library's unique identifier

        Returns:
            BuildIndexResult with build information

        Raises:
            KeyError: If library doesn't exist
            ValueError: If no chunks or inconsistent dimensions
        """
        # Get library from storage
        library = self._storage.get_library(library_id)
        if not library:
            raise KeyError(f"Library with ID {library_id} not found")

        # Get all chunks in the library
        chunks = self._storage.get_all_chunks_by_library(library_id)
        if not chunks:
            raise ValueError(f"Library {library_id} has no chunks to index")

        # Auto-detect dimension from first chunk
        dimension = len(chunks[0].embedding)

        # Validate all chunks have consistent dimensions
        for chunk in chunks:
            if len(chunk.embedding) != dimension:
                raise ValueError(
                    f"Inconsistent embedding dimensions: expected {dimension}, "
                    f"got {len(chunk.embedding)} for chunk {chunk.id}"
                )

        # Create appropriate index based on library's index type
        index = self._create_index(
            index_type=library.index_type,
            dimension=dimension,
            config=library.index_config,
        )

        # Add all chunk embeddings to the index
        for chunk in chunks:
            index.add(chunk.id, chunk.embedding)

        # Store index in memory
        self._indexes[library_id] = index

        # Mark as no longer dirty
        self._dirty_indexes.discard(library_id)

        # Return build information
        return BuildIndexResult(
            library_id=library_id,
            total_vectors=len(chunks),
            dimension=dimension,
            index_type=library.index_type,
            index_config=library.index_config,
        )

    def get_index(self, library_id: UUID) -> VectorIndex:
        """
        Get the vector index for a library.

        If the index doesn't exist yet, build it.
        If the index is dirty (data changed), rebuild it.

        Args:
            library_id: The library's unique identifier

        Returns:
            The vector index

        Raises:
            KeyError: If library doesn't exist
            ValueError: If library has no chunks
        """
        # Check if index needs rebuilding (doesn't exist or is dirty)
        if library_id not in self._indexes or library_id in self._dirty_indexes:
            # Build/rebuild the index
            self.build_index(library_id)
            # Mark as clean after rebuild
            self._dirty_indexes.discard(library_id)

        return self._indexes[library_id]

    def invalidate_index(self, library_id: UUID) -> None:
        """
        Mark a library's index as dirty.

        The index will be rebuilt on the next query.
        This is called automatically when chunks are added/updated/deleted.

        Args:
            library_id: The library's unique identifier
        """
        self._dirty_indexes.add(library_id)

    def _create_index(
        self, index_type: IndexType, dimension: int, config: Optional[Dict]
    ) -> VectorIndex:
        """
        Factory method to create a vector index.

        Args:
            index_type: Type of index
            dimension: Vector dimension
            config: Index configuration

        Returns:
            A VectorIndex instance
        """
        config = config or {}

        # Factory pattern: match index type to implementation
        if index_type == IndexType.FLAT:
            return FlatIndex(dimension=dimension)
        elif index_type == IndexType.HNSW:
            return HNSWIndex(dimension=dimension, **config)
        else:
            # This should never happen due to enum validation
            raise ValueError(f"Unknown index type: {index_type}")
