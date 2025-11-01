"""
FastAPI routes for the Vector Database API.

This module defines all REST API endpoints.
Each route delegates business logic to the appropriate service.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from my_vector_db.domain.models import IndexType
from my_vector_db.api.schemas import (
    BatchChunkCreateRequest,
    BatchChunkResponse,
    BatchDocumentCreateRequest,
    BatchDocumentResponse,
    ChunkResponse,
    CreateChunkRequest,
    CreateDocumentRequest,
    CreateLibraryRequest,
    DocumentResponse,
    IndexBuildResponse,
    LibraryResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    UpdateChunkRequest,
    UpdateDocumentRequest,
    UpdateLibraryRequest,
)
from my_vector_db.services.document_service import DocumentService
from my_vector_db.services.library_service import LibraryService
from my_vector_db.services.search_service import SearchService
from my_vector_db.storage import storage

# Initialize services
library_service = LibraryService(storage)
document_service = DocumentService(storage, library_service)
search_service = SearchService(storage, library_service)

# Create router
router = APIRouter()


# ============================================================================
# Library Endpoints
# ============================================================================


@router.post(
    "/libraries",
    response_model=LibraryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["libraries"],
)
def create_library(request: CreateLibraryRequest) -> LibraryResponse:
    """
    Create a new library.

    Args:
        request: Library creation request

    Returns:
        The created library
    """
    library = library_service.create_library(
        name=request.name,
        metadata=request.metadata,
        index_type=IndexType(request.index_type),
        index_config=request.index_config,
    )

    return LibraryResponse(
        id=library.id,
        name=library.name,
        document_ids=library.document_ids,
        metadata=library.metadata,
        index_type=library.index_type.value,
        index_config=library.index_config,
        created_at=library.created_at,
        updated_at=library.updated_at,
    )


@router.get(
    "/libraries",
    response_model=list[LibraryResponse],
    status_code=status.HTTP_200_OK,
    tags=["libraries"],
)
def list_libraries() -> list[LibraryResponse]:
    """
    Get all libraries.

    Returns:
        List of all libraries

    TODO: Implement
    - Call library_service.list_libraries()
    - Convert to LibraryResponse list
    """
    libraries = library_service.list_libraries()
    return [
        LibraryResponse(
            id=library.id,
            name=library.name,
            document_ids=library.document_ids,
            metadata=library.metadata,
            index_type=library.index_type.value,
            index_config=library.index_config,
            created_at=library.created_at,
            updated_at=library.updated_at,
        )
        for library in libraries
    ]


@router.get(
    "/libraries/{library_id}",
    response_model=LibraryResponse,
    status_code=status.HTTP_200_OK,
    tags=["libraries"],
)
def get_library(library_id: UUID) -> LibraryResponse:
    """
    Get a library by ID.

    Args:
        library_id: Library unique identifier

    Returns:
        The library

    Raises:
        HTTPException: 404 if library not found
    """
    library = library_service.get_library(library_id)
    if library is None:
        raise HTTPException(status_code=404, detail="Library not found")

    return LibraryResponse(
        id=library.id,
        name=library.name,
        document_ids=library.document_ids,
        metadata=library.metadata,
        index_type=library.index_type.value,
        index_config=library.index_config,
        created_at=library.created_at,
        updated_at=library.updated_at,
    )


@router.put(
    "/libraries/{library_id}",
    response_model=LibraryResponse,
    status_code=status.HTTP_200_OK,
    tags=["libraries"],
)
def update_library(library_id: UUID, request: UpdateLibraryRequest) -> LibraryResponse:
    """
    Update a library.

    Args:
        library_id: Library unique identifier
        request: Update request with optional fields

    Returns:
        The updated library

    Raises:
        HTTPException: 404 if library not found
    """
    try:
        library = library_service.update_library(
            library_id=library_id,
            name=request.name,
            metadata=request.metadata,
            index_type=IndexType(request.index_type) if request.index_type else None,
            index_config=request.index_config,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Library not found")

    return LibraryResponse(
        id=library.id,
        name=library.name,
        document_ids=library.document_ids,
        metadata=library.metadata,
        index_type=library.index_type.value,
        index_config=library.index_config,
        created_at=library.created_at,
        updated_at=library.updated_at,
    )


@router.delete(
    "/libraries/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["libraries"],
)
def delete_library(library_id: UUID) -> None:
    """
    Delete a library and all its data.

    Args:
        library_id: Library unique identifier

    Raises:
        HTTPException: 404 if library not found
    """
    deleted = library_service.delete_library(library_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Library not found")


@router.post(
    "/libraries/{library_id}/index/build",
    response_model=IndexBuildResponse,
    status_code=status.HTTP_200_OK,
    tags=["libraries"],
)
def build_library_index(library_id: UUID) -> IndexBuildResponse:
    """
    Build or rebuild the vector index for a library.

    This endpoint explicitly builds the vector index from all chunks in the library.
    For HNSW indexes, this should be called after adding/updating chunks to optimize search performance.
    FLAT indexes automatically update, but this can still be used to validate the index.

    Args:
        library_id: Library unique identifier

    Returns:
        Index build information including total vectors and dimension

    Raises:
        HTTPException: 404 if library not found, 400 if no chunks or invalid dimensions
    """
    try:
        build_result = library_service.build_index(library_id)
        return IndexBuildResponse(
            library_id=build_result.library_id,
            total_vectors=build_result.total_vectors,
            dimension=build_result.dimension,
            index_type=build_result.index_type.value,  # Convert enum to string
            index_config=build_result.index_config,
            status="success",
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Library not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Document Endpoints
# ============================================================================


@router.post(
    "/libraries/{library_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def create_document(
    library_id: UUID, request: CreateDocumentRequest
) -> DocumentResponse:
    """
    Create a new document in a library.

    Args:
        library_id: Parent library ID
        request: Document creation request

    Returns:
        The created document

    Raises:
        HTTPException: 404 if library not found
    """
    try:
        document = document_service.create_document(
            library_id=library_id, name=request.name, metadata=request.metadata
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Library not found")

    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        name=document.name,
        chunk_ids=document.chunk_ids,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.get(
    "/libraries/{library_id}/documents",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK,
    tags=["documents"],
)
def list_documents(library_id: UUID) -> list[DocumentResponse]:
    """
    Get all documents in a library.

    Args:
        library_id: Library unique identifier

    Returns:
        List of documents
    """
    documents = document_service.list_documents(library_id)
    return [
        DocumentResponse(
            id=document.id,
            library_id=document.library_id,
            name=document.name,
            chunk_ids=document.chunk_ids,
            metadata=document.metadata,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
        for document in documents
    ]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    tags=["documents"],
)
def get_document(document_id: UUID) -> DocumentResponse:
    """
    Get a document by ID.

    Args:
        document_id: Document unique identifier

    Returns:
        The document

    Raises:
        HTTPException: 404 if document not found
    """
    document = document_service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        name=document.name,
        chunk_ids=document.chunk_ids,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.put(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    tags=["documents"],
)
def update_document(
    document_id: UUID, request: UpdateDocumentRequest
) -> DocumentResponse:
    """
    Update a document.

    Args:
        document_id: Document unique identifier
        request: Update request with optional fields

    Returns:
        The updated document

    Raises:
        HTTPException: 404 if document not found
    """
    try:
        document = document_service.update_document(
            document_id=document_id, name=request.name, metadata=request.metadata
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        name=document.name,
        chunk_ids=document.chunk_ids,
        metadata=document.metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
)
def delete_document(document_id: UUID) -> None:
    """
    Delete a document and all its chunks.

    Args:
        document_id: Document unique identifier

    Raises:
        HTTPException: 404 if document not found
    """
    deleted = document_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


@router.post(
    "/libraries/{library_id}/documents/batch",
    response_model=BatchDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def create_documents_batch(
    library_id: UUID, request: BatchDocumentCreateRequest
) -> BatchDocumentResponse:
    """
    Create multiple documents in a single request.

    This is more efficient than creating documents one by one.

    Args:
        library_id: Parent library ID
        request: Batch document creation request

    Returns:
        Batch response with all created documents

    Raises:
        HTTPException: 404 if library not found
    """
    try:
        # Convert request documents to domain models
        from my_vector_db.domain.models import Document

        documents = [
            Document(library_id=library_id, name=doc.name, metadata=doc.metadata)
            for doc in request.documents
        ]

        # Create all documents
        created_documents = document_service.create_documents_batch(
            library_id=library_id, documents=documents
        )

        # Convert to response format
        document_responses = [
            DocumentResponse(
                id=doc.id,
                library_id=doc.library_id,
                name=doc.name,
                chunk_ids=doc.chunk_ids,
                metadata=doc.metadata,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
            for doc in created_documents
        ]

        return BatchDocumentResponse(
            documents=document_responses, total=len(document_responses)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Library not found")


# ============================================================================
# Chunk Endpoints
# ============================================================================


@router.post(
    "/documents/{document_id}/chunks",
    response_model=ChunkResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["chunks"],
)
def create_chunk(document_id: UUID, request: CreateChunkRequest) -> ChunkResponse:
    """
    Create a new chunk in a document.

    This also adds the chunk's embedding to the library's vector index.

    Args:
        document_id: Parent document ID
        request: Chunk creation request

    Returns:
        The created chunk

    Raises:
        HTTPException: 404 if document not found
    """
    try:
        chunk = document_service.create_chunk(
            document_id=document_id,
            text=request.text,
            embedding=request.embedding,
            metadata=request.metadata,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")

    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        metadata=chunk.metadata,
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
    )


@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[ChunkResponse],
    status_code=status.HTTP_200_OK,
    tags=["chunks"],
)
def list_chunks(document_id: UUID) -> list[ChunkResponse]:
    """
    Get all chunks in a document.

    Args:
        document_id: Document unique identifier

    Returns:
        List of chunks
    """
    chunks = document_service.list_chunks(document_id)
    return [
        ChunkResponse(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            embedding=chunk.embedding,
            metadata=chunk.metadata,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
        )
        for chunk in chunks
    ]


@router.get(
    "/chunks/{chunk_id}",
    response_model=ChunkResponse,
    status_code=status.HTTP_200_OK,
    tags=["chunks"],
)
def get_chunk(chunk_id: UUID) -> ChunkResponse:
    """
    Get a chunk by ID.

    Args:
        chunk_id: Chunk unique identifier

    Returns:
        The chunk

    Raises:
        HTTPException: 404 if chunk not found
    """
    chunk = document_service.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        metadata=chunk.metadata,
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
    )


@router.put(
    "/chunks/{chunk_id}",
    response_model=ChunkResponse,
    status_code=status.HTTP_200_OK,
    tags=["chunks"],
)
def update_chunk(chunk_id: UUID, request: UpdateChunkRequest) -> ChunkResponse:
    """
    Update a chunk.

    If embedding is updated, the vector index is also updated.

    Args:
        chunk_id: Chunk unique identifier
        request: Update request with optional fields

    Returns:
        The updated chunk

    Raises:
        HTTPException: 404 if chunk not found
    """
    try:
        chunk = document_service.update_chunk(
            chunk_id=chunk_id,
            text=request.text,
            embedding=request.embedding,
            metadata=request.metadata,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding=chunk.embedding,
        metadata=chunk.metadata,
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
    )


@router.delete(
    "/chunks/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["chunks"],
)
def delete_chunk(chunk_id: UUID) -> None:
    """
    Delete a chunk.

    This also removes the chunk from the vector index.

    Args:
        chunk_id: Chunk unique identifier

    Raises:
        HTTPException: 404 if chunk not found
    """
    deleted = document_service.delete_chunk(chunk_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chunk not found")


@router.post(
    "/documents/{document_id}/chunks/batch",
    response_model=BatchChunkResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["chunks"],
)
def create_chunks_batch(
    document_id: UUID, request: BatchChunkCreateRequest
) -> BatchChunkResponse:
    """
    Create multiple chunks in a single request.

    This is more efficient than creating chunks one by one and only
    invalidates the vector index once after all chunks are added.

    Args:
        document_id: Parent document ID
        request: Batch chunk creation request

    Returns:
        Batch response with all created chunks

    Raises:
        HTTPException: 404 if document not found
    """
    try:
        # Convert request chunks to domain models
        from my_vector_db.domain.models import Chunk

        chunks = [
            Chunk(
                document_id=document_id,
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
            )
            for chunk in request.chunks
        ]

        # Create all chunks
        created_chunks = document_service.create_chunks_batch(
            document_id=document_id, chunks=chunks
        )

        # Convert to response format
        chunk_responses = [
            ChunkResponse(
                id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=chunk.metadata,
                created_at=chunk.created_at,
                updated_at=chunk.updated_at,
            )
            for chunk in created_chunks
        ]

        return BatchChunkResponse(chunks=chunk_responses, total=len(chunk_responses))
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")


# ============================================================================
# Search Endpoint
# ============================================================================


@router.post(
    "/libraries/{library_id}/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["search"],
)
def query_library(library_id: UUID, request: QueryRequest) -> QueryResponse:
    """
    Perform k-nearest neighbor search on a library.

    Args:
        library_id: Library to search
        request: Query request with embedding, k, and optional filters

    Returns:
        Query results with similarity scores

    Raises:
        HTTPException: 404 if library not found
        HTTPException: 400 if library has no chunks
    """
    try:
        results, query_time_ms = search_service.search(
            library_id=library_id,
            query_embedding=request.embedding,
            k=request.k,
            filters=request.filters,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Library not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    query_results = [
        QueryResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=score,
            metadata=chunk.metadata,
        )
        for chunk, score in results
    ]

    return QueryResponse(
        results=query_results, total=len(query_results), query_time_ms=query_time_ms
    )


# ============================================================================
# Admin / Persistence Endpoints
# ============================================================================


@router.post(
    "/admin/snapshot/save",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    tags=["admin"],
)
def save_snapshot():
    """
    Manually trigger a snapshot save.

    This saves the current database state to disk immediately,
    regardless of the automatic save threshold.

    Returns:
        Success message with snapshot details

    Raises:
        HTTPException: 503 if persistence is not enabled
    """
    from datetime import datetime
    from my_vector_db.serialization import get_snapshot_info

    if not storage._persistence_enabled:
        raise HTTPException(
            status_code=503,
            detail="Persistence is not enabled. Set ENABLE_PERSISTENCE=true in environment.",
        )

    try:
        storage.save_snapshot()

        # Get stats
        stats = {
            "libraries": len(storage._libraries),
            "documents": len(storage._documents),
            "chunks": len(storage._chunks),
        }

        snapshot_info = (
            get_snapshot_info(storage._snapshot_path) if storage._snapshot_path else {}
        )

        return {
            "message": "Snapshot saved successfully",
            "snapshot_path": str(storage._snapshot_path)
            if storage._snapshot_path
            else "unknown",
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "snapshot_info": snapshot_info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save snapshot: {str(e)}"
        )


@router.post(
    "/admin/snapshot/restore",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    tags=["admin"],
)
def restore_snapshot():
    """
    Restore database state from the latest snapshot.

    WARNING: This will replace all current data with the snapshot data.

    Returns:
        Success message with restored counts

    Raises:
        HTTPException: 404 if no snapshot exists
        HTTPException: 503 if persistence is not enabled
    """
    if not storage._persistence_enabled:
        raise HTTPException(
            status_code=503,
            detail="Persistence is not enabled. Set ENABLE_PERSISTENCE=true in environment.",
        )

    if not storage.snapshot_exists():
        raise HTTPException(
            status_code=404, detail="No snapshot file found to restore from."
        )

    try:
        loaded = storage.load_snapshot()

        if not loaded:
            raise HTTPException(status_code=500, detail="Failed to load snapshot")

        # Get restored stats
        stats = {
            "libraries": len(storage._libraries),
            "documents": len(storage._documents),
            "chunks": len(storage._chunks),
        }

        return {"message": "Snapshot restored successfully", "stats": stats}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid snapshot: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to restore snapshot: {str(e)}"
        )


@router.get(
    "/admin/persistence/status",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    tags=["admin"],
)
def get_persistence_status():
    """
    Get current persistence status and statistics.

    Returns:
        Persistence status including snapshot info
    """
    from my_vector_db.serialization import get_snapshot_info

    if not storage._persistence_enabled:
        return {
            "enabled": False,
            "snapshot_exists": False,
            "operations_since_save": 0,
            "snapshot_info": None,
        }

    snapshot_info = None
    if storage._snapshot_path and storage._snapshot_path.exists():
        snapshot_info = get_snapshot_info(storage._snapshot_path)

    return {
        "enabled": True,
        "snapshot_exists": storage.snapshot_exists(),
        "operations_since_save": storage._operation_counter,
        "snapshot_info": snapshot_info,
        "save_threshold": storage._save_every,
        "stats": {
            "libraries": len(storage._libraries),
            "documents": len(storage._documents),
            "chunks": len(storage._chunks),
        },
    }
