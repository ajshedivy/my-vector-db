"""
Business logic services for the Vector Database.

This package contains service classes that implement the core business logic
for managing libraries, documents, and performing search operations. Services
act as an intermediary between the API layer and the storage/index layers.

Available Services:
- LibraryService: Manage libraries and their vector indexes
- DocumentService: Handle document and chunk operations
- SearchService: Perform similarity searches across documents

Example:
    from my_vector_db.services import LibraryService, DocumentService, SearchService
    from my_vector_db.storage import VectorStorage

    storage = VectorStorage()
    library_service = LibraryService(storage)
    document_service = DocumentService(storage, library_service)
    search_service = SearchService(storage, library_service)
"""

from my_vector_db.services.document_service import DocumentService
from my_vector_db.services.library_service import LibraryService
from my_vector_db.services.search_service import SearchService

__all__ = [
    "LibraryService",
    "DocumentService",
    "SearchService",
]
