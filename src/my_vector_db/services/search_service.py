"""
Search Service - Business logic for vector search operations.

This service handles kNN (k-nearest neighbor) search with optional metadata filtering.
"""

import time
from typing import List, Optional, Tuple, Union
from uuid import UUID

from my_vector_db.domain.models import Chunk, SearchFilters, SearchFiltersWithCallable
from my_vector_db.filters.evaluator import evaluate_search_filters
from my_vector_db.services.library_service import LibraryService
from my_vector_db.storage import VectorStorage


class SearchService:
    """
    Service for performing vector similarity search.

    This service coordinates between the vector index and storage to perform
    efficient kNN search with optional metadata filtering.
    """

    def __init__(self, storage: VectorStorage, library_service: LibraryService) -> None:
        """
        Initialize the search service.

        Args:
            storage: The storage instance
            library_service: Library service for accessing indexes
        """
        self._storage = storage
        self._library_service = library_service

    def search(
        self,
        library_id: UUID,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[SearchFilters, SearchFiltersWithCallable]] = None,
    ) -> Tuple[List[Tuple[Chunk, float]], float]:
        """
        Perform k-nearest neighbor search with optional filtering.

        Algorithm:
        1. Get the library's vector index
        2. Over-fetch if filters provided (k*3 to account for filtering)
        3. Perform kNN search on the index
        4. Retrieve full chunk data for results
        5. Apply filters in a single pass if provided
        6. Return top k results with similarity scores

        The filtering uses a single-pass approach for efficiency, evaluating each
        chunk against all filter criteria (metadata, time-based, document IDs) in
        one iteration.

        Args:
            library_id: The library to search
            query_embedding: Query vector
            k: Number of results to return after filtering
            filters: Optional search filters (metadata, time-based, document IDs)

        Returns:
            Tuple of (results, query_time_ms) where results is a list of
            (Chunk, similarity_score) tuples sorted by similarity score

        Raises:
            KeyError: If library doesn't exist
            ValueError: If library has no chunks
        """
        start_time = time.time()

        # Get library and its index
        library = self._library_service.get_library(library_id)
        if library is None:
            raise KeyError(f"Library ID {library_id} not found")

        # Get index (this will build it if not already built)
        index = self._library_service.get_index(library_id)

        # Over-fetch if filters are provided (fetch more, then filter, then limit)
        # Only over-fetch if there are actual filter criteria (not just empty SearchFilters object)
        # Note: custom_filter is client-side only and not present in SearchFilters
        has_filters = filters and (
            filters.metadata is not None
            or filters.created_after is not None
            or filters.created_before is not None
            or filters.document_ids is not None
        )
        fetch_k = k * 3 if has_filters else k

        # Perform kNN search on the index
        knn_results = index.search(query_embedding, fetch_k)

        # Retrieve full chunk data for each result
        chunks_with_scores = []
        for chunk_id, score in knn_results:
            chunk = self._storage.get_chunk(chunk_id)
            if chunk:  # Chunk might have been deleted after index was built
                chunks_with_scores.append((chunk, score))

        # Apply metadata filters if provided (single-pass filtering)
        if filters:
            chunks_with_scores = [
                (chunk, score)
                for chunk, score in chunks_with_scores
                if evaluate_search_filters(chunk, filters)
            ]

        # Limit to top k results (already sorted by score from index.search)
        final_results = chunks_with_scores[:k]

        query_time_ms = (time.time() - start_time) * 1000
        return final_results, query_time_ms
