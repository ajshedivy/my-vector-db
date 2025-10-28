"""
Pure serialization functions for vector database persistence.

This module provides stateless functions to serialize and deserialize
database entities to/from JSON format. No circular dependencies.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import UUID

from my_vector_db.domain.models import Chunk, Document, Library


class UUIDEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for UUID and datetime objects.

    Converts:
    - UUID -> string
    - datetime -> ISO format string
    """

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def serialize_to_json(
    libraries: List[Library],
    documents: List[Document],
    chunks: List[Chunk],
    file_path: Path,
) -> None:
    """
    Serialize storage entities to JSON file with atomic write.

    Uses atomic write pattern (temp file + rename) to prevent corruption.

    Args:
        libraries: List of library entities
        documents: List of document entities
        chunks: List of chunk entities
        file_path: Target file path for snapshot

    Raises:
        IOError: If unable to write to file
    """
    # Prepare snapshot data
    snapshot = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "libraries": [lib.model_dump() for lib in libraries],
        "documents": [doc.model_dump() for doc in documents],
        "chunks": [chunk.model_dump() for chunk in chunks],
    }

    # Atomic write: write to temp file, then rename
    temp_path = file_path.parent / f"{file_path.name}.tmp"

    with open(temp_path, "w") as f:
        json.dump(snapshot, f, indent=2, cls=UUIDEncoder)

    # Atomic rename (POSIX systems)
    temp_path.rename(file_path)


def deserialize_from_json(
    file_path: Path,
) -> Tuple[List[Library], List[Document], List[Chunk]]:
    """
    Deserialize storage entities from JSON file.

    Args:
        file_path: Path to snapshot file

    Returns:
        Tuple of (libraries, documents, chunks)

    Raises:
        FileNotFoundError: If snapshot file doesn't exist
        ValueError: If snapshot is corrupted or incompatible version
        json.JSONDecodeError: If JSON is malformed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Validate version
    if data.get("version") != "1.0":
        raise ValueError(f"Incompatible snapshot version: {data.get('version')}")

    # Reconstruct entities
    libraries = []
    for lib_data in data.get("libraries", []):
        lib_data["id"] = UUID(lib_data["id"])
        lib_data["document_ids"] = [UUID(doc_id) for doc_id in lib_data["document_ids"]]
        lib_data["created_at"] = datetime.fromisoformat(lib_data["created_at"])
        libraries.append(Library(**lib_data))

    documents = []
    for doc_data in data.get("documents", []):
        doc_data["id"] = UUID(doc_data["id"])
        doc_data["library_id"] = UUID(doc_data["library_id"])
        doc_data["chunk_ids"] = [UUID(chunk_id) for chunk_id in doc_data["chunk_ids"]]
        doc_data["created_at"] = datetime.fromisoformat(doc_data["created_at"])
        documents.append(Document(**doc_data))

    chunks = []
    for chunk_data in data.get("chunks", []):
        chunk_data["id"] = UUID(chunk_data["id"])
        chunk_data["document_id"] = UUID(chunk_data["document_id"])
        chunk_data["created_at"] = datetime.fromisoformat(chunk_data["created_at"])
        chunks.append(Chunk(**chunk_data))

    return libraries, documents, chunks


def get_snapshot_info(file_path: Path) -> Dict[str, Any]:
    """
    Get metadata about a snapshot file.

    Args:
        file_path: Path to snapshot file

    Returns:
        Dictionary with snapshot metadata
    """
    if file_path.exists():
        stat = file_path.stat()
        return {
            "exists": True,
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    else:
        return {
            "exists": False,
            "path": str(file_path),
            "size_bytes": 0,
            "last_modified": None,
        }
