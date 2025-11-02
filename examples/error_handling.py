#!/usr/bin/env python3
"""
Example: Error Handling and Production Patterns

This script demonstrates proper error handling, retry logic,
and production-ready patterns for the Vector Database SDK.

Prerequisites:
1. API server running on http://localhost:8000 (for successful operations)
2. Vector DB SDK installed: pip install my-vector-db

What you'll learn:
- Handling specific exceptions
- Using context managers for cleanup
- Implementing retry logic
- Connection error handling
- Validation error patterns
"""

import time
from my_vector_db.sdk import (
    VectorDBClient,
    VectorDBError,
    ValidationError,
    NotFoundError,
    ServerError,
    ServerConnectionError,
    TimeoutError,
)


def main():
    """Demonstrate error handling patterns."""

    print("=" * 70)
    print("Error Handling and Production Patterns Example")
    print("=" * 70)

    # Pattern 1: Context manager (recommended)
    print("\nPattern 1: Using context manager for automatic cleanup\n")

    try:
        with VectorDBClient(base_url="http://localhost:8000") as client:
            library = client.create_library(name="error_demo", index_type="flat")
            print(f"✓ Created library (auto-cleanup on exit)")
            # Client automatically closes on exit
    except ServerConnectionError as e:
        print(f"✗ Connection error: {e}")
        print("  Make sure the API server is running on http://localhost:8000")

    # Pattern 2: Explicit error handling
    print("\n" + "=" * 70)
    print("Pattern 2: Specific exception handling")
    print("=" * 70 + "\n")

    client = VectorDBClient(base_url="http://localhost:8000")

    try:
        # This will raise NotFoundError
        library = client.get_library("00000000-0000-0000-0000-000000000000")
    except NotFoundError:
        print("✓ Correctly handled NotFoundError")

    try:
        # This will raise ValidationError (empty name)
        library = client.create_library(name="", index_type="flat")
    except ValidationError as e:
        print(f"✓ Correctly handled ValidationError: {e}")
    except ValueError as e:
        # SDK may raise ValueError for enum/type validation before API call
        print(f"✓ Correctly handled ValueError: {e}")

    # Pattern 3: Retry logic for transient errors
    print("\n" + "=" * 70)
    print("Pattern 3: Retry logic with exponential backoff")
    print("=" * 70 + "\n")

    def create_library_with_retry(client, name, max_retries=3):
        """Create library with automatic retry on transient errors."""
        for attempt in range(max_retries):
            try:
                library = client.create_library(name=name, index_type="flat")
                print(f"✓ Created library on attempt {attempt + 1}")
                return library
            except ServerConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"  Connection failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed after {max_retries} attempts")
                    raise
            except ValidationError:
                # Don't retry validation errors (they won't succeed)
                print("✗ Validation error - not retrying")
                raise

    try:
        library = create_library_with_retry(client, "retry_demo")
    except Exception as e:
        print(f"Failed to create library: {e}")

    # Pattern 4: Graceful degradation
    print("\n" + "=" * 70)
    print("Pattern 4: Graceful degradation")
    print("=" * 70 + "\n")

    def search_with_fallback(client, library_id, embedding, k=10):
        """Search with fallback to smaller k if needed."""
        try:
            return client.search(library_id=library_id, embedding=embedding, k=k)
        except (ServerError, TimeoutError) as e:
            print(f"  Primary search failed ({e}), trying with smaller k...")
            try:
                return client.search(library_id=library_id, embedding=embedding, k=5)
            except Exception:
                print("  Fallback also failed, returning empty results")
                return None

    # Pattern 5: Resource cleanup
    print("\n" + "=" * 70)
    print("Pattern 5: Guaranteed cleanup with finally")
    print("=" * 70 + "\n")

    created_libraries = []

    try:
        for i in range(3):
            lib = client.create_library(name=f"temp_lib_{i}", index_type="flat")
            created_libraries.append(lib)
            print(f"  Created library {i + 1}")

        # Simulate error
        raise Exception("Simulated error during processing")

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print("  Cleaning up created resources...\n")

    finally:
        # Cleanup happens even if error occurred
        for lib in created_libraries:
            try:
                client.delete_library(lib.id)
                print(f"  ✓ Cleaned up {lib.name}")
            except Exception as e:
                print(f"  ✗ Failed to cleanup {lib.name}: {e}")

    # Pattern 6: Comprehensive try-except-finally
    print("\n" + "=" * 70)
    print("Pattern 6: Production-ready error handling")
    print("=" * 70 + "\n")

    try:
        library = client.create_library(name="production_demo", index_type="flat")
        document = client.create_document(library_id=library.id, name="doc")

        chunks = [
            {"text": f"Chunk {i}", "embedding": [0.1, 0.2, 0.3]} for i in range(3)
        ]
        client.add_chunks(document_id=document.id, chunks=chunks)

        print("✓ Successfully completed all operations")

    except ValidationError as e:
        print(f"✗ Validation error: {e}")
    except NotFoundError as e:
        print(f"✗ Resource not found: {e}")
    except ServerConnectionError as e:
        print(f"✗ Connection error: {e}")
    except TimeoutError as e:
        print(f"✗ Request timeout: {e}")
    except VectorDBError as e:
        print(f"✗ Vector DB error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    finally:
        try:
            if "library" in locals():
                client.delete_library(library.id)
                print("✓ Cleanup completed")
        except Exception:
            pass  # Best effort cleanup

    client.close()
    print("\n✓ Example complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
