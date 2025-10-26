"""
SDK Client Tests

Tests the VectorDB SDK client with a focus on error handling.
Run with: pytest tests/test_sdk.py -v
"""

import pytest
from unittest.mock import Mock, patch
import httpx

from my_vector_db.sdk import VectorDBClient
from my_vector_db.sdk.exceptions import (
    ServerConnectionError,
    NotFoundError,
    ServerError,
    TimeoutError,
    ValidationError,
    VectorDBError,
)


class TestSDKErrorHandling:
    """Tests for SDK error handling and exception mapping."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock httpx client."""
        return Mock(spec=httpx.Client)

    @pytest.fixture
    def sdk_client(self, mock_client):
        """Create SDK client with mocked httpx client."""
        with patch("my_vector_db.sdk.client.httpx.Client", return_value=mock_client):
            client = VectorDBClient(base_url="http://localhost:8000")
            # Replace the internal httpx client with our mock
            client._client = mock_client
            return client

    def test_connection_error(self, sdk_client, mock_client):
        """Test that connection errors are properly handled."""
        # Simulate connection error
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(ServerConnectionError) as exc_info:
            sdk_client.list_libraries()

        assert "Cannot connect to VectorDB" in str(exc_info.value)
        assert "Ensure the server is running" in str(exc_info.value)

    def test_timeout_error(self, sdk_client, mock_client):
        """Test that timeout errors are properly handled."""
        # Simulate timeout
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(TimeoutError) as exc_info:
            sdk_client.list_libraries()

        assert "Request timed out after 30.0s" in str(exc_info.value)
        assert "overloaded or unreachable" in str(exc_info.value)

    def test_404_not_found_error(self, sdk_client, mock_client):
        """Test that 404 errors are mapped to NotFoundError."""
        # Create mock response with 404
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Library not found"}

        # Create HTTPStatusError
        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError) as exc_info:
            sdk_client.get_library("nonexistent-id")

        assert "Library not found" in str(exc_info.value)
        assert "list method" in str(exc_info.value)

    def test_400_validation_error(self, sdk_client, mock_client):
        """Test that 400 errors are mapped to ValidationError."""
        # Create mock response with 400
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Library has no chunks to query"}

        # Create HTTPStatusError
        error = httpx.HTTPStatusError(
            "400 Bad Request", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(ValidationError) as exc_info:
            # Use flat API: client.search() instead of client.search.query()
            sdk_client.search(library_id="lib-id", embedding=[0.1, 0.2])

        assert "Invalid request" in str(exc_info.value)
        assert "Library has no chunks" in str(exc_info.value)

    def test_500_server_error(self, sdk_client, mock_client):
        """Test that 500+ errors are mapped to ServerError."""
        # Create mock response with 500
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}

        # Create HTTPStatusError
        error = httpx.HTTPStatusError(
            "500 Internal Server Error", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(ServerError) as exc_info:
            sdk_client.list_libraries()

        assert "Server error" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    def test_unknown_status_code(self, sdk_client, mock_client):
        """Test that unexpected status codes are mapped to VectorDBError."""
        # Create mock response with unusual status code
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 418  # I'm a teapot
        mock_response.json.return_value = {"detail": "I'm a teapot"}

        # Create HTTPStatusError
        error = httpx.HTTPStatusError(
            "418 I'm a teapot", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(VectorDBError) as exc_info:
            sdk_client.list_libraries()

        assert "API error (418)" in str(exc_info.value)
        assert exc_info.value.status_code == 418

    def test_malformed_json_response(self, sdk_client, mock_client):
        """Test handling of non-JSON error responses."""
        # Create mock response with 404 but no JSON
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.side_effect = ValueError("Invalid JSON")

        # Create HTTPStatusError
        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError) as exc_info:
            sdk_client.get_library("test-id")

        # Should still raise NotFoundError even with malformed JSON
        assert "404" in str(exc_info.value)

    def test_successful_request(self, sdk_client, mock_client):
        """Test that successful requests return data correctly."""
        # Create successful mock response
        # The decorator now returns response.json() directly
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "d6b2b8db-41a1-4b51-a2f3-26a95c7b0c1e",
                "name": "Test Library",
                "document_ids": [],
                "metadata": {},
                "index_type": "flat",
                "index_config": {},
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        mock_response.raise_for_status.return_value = None

        mock_client.get.return_value = mock_response

        result = sdk_client.list_libraries()

        # Result is now a list of Library objects
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].name == "Test Library"

    def test_request_error(self, sdk_client, mock_client):
        """Test that generic request errors are handled."""
        # Simulate SSL or other request error
        mock_client.get.side_effect = httpx.RequestError("SSL verification failed")

        with pytest.raises(VectorDBError) as exc_info:
            sdk_client.list_libraries()

        assert "Request failed" in str(exc_info.value)
        assert "SSL verification failed" in str(exc_info.value)


class TestSDKIntegration:
    """Integration tests for SDK with running server."""

    @pytest.fixture
    def client(self):
        """Create SDK client pointing to test server."""
        # This will only work if server is running
        return VectorDBClient(base_url="http://localhost:8000")

    def test_health_check_integration(self, client):
        """Test health check through SDK (requires running server)."""
        try:
            # Try to make a request
            result = client.list_libraries()
            # If we get here, the request succeeded
            # Result is a list of Library objects
            assert isinstance(result, list)
        except ServerConnectionError:
            # Server not running, skip test
            pytest.skip("Server not running")

    def test_create_and_list_library_integration(self, client):
        """Test creating and listing libraries (requires running server)."""
        try:
            # Create a library
            library = client.create_library(
                name="SDK Test Library",
                index_type="flat",
                index_config={"metric": "cosine"},
            )

            # library is now a Library object, not a dict
            assert library.id is not None
            assert library.name == "SDK Test Library"

            # List libraries
            result = client.list_libraries()
            # result is a list of Library objects
            assert isinstance(result, list)

            # Clean up
            client.delete_library(library.id)

        except ServerConnectionError:
            pytest.skip("Server not running")

    def test_404_error_integration(self, client):
        """Test 404 error handling with real server (requires running server)."""
        try:
            # Use a valid UUID format that doesn't exist
            # This will pass FastAPI validation but fail in the service layer
            non_existent_uuid = "00000000-0000-0000-0000-000000000000"

            with pytest.raises(NotFoundError) as exc_info:
                client.get_library(non_existent_uuid)

            assert "not found" in str(exc_info.value).lower()

        except ServerConnectionError:
            pytest.skip("Server not running")


class TestSDKComprehensiveErrorHandling:
    """Comprehensive tests covering all error scenarios and edge cases."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock httpx client."""
        return Mock(spec=httpx.Client)

    @pytest.fixture
    def sdk_client(self, mock_client):
        """Create SDK client with mocked httpx client."""
        with patch("my_vector_db.sdk.client.httpx.Client", return_value=mock_client):
            client = VectorDBClient(base_url="http://localhost:8000")
            client._client = mock_client
            return client

    # ========================================================================
    # HTTP Status Code Tests
    # ========================================================================

    def test_422_unprocessable_entity_error(self, sdk_client, mock_client):
        """Test that 422 errors (Pydantic validation) are mapped to ValidationError."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "detail": [
                {
                    "loc": ["body", "embedding"],
                    "msg": "field required",
                    "type": "value_error.missing",
                }
            ]
        }

        error = httpx.HTTPStatusError(
            "422 Unprocessable Entity", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(ValidationError) as exc_info:
            sdk_client.search(library_id="lib-id", embedding=[0.1, 0.2], k=5)

        assert "Validation error" in str(exc_info.value)

    def test_204_no_content_delete(self, sdk_client, mock_client):
        """Test that 204 NO CONTENT responses are handled correctly."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.raise_for_status.return_value = None

        mock_client.delete.return_value = mock_response

        # Should not raise an error and return None
        result = sdk_client.delete_library("lib-id")
        assert result is None

    def test_503_service_unavailable(self, sdk_client, mock_client):
        """Test that 503 errors are mapped to ServerError."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.json.return_value = {"detail": "Service temporarily unavailable"}

        error = httpx.HTTPStatusError(
            "503 Service Unavailable", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(ServerError) as exc_info:
            sdk_client.list_libraries()

        assert exc_info.value.status_code == 503
        assert "Service temporarily unavailable" in str(exc_info.value)

    # ========================================================================
    # Document Operations Error Tests
    # ========================================================================

    def test_create_document_library_not_found(self, sdk_client, mock_client):
        """Test creating a document in non-existent library."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Library not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError) as exc_info:
            sdk_client.create_document(
                library_id="00000000-0000-0000-0000-000000000001", name="Test Doc"
            )

        assert "Library not found" in str(exc_info.value)

    def test_get_document_not_found(self, sdk_client, mock_client):
        """Test getting non-existent document."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Document not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.get_document(library_id="lib-id", document_id="nonexistent-doc")

    def test_update_document_not_found(self, sdk_client, mock_client):
        """Test updating non-existent document."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Document not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.put.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.update_document(
                library_id="lib-id", document_id="nonexistent-doc", name="Updated Name"
            )

    def test_delete_document_not_found(self, sdk_client, mock_client):
        """Test deleting non-existent document."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Document not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.delete.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.delete_document(
                library_id="lib-id", document_id="nonexistent-doc"
            )

    # ========================================================================
    # Chunk Operations Error Tests
    # ========================================================================

    def test_create_chunk_document_not_found(self, sdk_client, mock_client):
        """Test creating a chunk in non-existent document."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Document not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.create_chunk(
                library_id="00000000-0000-0000-0000-000000000001",
                document_id="00000000-0000-0000-0000-000000000002",
                text="Test chunk",
                embedding=[0.1, 0.2, 0.3],
            )

    def test_get_chunk_not_found(self, sdk_client, mock_client):
        """Test getting non-existent chunk."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Chunk not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.get.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.get_chunk(
                library_id="lib-id", document_id="doc-id", chunk_id="nonexistent-chunk"
            )

    def test_update_chunk_not_found(self, sdk_client, mock_client):
        """Test updating non-existent chunk."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Chunk not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.put.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.update_chunk(
                library_id="lib-id",
                document_id="doc-id",
                chunk_id="nonexistent-chunk",
                text="Updated text",
            )

    def test_delete_chunk_not_found(self, sdk_client, mock_client):
        """Test deleting non-existent chunk."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Chunk not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.delete.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.delete_chunk(
                library_id="lib-id", document_id="doc-id", chunk_id="nonexistent-chunk"
            )

    # ========================================================================
    # Search-Specific Error Tests
    # ========================================================================

    def test_search_library_not_found(self, sdk_client, mock_client):
        """Test searching in non-existent library."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Library not found"}

        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(NotFoundError):
            sdk_client.search(
                library_id="nonexistent-lib", embedding=[0.1, 0.2, 0.3], k=5
            )

    def test_search_empty_library(self, sdk_client, mock_client):
        """Test searching in library with no chunks (400 validation error)."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Library has no chunks to query"}

        error = httpx.HTTPStatusError(
            "400 Bad Request", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(ValidationError) as exc_info:
            sdk_client.search(library_id="empty-lib", embedding=[0.1, 0.2, 0.3], k=5)

        assert "Library has no chunks" in str(exc_info.value)

    def test_search_embedding_dimension_mismatch(self, sdk_client, mock_client):
        """Test searching with wrong embedding dimensions."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "detail": "Embedding dimension mismatch: expected 128, got 3"
        }

        error = httpx.HTTPStatusError(
            "400 Bad Request", request=Mock(), response=mock_response
        )
        mock_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error

        with pytest.raises(ValidationError) as exc_info:
            sdk_client.search(
                library_id="lib-id",
                embedding=[0.1, 0.2, 0.3],  # Wrong dimensions
                k=5,
            )

        assert "dimension mismatch" in str(exc_info.value).lower()

    # ========================================================================
    # Context Manager Tests
    # ========================================================================

    def test_context_manager_success(self):
        """Test that context manager properly closes the client."""
        with patch("my_vector_db.sdk.client.httpx.Client") as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance

            with VectorDBClient(base_url="http://localhost:8000") as client:
                assert client is not None

            # Verify close was called
            mock_client_instance.close.assert_called_once()

    def test_context_manager_with_exception(self):
        """Test that context manager closes client even when exception occurs."""
        with patch("my_vector_db.sdk.client.httpx.Client") as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance

            try:
                with VectorDBClient(base_url="http://localhost:8000") as client:
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Verify close was still called despite exception
            mock_client_instance.close.assert_called_once()

    # ========================================================================
    # Edge Cases and Network Errors
    # ========================================================================

    def test_dns_resolution_failure(self, sdk_client, mock_client):
        """Test handling of DNS resolution failures."""
        mock_client.get.side_effect = httpx.ConnectError("Failed to resolve hostname")

        with pytest.raises(ServerConnectionError) as exc_info:
            sdk_client.list_libraries()

        assert "Cannot connect to VectorDB" in str(exc_info.value)

    def test_ssl_certificate_error(self, sdk_client, mock_client):
        """Test handling of SSL certificate verification failures."""
        mock_client.get.side_effect = httpx.RequestError(
            "SSL: CERTIFICATE_VERIFY_FAILED"
        )

        with pytest.raises(VectorDBError) as exc_info:
            sdk_client.list_libraries()

        assert "Request failed" in str(exc_info.value)
        assert "SSL" in str(exc_info.value)

    def test_read_timeout(self, sdk_client, mock_client):
        """Test handling of read timeout errors."""
        mock_client.get.side_effect = httpx.ReadTimeout("Read operation timed out")

        with pytest.raises(TimeoutError) as exc_info:
            sdk_client.list_libraries()

        assert "timed out" in str(exc_info.value).lower()

    def test_empty_response_body(self, sdk_client, mock_client):
        """Test handling of responses with empty bodies."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b""
        mock_response.raise_for_status.return_value = None

        mock_client.get.return_value = mock_response

        result = sdk_client.list_libraries()
        # Should return empty dict which converts to empty list
        assert result == []

    def test_malformed_json_in_success_response(self, sdk_client, mock_client):
        """Test handling of malformed JSON in successful response."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b"not json"
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None

        mock_client.get.return_value = mock_response

        # This should raise an error since we can't parse the response
        with pytest.raises(Exception):  # Could be JSONDecodeError or similar
            sdk_client.list_libraries()

    # ========================================================================
    # Update Operations Tests
    # ========================================================================

    def test_update_library_success(self, sdk_client, mock_client):
        """Test successful library update."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Updated Library",
            "document_ids": [],
            "metadata": {"updated": True},
            "index_type": "flat",
            "index_config": {},
            "created_at": "2024-01-01T00:00:00",
        }
        mock_response.raise_for_status.return_value = None

        mock_client.put.return_value = mock_response

        result = sdk_client.update_library(
            library_id="00000000-0000-0000-0000-000000000001",
            name="Updated Library",
            metadata={"updated": True},
        )

        assert result.name == "Updated Library"
        assert result.metadata["updated"] is True

    def test_update_with_partial_fields(self, sdk_client, mock_client):
        """Test updating with only some fields provided."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Updated Library",
            "document_ids": [],
            "metadata": {},
            "index_type": "flat",
            "index_config": {},
            "created_at": "2024-01-01T00:00:00",
        }
        mock_response.raise_for_status.return_value = None

        mock_client.put.return_value = mock_response

        # Update only name, not metadata
        result = sdk_client.update_library(
            library_id="00000000-0000-0000-0000-000000000001", name="Updated Library"
        )

        assert result.name == "Updated Library"

    # ========================================================================
    # UUID Validation Tests
    # ========================================================================

    def test_malformed_library_id_on_create_document(self, sdk_client, mock_client):
        """Test that malformed library ID in create_document raises ValueError."""
        # UUID validation happens in Pydantic models during create operations
        with pytest.raises(ValueError):
            sdk_client.create_document(library_id="not-a-valid-uuid", name="Test Doc")

        # No HTTP request should be made because validation fails first
        mock_client.post.assert_not_called()

    def test_malformed_document_id_on_create_chunk(self, sdk_client, mock_client):
        """Test that malformed document ID in create_chunk raises ValueError."""
        with pytest.raises(ValueError):
            sdk_client.create_chunk(
                library_id="00000000-0000-0000-0000-000000000001",
                document_id="invalid-doc-id",
                text="Test chunk",
                embedding=[0.1, 0.2, 0.3],
            )

        mock_client.post.assert_not_called()

    def test_malformed_library_id_on_create_chunk(self, sdk_client, mock_client):
        """Test that malformed library ID in create_chunk is caught."""
        # The library_id isn't validated in ChunkCreate model, but document_id is
        # So this tests the full path validation
        with pytest.raises(ValueError):
            sdk_client.create_chunk(
                library_id="bad-lib-id",
                document_id="bad-doc-id",
                text="Test",
                embedding=[0.1],
            )

        mock_client.post.assert_not_called()

    def test_server_side_uuid_validation_on_get(self, sdk_client, mock_client):
        """Test that invalid UUIDs in GET requests return 422 from server."""
        # For GET/UPDATE/DELETE operations, UUID validation happens server-side
        # The SDK passes the string directly to the URL path
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "detail": [
                {
                    "loc": ["path", "library_id"],
                    "msg": "value is not a valid uuid",
                    "type": "type_error.uuid",
                }
            ]
        }

        error = httpx.HTTPStatusError(
            "422 Unprocessable Entity", request=Mock(), response=mock_response
        )
        mock_response.raise_for_status.side_effect = error
        mock_client.get.return_value = mock_response

        # SDK will make the request and get 422 from server
        with pytest.raises(ValidationError) as exc_info:
            sdk_client.get_library("not-a-uuid")

        assert "Validation error" in str(exc_info.value)
        # HTTP request WAS made (server-side validation)
        mock_client.get.assert_called_once()

    def test_empty_string_library_id_on_create_document(self, sdk_client, mock_client):
        """Test that empty string as library ID in create raises ValueError."""
        with pytest.raises(ValueError):
            sdk_client.create_document(library_id="", name="Test")

        mock_client.post.assert_not_called()

    def test_partial_uuid_on_create_document(self, sdk_client, mock_client):
        """Test that partial UUID in create_document raises ValueError."""
        with pytest.raises(ValueError):
            sdk_client.create_document(
                library_id="00000000-0000",  # Incomplete UUID
                name="Test",
            )

        mock_client.post.assert_not_called()

    def test_uuid_with_invalid_characters_on_create(self, sdk_client, mock_client):
        """Test that UUID with invalid characters in create raises ValueError."""
        with pytest.raises(ValueError):
            sdk_client.create_document(
                library_id="00000000-0000-0000-0000-00000000000g",  # 'g' is invalid
                name="Test",
            )

        mock_client.post.assert_not_called()

    def test_valid_uuid_formats_accepted(self, sdk_client, mock_client):
        """Test that various valid UUID formats are accepted."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "12345678-1234-5678-1234-567812345678",
            "name": "Test Library",
            "document_ids": [],
            "metadata": {},
            "index_type": "flat",
            "index_config": {},
            "created_at": "2024-01-01T00:00:00",
        }
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response

        # Test uppercase UUID
        result1 = sdk_client.get_library("12345678-1234-5678-1234-567812345678")
        assert result1.name == "Test Library"

        # Test lowercase UUID
        result2 = sdk_client.get_library("abcdef12-3456-7890-abcd-ef1234567890")
        assert result2.name == "Test Library"

        # Test mixed case UUID
        result3 = sdk_client.get_library("AbCdEf12-3456-7890-ABCD-EF1234567890")
        assert result3.name == "Test Library"

        # Verify HTTP requests were made
        assert mock_client.get.call_count == 3

    # ========================================================================
    # Multiple Operations Tests
    # ========================================================================

    def test_sequential_operations_after_error(self, sdk_client, mock_client):
        """Test that client can recover and make requests after an error."""
        # First request fails
        mock_response_error = Mock(spec=httpx.Response)
        mock_response_error.status_code = 404
        mock_response_error.json.return_value = {"detail": "Not found"}
        error = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response_error
        )
        mock_response_error.raise_for_status.side_effect = error

        # Second request succeeds
        mock_response_success = Mock(spec=httpx.Response)
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = []
        mock_response_success.raise_for_status.return_value = None

        mock_client.get.side_effect = [mock_response_error, mock_response_success]

        # First call should fail
        with pytest.raises(NotFoundError):
            sdk_client.get_library("nonexistent-id")

        # Second call should succeed
        result = sdk_client.list_libraries()
        assert result == []


class TestSDKFiltering:
    """Tests for SDK filtering functionality (declarative and custom)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock httpx client."""
        return Mock(spec=httpx.Client)

    @pytest.fixture
    def sdk_client(self, mock_client):
        """Create SDK client with mocked httpx client."""
        with patch("my_vector_db.sdk.client.httpx.Client", return_value=mock_client):
            client = VectorDBClient(base_url="http://localhost:8000")
            client._client = mock_client
            return client

    # ========================================================================
    # Declarative Filter Tests
    # ========================================================================

    def test_search_with_dict_filters(self, sdk_client, mock_client):
        """Test search with declarative filters passed as dict."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "Python tutorial",
                    "score": 0.95,
                    "metadata": {"category": "tech", "price": 50},
                }
            ],
            "total": 1,
            "query_time_ms": 15.5,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # Search with dict filters
        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters={
                "metadata": {
                    "operator": "and",
                    "filters": [
                        {"field": "category", "operator": "eq", "value": "tech"}
                    ],
                }
            },
        )

        assert result.total == 1
        assert len(result.results) == 1
        assert result.results[0].text == "Python tutorial"
        assert result.results[0].score == 0.95

        # Verify POST was called
        mock_client.post.assert_called_once()

    def test_search_with_searchfilters_object(self, sdk_client, mock_client):
        """Test search with SearchFilters Pydantic object."""
        from my_vector_db.domain.models import (
            SearchFilters,
            FilterGroup,
            MetadataFilter,
            FilterOperator,
            LogicalOperator,
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "Advanced Python",
                    "score": 0.92,
                    "metadata": {"category": "tech", "level": "advanced"},
                }
            ],
            "total": 1,
            "query_time_ms": 12.3,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # Create SearchFilters object
        filters = SearchFilters(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    MetadataFilter(
                        field="category", operator=FilterOperator.EQUALS, value="tech"
                    ),
                    MetadataFilter(
                        field="level", operator=FilterOperator.EQUALS, value="advanced"
                    ),
                ],
            )
        )

        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=5,
            filters=filters,
        )

        assert result.total == 1
        assert result.results[0].metadata["level"] == "advanced"

    def test_search_with_time_based_filters(self, sdk_client, mock_client):
        """Test search with time-based filters."""

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [],
            "total": 0,
            "query_time_ms": 8.5,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters={
                "created_after": "2024-01-01T00:00:00Z",
                "created_before": "2024-12-31T23:59:59Z",
            },
        )

        assert result.total == 0
        mock_client.post.assert_called_once()

    def test_search_with_document_ids_filter(self, sdk_client, mock_client):
        """Test search with document_ids filter."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "From specific doc",
                    "score": 0.88,
                    "metadata": {},
                }
            ],
            "total": 1,
            "query_time_ms": 10.2,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters={
                "document_ids": [
                    "00000000-0000-0000-0000-000000000002",
                    "00000000-0000-0000-0000-000000000004",
                ]
            },
        )

        assert result.total == 1
        assert (
            str(result.results[0].document_id) == "00000000-0000-0000-0000-000000000002"
        )

    def test_search_with_complex_nested_filters(self, sdk_client, mock_client):
        """Test search with complex nested AND/OR filters."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "Premium content",
                    "score": 0.94,
                    "metadata": {"category": "tech", "price": 75, "premium": True},
                }
            ],
            "total": 1,
            "query_time_ms": 18.7,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # (category == "tech" AND price < 100) OR premium == True
        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters={
                "metadata": {
                    "operator": "or",
                    "filters": [
                        {
                            "operator": "and",
                            "filters": [
                                {
                                    "field": "category",
                                    "operator": "eq",
                                    "value": "tech",
                                },
                                {"field": "price", "operator": "lt", "value": 100},
                            ],
                        },
                        {"field": "premium", "operator": "eq", "value": True},
                    ],
                }
            },
        )

        assert result.total == 1
        assert result.results[0].metadata["premium"] is True

    # ========================================================================
    # Custom Filter Function Tests (SDK Only)
    # ========================================================================

    def test_search_with_custom_filter_lambda(self, sdk_client, mock_client):
        """Test search with custom filter lambda function."""
        from my_vector_db.domain.models import SearchFiltersWithCallable

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "High score content",
                    "score": 0.97,
                    "metadata": {"quality_score": 85},
                }
            ],
            "total": 1,
            "query_time_ms": 14.2,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # Custom filter with lambda
        filters = SearchFiltersWithCallable(
            custom_filter=lambda chunk: chunk.metadata.get("quality_score", 0) > 50
        )

        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters=filters,
        )

        assert result.total == 1
        assert result.results[0].metadata["quality_score"] == 85

    def test_search_with_custom_filter_complex_function(self, sdk_client, mock_client):
        """Test search with complex custom filter function."""
        from my_vector_db.domain.models import SearchFiltersWithCallable

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "Quality content with high engagement",
                    "score": 0.93,
                    "metadata": {
                        "views": 5000,
                        "rating": 4.8,
                        "verified": True,
                    },
                }
            ],
            "total": 1,
            "query_time_ms": 16.8,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # Complex scoring function
        def quality_filter(chunk):
            """Calculate quality score from multiple factors."""
            metadata = chunk.metadata
            score = 0
            score += metadata.get("rating", 0) * 10
            score += metadata.get("views", 0) / 100
            score += 20 if metadata.get("verified") else 0
            return score >= 70

        filters = SearchFiltersWithCallable(custom_filter=quality_filter)

        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters=filters,
        )

        assert result.total == 1
        assert result.results[0].metadata["verified"] is True

    def test_search_custom_filter_takes_precedence(self, sdk_client, mock_client):
        """Test that custom_filter takes precedence over declarative filters."""
        from my_vector_db.domain.models import (
            SearchFiltersWithCallable,
            FilterGroup,
            MetadataFilter,
            FilterOperator,
            LogicalOperator,
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "Custom filtered",
                    "score": 0.89,
                    "metadata": {"category": "sports"},  # Would fail declarative filter
                }
            ],
            "total": 1,
            "query_time_ms": 11.5,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # Create filters with BOTH custom and declarative
        # Custom should take precedence
        filters = SearchFiltersWithCallable(
            metadata=FilterGroup(
                operator=LogicalOperator.AND,
                filters=[
                    MetadataFilter(
                        field="category",
                        operator=FilterOperator.EQUALS,
                        value="tech",  # This would NOT match
                    )
                ],
            ),
            custom_filter=lambda chunk: True,  # This always passes
        )

        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters=filters,
        )

        # Should get results because custom_filter returns True
        assert result.total == 1
        # Verify it's the "sports" category (would fail declarative filter)
        assert result.results[0].metadata["category"] == "sports"

    # ========================================================================
    # Filter Validation Tests
    # ========================================================================

    def test_search_with_invalid_filter_dict(self, sdk_client, mock_client):
        """Test that invalid filter dict raises ValidationError."""
        with pytest.raises(ValueError):
            # Invalid operator
            sdk_client.search(
                library_id="00000000-0000-0000-0000-000000000003",
                embedding=[0.1, 0.2, 0.3],
                k=10,
                filters={
                    "metadata": {
                        "operator": "INVALID_OPERATOR",  # Invalid
                        "filters": [
                            {"field": "category", "operator": "eq", "value": "tech"}
                        ],
                    }
                },
            )

        # No HTTP request should be made (validation fails before)
        mock_client.post.assert_not_called()

    def test_search_with_empty_filter_group(self, sdk_client, mock_client):
        """Test that empty filter group raises ValidationError."""
        with pytest.raises(ValueError, match="filters list cannot be empty"):
            sdk_client.search(
                library_id="00000000-0000-0000-0000-000000000003",
                embedding=[0.1, 0.2, 0.3],
                k=10,
                filters={"metadata": {"operator": "and", "filters": []}},  # Empty!
            )

        mock_client.post.assert_not_called()

    def test_search_filter_operator_value_mismatch(self, sdk_client, mock_client):
        """Test that IN operator with non-list value raises ValidationError."""
        with pytest.raises(ValueError, match="requires a list value"):
            sdk_client.search(
                library_id="00000000-0000-0000-0000-000000000003",
                embedding=[0.1, 0.2, 0.3],
                k=10,
                filters={
                    "metadata": {
                        "operator": "and",
                        "filters": [
                            {
                                "field": "category",
                                "operator": "in",
                                "value": "tech",  # Should be list!
                            }
                        ],
                    }
                },
            )

        mock_client.post.assert_not_called()

    # ========================================================================
    # Edge Cases and Error Handling
    # ========================================================================

    def test_search_with_none_filters(self, sdk_client, mock_client):
        """Test search with no filters (None)."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "00000000-0000-0000-0000-000000000002",
                    "text": "Unfiltered result",
                    "score": 0.91,
                    "metadata": {},
                }
            ],
            "total": 1,
            "query_time_ms": 9.8,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters=None,  # No filters
        )

        assert result.total == 1
        mock_client.post.assert_called_once()

    def test_search_dict_conversion_to_searchfilters(self, sdk_client, mock_client):
        """Test that dict is properly converted to SearchFilters."""

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [],
            "total": 0,
            "query_time_ms": 7.3,
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        # Pass dict - SDK should convert to SearchFilters
        result = sdk_client.search(
            library_id="00000000-0000-0000-0000-000000000003",
            embedding=[0.1, 0.2, 0.3],
            k=10,
            filters={"created_after": "2024-01-01T00:00:00Z"},
        )

        assert result.total == 0
        # Verify the request was made
        mock_client.post.assert_called_once()

        # Verify the call had the correct structure
        call_args = mock_client.post.call_args
        request_body = call_args.kwargs["json"]

        # Should have filters field with SearchFilters data
        assert "filters" in request_body
        # custom_filter should be excluded (not serializable)
        assert "custom_filter" not in str(request_body["filters"])
