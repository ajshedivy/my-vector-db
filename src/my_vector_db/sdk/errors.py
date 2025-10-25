"""
HTTP error handling utilities for the SDK.

This module provides error handling for HTTP requests via a decorator pattern.
The decorator handles all httpx errors and converts them to SDK exceptions,
keeping client method bodies clean and focused on business logic.

Following principles:
- DRY: Single decorator handles all error cases
- Composition: Applied to private helper methods, not public API
- Reusable for both sync and async clients
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict

import httpx
from httpx import codes

from my_vector_db.sdk.exceptions import (
    ServerConnectionError,
    NotFoundError,
    ServerError,
    TimeoutError,
    ValidationError,
    VectorDBError,
)


def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle all HTTP errors and convert to SDK exceptions.

    This decorator wraps HTTP request methods to:
    1. Execute the request
    2. Handle HTTP status errors (400, 404, 500, etc.)
    3. Handle network errors (connection, timeout, etc.)
    4. Parse JSON response on success
    5. Return empty dict for 204 No Content

    The decorated function should return an httpx.Response object.
    The decorator returns a parsed JSON dict or raises an SDK exception.

    Args:
        func: Function that returns httpx.Response

    Returns:
        Wrapped function that returns dict or raises SDK exception

    Example:
        >>> @handle_errors
        >>> def _get(self, path: str, **kwargs) -> httpx.Response:
        >>>     return self._client.get(f"{self.base_url}{path}", **kwargs)
    """

    @wraps(func)
    def wrapper(self: Any, path: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            # Execute the HTTP request (returns httpx.Response)
            response = func(self, path, **kwargs)

            # Check for HTTP errors and handle them
            try:
                response.raise_for_status()

                # Handle empty responses (e.g., 204 NO CONTENT)
                if response.status_code == codes.NO_CONTENT or not response.content:
                    return {}

                # Parse and return JSON
                return response.json()

            except httpx.HTTPStatusError as e:
                # Map HTTP status codes to SDK exceptions
                raise map_http_error(e) from e

        except httpx.ConnectError as e:
            # Connection refused, network down, DNS failure
            raise ServerConnectionError(
                f"Cannot connect to VectorDB at {self.base_url}. "
                "Ensure the server is running."
            ) from e

        except httpx.TimeoutException as e:
            # Request took too long
            raise TimeoutError(
                f"Request timed out after {self.timeout}s. "
                "The server may be overloaded or unreachable."
            ) from e

        except httpx.RequestError as e:
            # Other request errors (SSL, malformed URLs, etc.)
            raise VectorDBError(f"Request failed: {str(e)}") from e

    return wrapper


def handle_response(response: httpx.Response) -> Dict[str, Any]:
    """
    Handle HTTP response and convert errors to SDK exceptions.

    This function centralizes all error handling logic. It checks the response
    status code, raises appropriate SDK exceptions for errors, and returns
    parsed JSON for successful responses.

    Args:
        response: httpx Response object

    Returns:
        Parsed JSON response as dictionary

    Raises:
        ValidationError: If request validation fails (400)
        NotFoundError: If resource not found (404)
        ServerError: If server error occurs (500+)
        VectorDBError: For other HTTP errors

    Example:
        >>> response = client.get("/api/libraries")
        >>> data = handle_response(response)
        >>> libraries = [Library(**lib) for lib in data]
    """
    try:
        # Raise for HTTP errors (4xx, 5xx)
        response.raise_for_status()

        # Handle empty responses (e.g., 204 NO CONTENT)
        if response.status_code == codes.NO_CONTENT or not response.content:
            return {}

        # Parse and return JSON
        return response.json()

    except httpx.HTTPStatusError as e:
        # Map HTTP status codes to SDK exceptions
        raise map_http_error(e) from e


def map_http_error(error: httpx.HTTPStatusError) -> VectorDBError:
    """
    Map httpx HTTPStatusError to appropriate SDK exception.

    Args:
        error: httpx HTTPStatusError with response

    Returns:
        Appropriate SDK exception instance

    Example:
        >>> try:
        ...     response.raise_for_status()
        ... except httpx.HTTPStatusError as e:
        ...     raise map_http_error(e) from e
    """
    # Extract error message from API response
    try:
        error_data = error.response.json()
        message = error_data.get("detail", str(error))
    except Exception:
        message = str(error)

    status_code = error.response.status_code

    # Map status codes to SDK exceptions
    if status_code == codes.BAD_REQUEST:
        # Invalid input (e.g., querying empty library, validation failure)
        return ValidationError(f"Invalid request: {message}")

    if status_code == codes.UNPROCESSABLE_ENTITY:
        # Pydantic validation errors (FastAPI automatic validation)
        return ValidationError(f"Validation error: {message}")

    if status_code == codes.NOT_FOUND:
        # Resource not found - provide helpful context
        return NotFoundError(
            f"{message}. Use the appropriate list method to see available resources."
        )

    if codes.INTERNAL_SERVER_ERROR <= status_code < 600:
        # Server errors
        return ServerError(f"Server error: {message}", status_code=status_code)

    # Unexpected status codes
    return VectorDBError(
        f"API error ({status_code}): {message}", status_code=status_code
    )


def handle_network_error(error: Exception, base_url: str) -> VectorDBError:
    """
    Map network-level errors to SDK exceptions.

    Handles connection failures, timeouts, and other request errors that
    occur before receiving an HTTP response.

    Args:
        error: Exception from httpx request
        base_url: Base URL being connected to (for error messages)

    Returns:
        Appropriate SDK exception instance

    Example:
        >>> try:
        ...     response = client.get(url)
        ... except httpx.ConnectError as e:
        ...     raise handle_network_error(e, base_url) from e
    """
    if isinstance(error, httpx.ConnectError):
        # Connection refused, network down, DNS failure
        return ServerConnectionError(
            f"Cannot connect to VectorDB at {base_url}. Ensure the server is running."
        )

    if isinstance(error, httpx.TimeoutException):
        # Request took too long
        return TimeoutError(
            "Request timed out. The server may be overloaded or unreachable."
        )

    if isinstance(error, httpx.RequestError):
        # Other request errors (SSL, malformed URLs, etc.)
        return VectorDBError(f"Request failed: {str(error)}")

    # Unexpected error type
    return VectorDBError(f"Unexpected error: {str(error)}")
