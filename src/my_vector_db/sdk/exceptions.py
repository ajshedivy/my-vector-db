"""
Custom exceptions for the Vector Database SDK.

These exceptions map HTTP status codes to Python exceptions for better error handling.
"""

from __future__ import annotations


class VectorDBError(Exception):
    """Base exception for all Vector DB SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(VectorDBError):
    """Raised when request validation fails (400)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class NotFoundError(VectorDBError):
    """Raised when a resource is not found (404)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


class ServerError(VectorDBError):
    """Raised when server returns 5xx error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message, status_code=status_code)


class ServerConnectionError(VectorDBError):
    """Raised when connection to the API fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)


class TimeoutError(VectorDBError):
    """Raised when a request times out."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=None)
