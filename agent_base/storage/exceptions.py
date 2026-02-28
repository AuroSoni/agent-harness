"""Custom exceptions for storage adapters.

These exceptions provide a consistent error interface across all adapter implementations.
"""


class StorageError(Exception):
    """Base exception for all storage-related errors."""
    pass


class StorageConnectionError(StorageError):
    """Failed to connect to storage backend."""
    pass


class StorageNotFoundError(StorageError):
    """Requested resource was not found."""
    
    def __init__(self, resource_type: str, key: str):
        self.resource_type = resource_type
        self.key = key
        super().__init__(f"{resource_type} not found: {key}")


class StorageValidationError(StorageError):
    """Data validation failed."""
    pass


class StorageOperationError(StorageError):
    """A storage operation failed."""
    pass
