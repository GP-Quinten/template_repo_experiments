from abc import ABC, abstractmethod
import hashlib
from typing import Any

from llm_inference.logger_mixin import LoggingMixin

class AbstractCacheStorage(ABC, LoggingMixin):
    """
    Abstract base class for cache storage implementations.
    This class defines the interface for caching operations using a hashable key.
    It also provides helper methods to generate a consistent hash for any given key.
    """

    def _generate_hash(self, data: Any) -> str:
        """
        Generate a SHA-256 hash for the given data.
        
        Args:
            data (Any): A hashable object.
            
        Returns:
            str: The SHA-256 hash of the string representation of the data.
        """
        data_str = str(data)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _get_hashed_key(self, key: Any) -> str:
        """
        Convert a hashable key into its corresponding hash string.
        
        Args:
            key (Any): A hashable object to be used as a key.
            
        Returns:
            str: The hashed key.
        """
        return self._generate_hash(key)

    @abstractmethod
    def get(self, key: Any):
        """
        Retrieve a cached value using the given key.
        
        Args:
            key (Any): A hashable object used as the key.
            
        Returns:
            The cached value if it exists, or None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def put(self, key: Any, value):
        """
        Store a value in the cache with the given key.
        
        Args:
            key (Any): A hashable object used as the key.
            value: The value to cache.
        """
        raise NotImplementedError
