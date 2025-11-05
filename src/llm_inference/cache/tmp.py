import os
import json
import tempfile
from typing import Any

from llm_inference.cache.base import AbstractCacheStorage

class TmpCacheStorage(AbstractCacheStorage):
    """
    Temporary file-based cache storage implementation.
    Uses a temporary directory to store cached values as JSON files.
    """

    def __init__(self):
        """
        Initializes a temporary directory for caching.
        """
        # Initialize logger if not provided by the base class.
        self._temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = self._temp_dir.name

    def _cache_path(self, hashed_key: str) -> str:
        """
        Generate the file path for a given hashed key in the temporary directory.
        """
        return os.path.join(self.cache_dir, f"{hashed_key}.cache")

    def get(self, key: Any):
        """
        Retrieve a cached value using the given key.
        """
        hashed_key = self._get_hashed_key(key)
        self.logger.debug(f"Attempting to retrieve cache for key: {hashed_key}")
        file_path = self._cache_path(hashed_key)
        if not os.path.exists(file_path):
            self.logger.debug(f"Cache file not found for key: {hashed_key}")
            return None
        try:
            with open(file_path, "r") as f:
                value = json.load(f)
            self.logger.info(f"Cache hit for key: {hashed_key}")
            return value
        except Exception as e:
            self.logger.error(f"Error retrieving cache for key {hashed_key}: {e}")
            raise e

    def put(self, key: Any, value):
        """
        Store a value in the cache with the given key.
        """
        hashed_key = self._get_hashed_key(key)
        self.logger.debug(f"Storing cache for key: {hashed_key}")
        file_path = self._cache_path(hashed_key)
        try:
            with open(file_path, "w") as f:
                json.dump(value, f)
            self.logger.debug(f"Cache stored successfully for key: {hashed_key}")
        except Exception as e:
            self.logger.error(f"Error storing cache for key {hashed_key}: {e}")
            raise e
