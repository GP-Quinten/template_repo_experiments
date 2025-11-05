import os
import json
from typing import Any

from llm_inference.cache.base import AbstractCacheStorage
from llm_inference.settings import CACHE_DIR

class DiskCacheStorage(AbstractCacheStorage):
    """
    Disk-based cache storage implementation.
    Stores cached values in JSON files on disk.
    """

    def __init__(self, subdir: str=None):
        """
        Initialize the DiskCacheStorage with a given directory for cache files.
        """
        if subdir:
            self.cache_dir = os.path.join(CACHE_DIR, subdir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, hashed_key: str) -> str:
        """
        Generate the full file path for a given hashed key.
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
