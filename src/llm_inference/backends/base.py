import json
from abc import ABC, abstractmethod
from typing import List, Generator, Optional, Dict

from llm_inference.cache.base import AbstractCacheStorage
from llm_inference.logger_mixin import LoggingMixin

class BaseBackend(LoggingMixin, ABC):
    """Base backend class for model inference with common logic."""

    def __init__(self, cache_storage: Optional["AbstractCacheStorage"] = None):
        """
        Initializes the backend with an optional cache storage.

        Args:
            cache_storage (AbstractCacheStorage, optional): A cache storage implementation.
        """
        self.cache_storage = cache_storage

    @abstractmethod
    def _call_api(self, prompt: str, model_config: dict) -> dict:
        """
        Abstract method for performing the API call.
        Must be implemented by subclasses with API-specific logic.

        Args:
            prompt (str): The input prompt for the API call.
            model_config (dict): A dictionary containing model parameters and settings.

        Returns:
            dict: The API response.
        """
        pass

    @abstractmethod
    def _parse_response(self, response: dict) -> dict:
        """
        Abstract method for parsing the API response.
        Must be implemented by subclasses with API-specific logic.

        Args:
            response (dict): The raw API response.

        Returns:
            dict: The parsed response.
        """
        pass

    def infer_one(self, prompt: str, model_config: dict, use_cache: bool = True) -> dict:
        """
        Performs inference on a single prompt, optionally using cache.

        Args:
            prompt (str): The input prompt.
            model_config (dict): A dictionary containing model parameters and settings.
            use_cache (bool): Whether to use caching (default: True).

        Returns:
            dict: The inference result.
        """
        # Check cache first
        if use_cache and self.cache_storage is not None:
            cached_response = self.cache_storage.get(prompt)
            if cached_response is not None:
                return cached_response

        # Call the API with the provided model_config
        result = self._call_api(prompt, model_config)

        # Save to cache if enabled
        if use_cache and self.cache_storage is not None:
            self.cache_storage.put(prompt, result)

        return result

    def infer_many(self, prompt_items: List[dict], model_config: dict, use_cache: bool = True) -> Generator[dict, None, None]:
        """
        Performs inference on a list of prompt dictionaries, yielding each result one at a time.
        Each input dictionary must have keys 'custom_id' and 'prompt', and each output
        dictionary will include the corresponding 'custom_id'.

        Args:
            prompt_items (List[dict]): A list of dictionaries, each with keys 'custom_id' and 'prompt'.
            model_config (dict): A dictionary containing model parameters and settings.
            use_cache (bool): Whether to use caching (default: True).

        Yields:
            dict: The inference result for each prompt, augmented with a 'custom_id' key.
        """
        for item in prompt_items:
            result = self.infer_one(item["prompt"], model_config, use_cache=use_cache)
            # Create a copy in case the result is cached/shared so that custom_id modifications are isolated.
            result = dict(result)
            result["custom_id"] = item["custom_id"]
            yield result
