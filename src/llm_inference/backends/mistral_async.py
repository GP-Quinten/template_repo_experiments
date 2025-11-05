from typing import Optional

import asyncio
from mistralai import Mistral

from llm_inference.backends.base_async import RateLimiter
from llm_inference.cache.base import AbstractCacheStorage
from llm_inference.backends.mistral_base import MistralAsyncBaseBackend


class MistralAsyncBackend(MistralAsyncBaseBackend):
    """
    Asynchronous backend implementation using the Mistral API for inference.
    """
    def __init__(self, api_key: str, cache_storage: Optional["AbstractCacheStorage"] = None):
        """
        Initializes the AsyncMistralBackend with API key and optional cache storage.
        
        Args:
            api_key (str): API key for authenticating with the Mistral API.
            cache_storage (AbstractCacheStorage, optional): A cache storage implementation.
        """
        super().__init__(cache_storage)
        self.client = Mistral(api_key=api_key)
        self.rate_limiter = RateLimiter(rate=6, per=1.0)  # Adjust rate as needed.
        self.max_retries = 5

    async def _call_api(self, prompt: str, model_config: dict) -> dict:
        """
        Implements the API-specific asynchronous call to the Mistral API.
        
        Args:
            prompt (str): The input prompt.
            model_config (dict): A dictionary containing model parameters and settings.
        
        Returns:
            dict: The API response.
        """
        messages = [{
            "role": "user",
            "content": prompt,
        }]

        # Run the blocking API call in a separate thread.
        response = await asyncio.to_thread(
            self.client.chat.complete,
            model=model_config["model"],
            messages=messages,
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            random_seed=model_config["random_seed"],
            response_format=model_config["response_format"],
            n=model_config.get("n", 1),
        )
        return response.model_dump()
