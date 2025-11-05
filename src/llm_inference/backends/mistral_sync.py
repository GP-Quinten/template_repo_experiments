from typing import Optional

from mistralai import Mistral

from llm_inference.cache.base import AbstractCacheStorage
from llm_inference.backends.mistral_base import MistralBaseBackend


class MistralBackend(MistralBaseBackend):
    """Backend implementation using the Mistral API for inference."""

    def __init__(self, api_key: str, cache_storage: Optional["AbstractCacheStorage"] = None):
        super().__init__(cache_storage)
        self.client = Mistral(api_key=api_key)

    def _call_api(self, prompt: str, model_config: dict) -> dict:
        """
        Implements the API-specific logic for performing an inference call to the Mistral API.

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

        response = self.client.chat.complete(
            model=model_config["model"],
            messages=messages,
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            random_seed=model_config["random_seed"],
            response_format=model_config["response_format"],
            n=model_config.get("n", 1),
        )
        return response.model_dump()
