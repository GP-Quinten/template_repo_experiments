import json
from typing import Dict
from abc import ABC, abstractmethod

from llm_inference.backends.base import BaseBackend
from llm_inference.backends.base_async import BaseAsyncBackend

class MistralBaseBackend(BaseBackend, ABC):
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

    def _parse_response(self, backend_response: dict) -> Dict[str, str]:
        """Parses the backend response to extract validation information.

        Args:
            backend_response (dict): The response dictionary from the backend inference.

        Returns:
            Dict[str, str]: A dictionary containing:
        """
        content = backend_response["choices"][0]["message"]["content"]
        parsed_content = json.loads(content)
        return parsed_content


class MistralAsyncBaseBackend(MistralBaseBackend, BaseAsyncBackend, ABC):
    @abstractmethod
    async def _call_api(self, prompt: str, model_config: dict) -> dict:
        """
        Abstract method for performing the API call asynchronously.
        Must be implemented by subclasses with API-specific logic.

        Args:
            prompt (str): The input prompt for the API call.
            model_config (dict): A dictionary containing model parameters and settings.

        Returns:
            dict: The API response.
        """
        pass

class MistralBatchBaseBackend(MistralBaseBackend):
    def _call_api(self, prompt, model_config):
        raise NotImplementedError("_call_api inference is not supported by this backend.")

