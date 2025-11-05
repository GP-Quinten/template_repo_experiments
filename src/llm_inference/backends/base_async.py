import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, AsyncGenerator, Optional

from mistralai.models.sdkerror import SDKError  # Adjust the import as needed
from llm_inference.backends.base import BaseBackend  # Provided base class (synchronous)
from llm_inference.cache.base import AbstractCacheStorage
from llm_inference.logger_mixin import LoggingMixin

# Define a simple asynchronous rate limiter.
class RateLimiter(LoggingMixin):
    """A simple token bucket rate limiter."""
    def __init__(self, rate: int, per: float):
        """
        Args:
            rate (int): Maximum number of tokens (calls) allowed per interval.
            per (float): Interval duration in seconds.
        """
        self._rate = rate
        self._per = per
        self._tokens = rate
        self._lock = asyncio.Lock()
        self._last = time.monotonic()

    async def acquire(self):
        """Acquire a token, waiting if necessary until one is available."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                # Refill tokens based on elapsed time.
                self._tokens = min(self._rate, self._tokens + elapsed * (self._rate / self._per))
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
            # Wait a bit before trying again.
            await asyncio.sleep(self._per / self._rate)

# Create an asynchronous base backend that reuses common logic.
class BaseAsyncBackend(BaseBackend, LoggingMixin, ABC):
    """
    Asynchronous version of BaseBackend that implements common logic
    such as caching, rate limiting, retry logic, and batch processing.
    """
    def __init__(self, cache_storage: Optional["AbstractCacheStorage"] = None):
        """
        Args:
            cache_storage (AbstractCacheStorage, optional): A cache storage implementation.
        """
        super().__init__(cache_storage)
        self.rate_limiter = None  # To be set by subclasses if needed.
        self.max_retries = 5      # Default maximum number of retries.

    @abstractmethod
    async def _call_api(self, prompt: str, model_config: dict) -> dict:
        """
        Abstract asynchronous method for performing the API call.
        Subclasses must implement API-specific logic here.

        Args:
            prompt (str): The input prompt.
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

    async def infer_one(self, prompt: str, model_config: dict, use_cache: bool = True) -> dict:
        """
        Performs asynchronous inference on a single prompt with caching,
        rate limiting, and retry logic.

        Args:
            prompt (str): The input prompt.
            model_config (dict): A dictionary containing model parameters and settings.
            use_cache (bool): Whether to use caching (default is True).

        Returns:
            dict: The inference result.
        """
        # Check cache if enabled.
        if use_cache and self.cache_storage is not None:
            cached_response = await asyncio.to_thread(self.cache_storage.get, prompt)
            if cached_response is not None:
                return cached_response

        retry_delay = 1.0
        for attempt in range(self.max_retries):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                result = await self._call_api(prompt, model_config)

                # Cache the result if enabled.
                if use_cache and self.cache_storage is not None:
                    await asyncio.to_thread(self.cache_storage.put, prompt, result)

                return result

            except SDKError as e:
                # If the error indicates rate limiting, retry with exponential backoff.
                if "429" in str(e):
                    self.logger.error(
                        f"Rate limit exceeded. Attempt {attempt+1} of {self.max_retries}. "
                        f"Retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        raise RuntimeError("Maximum retries exceeded due to rate limiting.")

    async def infer_many(self, prompt_items: List[dict], model_config: dict, use_cache: bool = True) -> AsyncGenerator[dict, None]:
        """
        Performs asynchronous inference on a batch of prompt_items.
        Each result includes a 'custom_id' corresponding to the input's custom_id.

        Args:
            prompt_items (List[dict]): A list of dictionaries, each with keys 'custom_id' and 'prompt'.
            model_config (dict): A dictionary containing model parameters and settings.
            use_cache (bool): Whether to use caching (default is True).

        Yields:
            dict: Inference results for each prompt, with an added 'custom_id' key.
        """
        async def wrap(item: dict) -> dict:
            # Await the inference result and create a shallow copy to avoid shared mutation.
            result = dict(await self.infer_one(item['prompt'], model_config, use_cache=use_cache))
            result['custom_id'] = item['custom_id']
            return result

        tasks = [asyncio.create_task(wrap(item)) for item in prompt_items]
        for completed in asyncio.as_completed(tasks):
            yield await completed
