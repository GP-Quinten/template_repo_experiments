# LLM Backends

## MistralBackend

`MistralBackend` is a synchronous backend implementation using the Mistral API for inference. It performs inference calls to the Mistral API and processes the responses synchronously.

### Example Usage

```python
from llm_inference.backends.mistral_sync import MistralBackend
from llm_inference.cache.disk import DiskCacheStorage

api_key = "your_api_key"
cache_storage = DiskCacheStorage(subdir="mistral_cache")
backend = MistralBackend(api_key=api_key, cache_storage=cache_storage)

# Perform inference
response = backend.infer_one(prompt, model_config)
print(response)
```

## MistralAsyncBackend

`MistralAsyncBackend` is an asynchronous backend implementation using the Mistral API for inference. It performs inference calls to the Mistral API asynchronously, allowing for non-blocking operations and better handling of concurrent requests.

### Example Usage

```python
import asyncio
from llm_inference.backends.mistral_async import MistralAsyncBackend
from llm_inference.cache.tmp import TmpCacheStorage

api_key = "your_api_key"
cache_storage = TmpCacheStorage()
backend = MistralAsyncBackend(api_key=api_key, cache_storage=cache_storage)

async def main():
    # Perform inference
    response = await backend.infer_one(prompt, model_config)
    print(response)

asyncio.run(main())
```

## MistralBatchBackend

`MistralBatchBackend` is a backend implementation designed for batch inference using the Mistral API. It allows for performing inference on multiple prompts in a single batch, optimizing the process for large-scale inference tasks.

### Example Usage

```python
from llm_inference.backends.mistral_batch import MistralBatchBackend
from llm_inference.cache.disk import DiskCacheStorage

api_key = "your_api_key"
cache_storage = DiskCacheStorage(subdir="mistral_batch_cache")
backend = MistralBatchBackend(api_key=api_key, cache_storage=cache_storage)

# Perform batch inference
results = backend.infer_many(prompts, model_config)
for result in results:
    print(result)
```

# Cache Storages

## DiskCacheStorage

`DiskCacheStorage` is a disk-based cache storage implementation. It stores cached values as JSON files on disk, providing a persistent caching mechanism that survives application restarts.

### Example Usage

```python
from llm_inference.cache.disk import DiskCacheStorage

cache_storage = DiskCacheStorage(subdir="my_cache")
key = "example_key"
value = {"data": "example_value"}

# Store value in cache
cache_storage.put(key, value)

# Retrieve value from cache
cached_value = cache_storage.get(key)
print(cached_value)
```

## TmpCacheStorage

`TmpCacheStorage` is a temporary file-based cache storage implementation. It uses a temporary directory to store cached values as JSON files, providing a non-persistent caching mechanism that is cleared when the temporary directory is deleted.

### Example Usage

```python
from llm_inference.cache.tmp import TmpCacheStorage

cache_storage = TmpCacheStorage()
key = "example_key"
value = {"data": "example_value"}

# Store value in cache
cache_storage.put(key, value)

# Retrieve value from cache
cached_value = cache_storage.get(key)
print(cached_value)
```
