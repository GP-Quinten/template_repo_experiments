import pytest
from unittest.mock import MagicMock
from llm_inference.backends.mistral_sync import MistralBackend


@pytest.fixture
def mistral_backend(model_config, mistral_fake_response):
    backend = MistralBackend(
        api_key="fake-api-key"
    )
    # Override the client's chat.complete method to return our fake response.
    backend.client = MagicMock()
    backend.client.chat = MagicMock()
    backend.client.chat.complete = MagicMock(return_value=mistral_fake_response)
    return backend

def test_infer_one(mistral_backend, model_config, llm_raw_response):
    prompt = "Test prompt"
    result = mistral_backend.infer_one(prompt, model_config=model_config)
    # Verify that chat.complete was called with the expected parameters.
    mistral_backend.client.chat.complete.assert_called_once_with(
        model=model_config["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        random_seed=model_config["random_seed"],
        response_format=model_config["response_format"],
        n=1,
    )
    # Assert that the returned result has the expected content from llm_raw_response.
    assert result["id"] == llm_raw_response["id"]

def test_infer_batch(mistral_backend, model_config, llm_raw_response):
    prompts = [
        {"custom_id": 1, "prompt": "Prompt 1"},
        {"custom_id": 2, "prompt": "Prompt 2"},
    ]
    results = list(mistral_backend.infer_many(prompts, model_config))
    # Ensure we have one result per prompt.
    assert len(results) == 2
    # Both responses should match the fake llm_raw_response.
    for result in results:
        assert result["id"] == llm_raw_response["id"]
    # Verify that complete was called once per prompt.
    assert mistral_backend.client.chat.complete.call_count == 2

@pytest.fixture
def mistral_backend_with_cache(mistral_fake_response):
    from llm_inference.cache.tmp import TmpCacheStorage

    cache_storage = TmpCacheStorage()
    backend = MistralBackend(
        api_key="fake-api-key",
        cache_storage=cache_storage
    )
    # Override the client's chat.complete method to return our fake response.
    backend.client = MagicMock()
    backend.client.chat = MagicMock()
    backend.client.chat.complete = MagicMock(return_value=mistral_fake_response)
    return backend

def test_infer_one_with_cache(mistral_backend_with_cache, model_config, llm_raw_response):
    prompt = "Test prompt"
    result = mistral_backend_with_cache.infer_one(prompt, model_config=model_config)
    # Verify that chat.complete was called with the expected parameters.
    result_from_cache = mistral_backend_with_cache.cache_storage.get(prompt)

    assert result == result_from_cache
