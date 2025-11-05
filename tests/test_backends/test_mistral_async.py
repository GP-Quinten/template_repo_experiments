import asyncio
import pytest
from unittest.mock import Mock

# Dummy response class to mimic the API response object.
class DummyResponse:
    def model_dump(self):
        return {"result": "ok", "choices": ["dummy-choice"]}

from llm_inference.backends.mistral_async import MistralAsyncBackend

@pytest.mark.asyncio
async def test_infer_one(model_config, mistral_fake_response):
    api_key = "dummy-key"

    # Instantiate the backend.
    backend = MistralAsyncBackend(api_key=api_key)

    # Mock the client's chat.complete method.
    backend.client.chat.complete = Mock(return_value=mistral_fake_response)

    prompt = "Hello, world!"
    result = await backend.infer_one(prompt, model_config=model_config)

    # Verify that the mock was called and the result is as expected.
    backend.client.chat.complete.assert_called_once()
    assert "choices" in result

@pytest.mark.asyncio
async def test_infer_batch(model_config, mistral_fake_response):
    api_key = "dummy-key"

    backend = MistralAsyncBackend(api_key=api_key)
    backend.client.chat.complete = Mock(return_value=mistral_fake_response)

    # Updated to use a list of dictionaries with keys 'custom_id' and 'prompt'.
    prompts = [
        {"custom_id": 1, "prompt": "Prompt 1"},
        {"custom_id": 2, "prompt": "Prompt 2"},
        {"custom_id": 3, "prompt": "Prompt 3"},
    ]
    results = []
    async for result in backend.infer_many(prompts, model_config=model_config):
        results.append(result)

    # Assert that the API was called for each prompt.
    assert backend.client.chat.complete.call_count == len(prompts)
    for res in results:
        assert "choices" in res

@pytest.mark.asyncio
async def test_infer_many_custom_ids(model_config, mistral_fake_response):
    """
    Test to ensure that each response from infer_many includes the corresponding custom_id.
    """
    api_key = "dummy-key"
    backend = MistralAsyncBackend(api_key=api_key)
    backend.client.chat.complete = Mock(return_value=mistral_fake_response)

    # Define prompts with distinct custom_ids.
    prompts = [
        {"custom_id": "a", "prompt": "Test prompt A"},
        {"custom_id": "b", "prompt": "Test prompt B"},
        {"custom_id": "c", "prompt": "Test prompt C"},
    ]
    results = []
    async for result in backend.infer_many(prompts, model_config=model_config):
        results.append(result)

    # Verify that each result has the expected custom_id.
    custom_ids_returned = {result["custom_id"] for result in results}
    expected_custom_ids = {"a", "b", "c"}
    assert custom_ids_returned == expected_custom_ids

@pytest.mark.asyncio
async def test_infer_many_out_of_order():
    """
    Test that infer_many returns results in the order tasks complete,
    which may be different from the order of the input prompts.
    """
    api_key = "dummy-key"
    backend = MistralAsyncBackend(api_key=api_key)

    # Override infer_one to simulate different processing times.
    async def delayed_infer_one(prompt, model_config, use_cache):
        if prompt == "Test prompt A":
            await asyncio.sleep(0.3)  # This will finish last.
            return {"choices": ["result A"]}
        elif prompt == "Test prompt B":
            await asyncio.sleep(0.1)  # This will finish first.
            return {"choices": ["result B"]}
        elif prompt == "Test prompt C":
            await asyncio.sleep(0.2)  # This will finish second.
            return {"choices": ["result C"]}
        else:
            return {"choices": ["default"]}
    
    # Monkey-patch the infer_one method.
    backend.infer_one = delayed_infer_one

    # Define prompts with custom IDs.
    prompts = [
        {"custom_id": "a", "prompt": "Test prompt A"},
        {"custom_id": "b", "prompt": "Test prompt B"},
        {"custom_id": "c", "prompt": "Test prompt C"},
    ]

    results = []
    async for result in backend.infer_many(prompts, model_config={}):
        results.append(result)

    # The expected order based on our delays is:
    # "Test prompt B" -> custom_id "b"
    # "Test prompt C" -> custom_id "c"
    # "Test prompt A" -> custom_id "a"
    expected_order = ["b", "c", "a"]
    actual_order = [result["custom_id"] for result in results]

    # Assert that the order is as expected (i.e., not the same as the input order).
    assert actual_order == expected_order
