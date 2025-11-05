import json
import pytest



@pytest.fixture
def llm_parsed_response():
    return {
        "criterion_1": {
            "justification": "The text is written in English",
            "answer": "YES"
        },
        "criterion_2": {
            "justification": "There is no information provided to determine if this is a primary study",
            "answer": "UNCERTAIN",
        },
    }


@pytest.fixture
def llm_raw_response(llm_parsed_response):
    return {
        "custom_id": "doc1",
        "id": "ce5d302de2204802a2a31f98cac5f88a",
        "object": "chat.completion",
        "model": "mistral-large-latest",
        "usage": {"prompt_tokens": 437, "completion_tokens": 82, "total_tokens": 519},
        "created": 1740663485,
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": json.dumps(llm_parsed_response, indent=4),
                    "tool_calls": None,
                    "prefix": False,
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }
        ],
    }

@pytest.fixture
def model_config():
    return {
        "model": "mistral-large-latest",
        "temperature": 0.7,
        "max_tokens": 5,
        "random_seed": 42,
        "response_format": {"type": "json_object"},
    }

class FakeResponseFromRaw:
    def __init__(self, raw_response):
        self._raw_response = raw_response

    def model_dump(self):
        return self._raw_response
    
@pytest.fixture
def mistral_fake_response(llm_raw_response):
    return FakeResponseFromRaw(llm_raw_response)



@pytest.fixture
def dummy_mistral_async_backend():
    from llm_inference.backends.mistral_base import MistralAsyncBaseBackend

    class DummyMistralAsyncBackend(MistralAsyncBaseBackend):
        def __init__(self, *args, **kwargs):
            pass

        async def _call_api(self, prompt, model_config):
            pass
        
    return DummyMistralAsyncBackend(api_key="dummy")
