import json
import pytest

from llm_inference.backends.mistral_batch import MistralBatchBackend


# --- Fake Classes for Simulating the Mistral Client ---

class FakeFiles:
    def upload(self, file, purpose):
        # Simulate file upload and return a fake batch file with an id.
        class FakeBatchFile:
            id = "fake_batch_file_id"
        return FakeBatchFile()

    def download(self, file_id):
        # Simulate downloading a file by returning a fake file-like object.
        class FakeFileDownload:
            def read(self):
                # Simulate two JSONL entries.
                content = (
                    json.dumps({"custom_id": "1", "response": {"body": {"choice": "A"}}}) + "\n" +
                    json.dumps({"custom_id": "2", "response": {"body": {"choice": "B"}}})
                )
                return content.encode("utf-8")
        return FakeFileDownload()


class FakeJob:
    def __init__(self, job_id, status, output_file):
        self.id = job_id
        self.status = status
        self.output_file = output_file


class FakeBatchJobs:
    def create(self, input_files, model, endpoint, metadata):
        # Return a fake job that initially is in a RUNNING state.
        return FakeJob("fake_job_id", "RUNNING", "fake_output_file")

    def get(self, job_id):
        # Always return a job with SUCCESS status (for the purpose of testing).
        return FakeJob(job_id, "SUCCESS", "fake_output_file")


class FakeBatch:
    def __init__(self):
        self.jobs = FakeBatchJobs()


class FakeMistralClient:
    def __init__(self):
        self.files = FakeFiles()
        self.batch = FakeBatch()


# --- Fake Cache Storage for Testing Cache Functionality ---

class FakeCacheStorage:
    def __init__(self):
        self.storage = {}

    def _generate_hash(self, value):
        # For simplicity, always return the same hash.
        return "fakehash"

    def get(self, key):
        return self.storage.get(key)

    def put(self, key, value):
        self.storage[key] = value


# --- Test Functions ---

def test_make_batch_data_from_prompts():
    backend = MistralBatchBackend(api_key="dummy")
    model_config = {
        "max_tokens": 50,
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
        "random_seed": 123,
        "model": "test-model",
        "n": 1,
    }
    prompts = [
        "Simple prompt",
        {"prompt": "Dict prompt", "custom_id": "custom1"},
    ]
    batch_data = backend._make_batch_data_from_prompts(prompts, model_config)
    # Validate that batch_data is a list with two items.
    assert isinstance(batch_data, list)
    assert len(batch_data) == 2

    # Check the first prompt (a simple string prompt).
    first = batch_data[0]
    assert first["custom_id"] == "0"
    assert first["body"]["messages"][0]["content"] == "Simple prompt"

    # Check the second prompt (a dictionary with a custom_id).
    second = batch_data[1]
    assert second["custom_id"] == "custom1"
    assert second["body"]["messages"][0]["content"] == "Dict prompt"


def test_postprocess():
    backend = MistralBatchBackend(api_key="dummy")
    raw_results = [
        {"custom_id": "1", "response": {"body": {"choice": "A"}}},
        {"custom_id": "2", "response": {"body": {"choice": "B"}}},
    ]
    processed = list(backend.postprocess(raw_results))
    expected = [
        {"custom_id": "1", "choice": "A"},
        {"custom_id": "2", "choice": "B"},
    ]
    assert processed == expected


def test_infer_many_without_cache():
    backend = MistralBatchBackend(api_key="dummy")
    # Inject the fake client into the backend.
    backend.client = FakeMistralClient()
    model_config = {
        "max_tokens": 50,
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
        "random_seed": 123,
        "model": "test-model",
        "n": 1,
    }
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    # Since use_cache=False, the job should be executed.
    results = list(backend.infer_many(prompts, model_config=model_config, use_cache=False))
    # Our FakeFiles.download returns 2 results irrespective of the number of prompts.
    expected = [
        {"custom_id": "1", "choice": "A"},
        {"custom_id": "2", "choice": "B"},
    ]
    assert results == expected


def test_infer_many_with_cache():
    cache_storage = FakeCacheStorage()
    backend = MistralBatchBackend(api_key="dummy", cache_storage=cache_storage)
    backend.client = FakeMistralClient()
    model_config = {
        "max_tokens": 50,
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
        "random_seed": 123,
        "model": "test-model",
        "n": 1,
    }
    prompts = ["Prompt 1", "Prompt 2"]

    # First call: no cached result exists so the batch job will execute.
    results_first = list(backend.infer_many(prompts, model_config=model_config, use_cache=True))
    expected = [
        {"custom_id": "1", "choice": "A"},
        {"custom_id": "2", "choice": "B"},
    ]
    assert results_first == expected

    # Now, simulate that the cache is used by overriding _upload_batch_file.
    def fake_upload_batch_file(batch_data):
        raise Exception("upload should not be called when using cache")
    backend._upload_batch_file = fake_upload_batch_file

    # Second call: results should be returned from the cache.
    results_second = list(backend.infer_many(prompts, model_config=model_config, use_cache=True))
    assert results_second == expected


def test_execute_batch_job_failure(monkeypatch):
    backend = MistralBatchBackend(api_key="dummy")

    # Create fake classes to simulate a failure in the batch job.
    class FakeJobFailure:
        def __init__(self, job_id, status, output_file):
            self.id = job_id
            self.status = status
            self.output_file = output_file

    class FakeBatchJobsFailure:
        def create(self, input_files, model, endpoint, metadata):
            return FakeJobFailure("job_fail", "RUNNING", "fake_output_file")

        def get(self, job_id):
            return FakeJobFailure(job_id, "FAILED", "fake_output_file")

    class FakeBatchFailure:
        def __init__(self):
            self.jobs = FakeBatchJobsFailure()

    class FakeMistralClientFailure:
        def __init__(self):
            self.files = FakeFiles()
            self.batch = FakeBatchFailure()

    backend.client = FakeMistralClientFailure()

    with pytest.raises(Exception, match="Job failed: FAILED"):
        backend._execute_batch_job("dummy_file_id", {"model": "test-model"})
