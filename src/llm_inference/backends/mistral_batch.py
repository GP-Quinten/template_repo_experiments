import json
import tempfile
import time
from typing import List, Union, Generator, Optional

from mistralai import Mistral
from llm_inference.cache.base import AbstractCacheStorage
from llm_inference.backends.mistral_base import MistralBatchBaseBackend


def map_batch_results(batch_result):
    result = {
        "custom_id": batch_result["custom_id"],
    }
    result.update(batch_result["response"]["body"])
    return result


class MistralBatchBackend(MistralBatchBaseBackend):
    """Backend implementation using the Mistral API for batch inference.

    This class performs inference in batches using the Mistral API and supports caching.
    """

    def __init__(self, api_key: str, cache_storage: Optional[AbstractCacheStorage] = None):
        """
        Initializes the MistralBatchBackend.

        Args:
            api_key (str): API key for authenticating with the Mistral service.
            cache_storage (AbstractCacheStorage, optional): A cache storage implementation.
        """
        self.client = Mistral(api_key=api_key)
        self.cache_storage = cache_storage
        self.logger.info("MistralBatchBackend initialized.")

    def _make_batch_data_from_prompts(
        self, prompts: List[Union[str, dict]], model_config: dict
    ) -> List[dict]:
        """
        Creates batch data from a list of prompts.

        Each prompt is converted into a dictionary with a custom ID and
        the required body for the batch inference request.

        Args:
            prompts (List[Union[str, dict]]): A list of input prompts.
            model_config (dict): A dictionary containing model parameters and settings.

        Returns:
            List[dict]: A list of dictionaries representing the batch data.
        """
        self.logger.info("Creating batch data from prompts.")
        batch_data = []
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, dict):
                content = prompt.get("prompt")
                custom_id = str(prompt["custom_id"])
            else:
                content = prompt
                custom_id = str(i)
            batch_data.append({
                "custom_id": custom_id,
                "body": {
                    "max_tokens": model_config["max_tokens"],
                    "temperature": model_config["temperature"],
                    "response_format": model_config["response_format"],
                    "random_seed": model_config["random_seed"],
                    "n": model_config.get("n", 1),
                    "messages": [{
                        "role": "user",
                        "content": content,
                    }],
                },
            })
        self.logger.info(f"Created batch data for {len(batch_data)} prompts.")
        return batch_data

    def _upload_batch_file(self, batch_data: List[dict]):
        """
        Uploads a batch file containing the batch data to the Mistral service.

        This method writes the batch data to a temporary JSONL file and then
        uploads it to the Mistral service.

        Args:
            batch_data (List[dict]): The batch data to upload.

        Returns:
            The uploaded file object.
        """
        self.logger.info("Uploading batch file.")
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            for data in batch_data:
                f.write(json.dumps(data) + "\n")
            f.flush()
            batch_file = self.client.files.upload(
                file={
                    "file_name": "batch.jsonl",
                    "content": open(f.name, "rb"),
                },
                purpose="batch",
            )
        self.logger.info(f"Batch file uploaded with ID: {batch_file.id}")
        return batch_file

    def _get_batch_results(self, results_file: str) -> List[dict]:
        """
        Downloads and parses the batch results from a given file.

        Args:
            results_file (str): The identifier of the results file.

        Returns:
            List[dict]: A list of dictionaries containing the results.
        """
        self.logger.info(f"Downloading results from file ID: {results_file}")
        job_result_file = self.client.files.download(file_id=results_file)
        job_content = job_result_file.read().decode("utf-8")
        results = []
        for line in job_content.split("\n"):
            if line:
                results.append(json.loads(line))
        self.logger.info(f"Downloaded {len(results)} results.")
        return results

    def _execute_batch_job(self, batch_file_id: str, model_config: dict):
        """
        Executes a batch inference job using the uploaded batch file.

        This method creates a batch job, polls until the job is complete, and
        returns the job result.

        Args:
            batch_file_id (str): The ID of the uploaded batch file.
            model_config (dict): A dictionary containing model parameters and settings.

        Returns:
            The job object containing the results.

        Raises:
            Exception: If the job status is not 'SUCCESS'.
        """
        self.logger.info(f"Creating batch job with file ID: {batch_file_id}")
        created_job = self.client.batch.jobs.create(
            input_files=[batch_file_id],
            model=model_config["model"],
            endpoint="/v1/chat/completions",
            metadata={"job_type": "inference"},
        )
        self.logger.info(f"Job created with ID: {created_job.id}")

        retrieved_job = self.client.batch.jobs.get(job_id=created_job.id)
        while retrieved_job.status in ["RUNNING", "QUEUED"]:
            self.logger.info(f"Job status: {retrieved_job.status}")
            retrieved_job = self.client.batch.jobs.get(job_id=created_job.id)
            time.sleep(0.5)

        if retrieved_job.status != "SUCCESS":
            self.logger.error(f"Job failed with status: {retrieved_job.status}")
            raise Exception(f"Job failed: {retrieved_job.status}")

        self.logger.info("Batch job completed successfully.")
        return retrieved_job
    
    def postprocess(self, raw_results):
        results = map(map_batch_results, raw_results)
        return results

    def infer_many(
        self, 
        prompts: List[Union[str, dict]], 
        model_config: dict, 
        use_cache: bool = True
    ) -> Generator[dict, None, None]:
        """
        Performs batch inference on a list of prompts.

        This method calculates a cache key based on the batch data by converting
        it to a stable JSON string and then calling the cache storage's _generate_hash.
        If the result is cached, it yields the cached results; otherwise, it executes
        the batch job, caches the result, and yields the results.

        Args:
            prompts (List[Union[str, dict]]): A list of input prompts.
            model_config (dict): A dictionary containing model parameters and settings.
            use_cache (bool): Whether to use caching (default: True).

        Yields:
            dict: Each inference result from the batch.
        """
        self.logger.info("Starting batch inference.")
        batch_data = self._make_batch_data_from_prompts(prompts, model_config)

        # Use a stable JSON string representation for hashing.
        if self.cache_storage is not None:
            batch_data_str = json.dumps(batch_data, sort_keys=True)
            cache_key = f"mistral_batch_{self.cache_storage._generate_hash(batch_data_str)}"

        if use_cache and self.cache_storage is not None:
            cached_result = self.cache_storage.get(cache_key)
            if cached_result is not None:
                self.logger.info("Using cached batch results.")
                results = map(map_batch_results, cached_result)
                for res in results:
                    yield res
                return

        self.logger.info("No cached result found, executing batch job.")
        batch_file = self._upload_batch_file(batch_data)
        job = self._execute_batch_job(batch_file.id, model_config)
        raw_results = self._get_batch_results(job.output_file)
        self.logger.info("Batch inference completed.")
        
        if use_cache and self.cache_storage is not None:
            self.cache_storage.put(cache_key, raw_results)

        results = map(map_batch_results, raw_results)
        for res in results:
            yield res
