import os
import time

from dotenv import load_dotenv
from tqdm import tqdm

from llm_inference.backends.mistral_sync import MistralBackend
from llm_inference.helpers import get_model_config


load_dotenv()


def highload_test(num_requests: int):
    backend = MistralBackend(
        model_config=get_model_config("mistral_test"),
        api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    prompts = [f"{i}. Tell me a story." for i in range(num_requests)]

    start_time = time.time()

    ##################################
    # Example of using MistralBackend (synchronous version)
    ##################################
    results = []
    for result in tqdm(
        backend.infer_many(prompts), total=num_requests, desc="Processing sync batch"
    ):
        results.append(result)
    ##################################

    end_time = time.time()
    print(f"Processed {num_requests} requests in {end_time - start_time:.2f} seconds")

    for idx, result in enumerate(results):
        print(f"Result {idx}: {result}")


if __name__ == "__main__":
    num_requests = 20  # For example, 100 requests in a single batch.
    highload_test(num_requests)
# Processed 20 requests in 13.40 seconds
