# highload_test_mistral_batch.py
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv

from llm_inference.backends.mistral_batch import MistralBatchBackend
from llm_inference.helpers import get_model_config

load_dotenv()

def highload_test(num_requests: int):
    # Get model configuration (ensure "mistral_test" is defined in your config)
    model_config = get_model_config("mistral_test")

    # Instantiate the batch backend using the Mistral API key from environment variables.
    backend = MistralBatchBackend(api_key=os.getenv("MISTRAL_API_KEY"))

    # Create a list of prompts with custom IDs.
    prompts = [{"prompt": f"{i}. Tell me a story.", "custom_id": i} for i in range(num_requests)]

    start_time = time.time()

    # Process the batch inference.
    results = []
    pbar = tqdm(total=num_requests, desc="Processing batch inference")
    for result in backend.infer_many(prompts, model_config):
        results.append(result)
        pbar.update(1)
    pbar.close()

    end_time = time.time()
    print(f"Processed {num_requests} requests in {end_time - start_time:.2f} seconds\n")

    # Print the results.
    for idx, result in enumerate(results):
        print(f"Result {idx}: {result}")

if __name__ == "__main__":
    num_requests = 20  # You can adjust the number of requests for testing.
    highload_test(num_requests)
# Processed 20 requests in 6.46 seconds