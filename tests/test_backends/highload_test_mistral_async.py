import os
import time

import asyncio
from tqdm import tqdm
from dotenv import load_dotenv

from llm_inference.backends.mistral_async import MistralAsyncBackend
from llm_inference.helpers import get_model_config


load_dotenv()


async def highload_test(num_requests: int):
    # Instantiate the async backend.
    model_config = get_model_config("mistral_test")

    backend = MistralAsyncBackend(    
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    prompts = [{"prompt": f"{i}. Tell me a story.", "custom_id": i} for i in range(num_requests)]

    start_time = time.time()

    ##################################
    # Example of using AsyncMistralBackend
    ##################################
    results = []
    pbar = tqdm(total=num_requests, desc="Processing async batch")
    async for result in backend.infer_many(prompts, model_config):
        results.append(result)
        pbar.update(1)
    pbar.close()
    ##################################

    end_time = time.time()
    print(f"Processed {num_requests} requests in {end_time - start_time:.2f} seconds")

    for idx, result in enumerate(results):
        print(f"Result {idx}: {result}")


if __name__ == "__main__":
    num_requests = 20
    asyncio.run(highload_test(num_requests))
# Processed 20 requests in 3.49 seconds
# Processed 20 requests in 4.80 seconds
