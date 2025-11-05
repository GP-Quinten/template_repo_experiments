import asyncio
import inspect

from typing import AsyncGenerator


async def _iter_backend_responses(responses) -> AsyncGenerator:
    if inspect.isasyncgen(responses):
        async for resp in responses:
            yield resp
    else:
        resp_list = await asyncio.to_thread(lambda: list(responses))
        for resp in resp_list:
            yield resp
