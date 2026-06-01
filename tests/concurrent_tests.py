import httpx
import asyncio
import time

async def send_request(client, i):
    start = time.perf_counter()
    response = await client.post(
        "http://localhost:8000/v1/completions",
        json={"prompt": "What is 2+2?", "max_tokens": 16}
    )
    elapsed = time.perf_counter() - start
    print(f"Request {i}: {elapsed*1000:.0f}ms — {response.json()}")

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [send_request(client, i) for i in range(10)]
        await asyncio.gather(*tasks)

asyncio.run(main())