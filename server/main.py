from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from .schemas import PromptSchema, FieldSchema
from .engine import InferenceEngine
from asyncio import Queue
import asyncio

from pathlib import Path

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
MAX_BATCH_SIZE = 4
BATCH_TIMEOUT = 0.05  # 50ms
queue: Queue = Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = InferenceEngine(model_path=MODEL_PATH)
    app.state.engine = engine
    task = asyncio.create_task(batch_loop(engine))
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

completion_schema = PromptSchema(
    name="completion",
    template="{prompt}",
    fields={
        "prompt": FieldSchema(type=str, required=True, description="The prompt text to complete"),
    },
)


async def handle_request(prompt: str, max_tokens: int) -> str:
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    await queue.put(({"prompt": prompt, "max_tokens": max_tokens}, future))
    return await future


async def batch_loop(engine: InferenceEngine):
    while True:
        batch = []

        # Block until the first request arrives
        try:
            batch.append(await queue.get())
        except asyncio.CancelledError:
            return

        # Collect more requests up to MAX_BATCH_SIZE within BATCH_TIMEOUT
        deadline = asyncio.get_event_loop().time() + BATCH_TIMEOUT
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                for _, fut in batch:
                    if not fut.done():
                        fut.cancel()
                return

        requests, futures = zip(*batch)
        loop = asyncio.get_event_loop()
        try:
            # Run blocking inference in a thread so the event loop stays responsive
            results = await loop.run_in_executor(
                None, lambda: [engine.generate(r["prompt"]) for r in requests]
            )
            for fut, result in zip(futures, results):
                if not fut.done():
                    fut.set_result(result)
        except Exception as exc:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(exc)


@app.post("/v1/completions")
async def completions(inputs: dict = Body(...)):
    try:
        rendered = completion_schema.render(inputs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    max_tokens = inputs.get("max_tokens", 128)
    return {"text": await handle_request(rendered, max_tokens)}
