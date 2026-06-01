from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Request, Depends
from .schemas import PromptSchema, FieldSchema
from .engine import InferenceEngine

from pathlib import Path
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = InferenceEngine(model_path=MODEL_PATH)
    yield

app = FastAPI(lifespan=lifespan)

def get_engine(request: Request) -> InferenceEngine:
    return request.app.state.engine

completion_schema = PromptSchema(
    name="completion",
    template="{prompt}",
    fields={
        "prompt": FieldSchema(type=str, required=True, description="The prompt text to complete"),
    },
)

@app.post("/v1/completions")
async def completions(
    inputs: dict = Body(...),
    engine: InferenceEngine = Depends(get_engine),
):
    try:
        rendered = completion_schema.render(inputs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"text": engine.generate(rendered)}
