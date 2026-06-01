# llm-inference-server

A small-scale LLM inference server built from scratch. The goal is not to compete with vLLM — it's to understand what vLLM does and why, by building a simplified version end-to-end and measuring every architectural decision against real numbers.

By the end of this project you should be able to explain, with benchmarks, why continuous batching beats static batching, what the KV cache is actually managing, and where a naive scheduler falls apart under load. That understanding is the deliverable. The code is just how you get there.

---

## What this is

An HTTP server that:

- Accepts text prompts over a REST API (OpenAI-compatible `/v1/completions`)  
- Holds a quantized language model loaded in memory  
- Generates token-by-token responses, optionally streamed  
- Manages a request queue and batches work efficiently against a single model instance

The hard part is not the HTTP layer or the model call. The hard part is the scheduler and memory manager — deciding which requests run together, how much memory each is allowed, and what happens when you're at capacity. That's where this project lives.

---

## Stack

### Language: Python

This might surprise you given the systems background, but it's the right call here for a specific reason: the inference engine is already a C++ binary underneath (`llama.cpp`). Your server is the orchestration layer above it. At that layer, Python's async primitives, readable concurrency model, and ecosystem (FastAPI, pytest, locust) are a better fit than C++ would be — and it's exactly what vLLM, TGI, and SGLang are built in.

If this were a latency-critical inner loop, the answer would be different. It's not. The bottleneck is the model forward pass, which is C++ regardless of what language wraps it.

**Tradeoff:** You give up fine-grained memory control and the ability to use lock-free structures in the scheduler. At small scale this doesn't matter. At production scale (thousands of QPS, H100 clusters) you'd need a different answer, and understanding why is something this project will make concrete.

### Inference Engine: llama-cpp-python

`llama.cpp` is a C++ inference runtime that runs quantized GGUF models on CPU and Apple Silicon with no GPU required. The Python bindings (`llama-cpp-python`) give you a clean interface without touching C++.

Why not Hugging Face Transformers? Transformers is the obvious choice but it's built for flexibility, not throughput. It doesn't give you control over the KV cache, doesn't expose batching internals, and runs significantly slower for inference on CPU/Apple Silicon. You'd be fighting the abstraction instead of learning from it.

Why not ONNX Runtime or TensorRT? Both require GPU and significantly more setup. The goal here is to focus on scheduling and memory management logic, not CUDA kernel optimization.

**Tradeoff:** `llama.cpp` abstracts away the forward pass, which means you can't do custom attention patterns or modify the KV cache at the tensor level. You're managing cache at the request/sequence level, not the tensor level. That's the right scope for this project. If you want to go deeper on the actual GPU kernels, FlashAttention and PagedAttention (vLLM's core innovation) are the next step after this.

### HTTP Server: FastAPI \+ Uvicorn

FastAPI is async-native, which matters here. When one request is waiting on a model forward pass, you want the event loop free to accept new connections and manage the queue — not blocked on a synchronous call. Uvicorn is the ASGI server that runs it.

Flask would work for week 1 but would require significant rework by week 3 when you add streaming (SSE) and concurrent request handling. Starting with FastAPI avoids that rewrite.

**Tradeoff:** FastAPI's async model means you need to be careful about blocking the event loop with synchronous model calls. The inference call runs in a thread pool executor — you'll implement this explicitly, which is a good forcing function for understanding Python's concurrency model.

### Model: Qwen2.5-1.5B-Instruct (Q4 quantized, GGUF)

1.5B parameters, \~1GB on disk after 4-bit quantization. Fast enough that you can run hundreds of benchmark requests without waiting all day, small enough to fit in RAM comfortably alongside your server process.

The specific model doesn't matter much — the scheduling logic is model-agnostic. What matters is that it's small enough to iterate on quickly and large enough to have realistic generation characteristics (variable output length, meaningful context window).

**Tradeoff:** At this size and quantization level, output quality is noticeably degraded compared to a full-precision 7B or 13B model. That's acceptable — you're measuring throughput and latency, not evaluating response quality.

---

## What you are not building

- Multi-GPU or distributed inference (no tensor parallelism, no pipeline parallelism)  
- Custom CUDA kernels or attention implementations  
- A production-grade deployment (no auth, no rate limiting per user, no SLA guarantees)  
- Speculative decoding

These are real production concerns. They're excluded not because they're unimportant but because they'd obscure the core ideas. Once you understand the single-machine scheduling and memory management problem deeply, the distributed extensions become much more readable.

---

## Development Roadmap

Each phase has a concrete deliverable and a measurable exit criterion. Don't move to the next phase until you can demonstrate the exit criterion with numbers.

---

### Phase 1 — Baseline: One Request at a Time

**Target: 1 week**

Build the simplest possible thing that works end-to-end.

- `engine.py`: Load the model once at startup, expose a `generate(prompt, max_tokens)` method  
- `main.py`: FastAPI app with a single `POST /v1/completions` route  
- `schemas.py`: Pydantic models for request/response validation  
- Basic integration tests against the live endpoint

**Exit criterion:** `curl` a prompt, get a response. One request at a time, no concurrency, no batching. Benchmark it: what's p50 latency? What's throughput in req/s? Write these numbers down. Everything you build after this is measured against them.

**What you'll learn:** How slow sequential inference actually is. Why you can't just throw threads at it naively.

---

### Phase 2 — Static Batching

**Target: 1 week**

Add a queue. Collect requests that arrive within a short window and send them to the model together as a batch.

- Request queue with a configurable wait window (e.g., 50ms)  
- Batch the queued prompts into a single model call  
- Return responses when the full batch completes

**Exit criterion:** Under simulated concurrent load (10 simultaneous requests), throughput improves meaningfully over Phase 1\. Quantify it. Also measure the latency regression — static batching makes fast requests wait for slow ones. Write that number down too. This is the core tradeoff that motivates Phase 3\.

**What you'll learn:** Why batching helps (hardware utilization), and why static batching has a fundamental problem (head-of-line blocking from variable-length requests).

---

### Phase 3 — Continuous Batching

**Target: 1–2 weeks**

The main algorithmic contribution of the project. Instead of waiting for a full batch to complete before starting new requests, new requests join the batch mid-flight. Requests that finish free their slot immediately.

- Token-by-token generation loop (iterate one step across all in-flight requests per forward pass)  
- Requests join the active batch when a slot is free  
- Requests leave when they hit their stop condition  
- Queue drains continuously rather than in fixed windows

**Exit criterion:** Under the same load as Phase 2, throughput improves again and p95 latency comes down. The improvement over static batching should be significant and visible in your benchmark charts. If it isn't, your implementation has a bug — find it before moving on.

**What you'll learn:** Why this is the central innovation in modern inference servers. Why vLLM's PagedAttention exists (KV cache fragmentation from variable sequence lengths, which this phase will make you feel viscerally).

---

### Phase 4 — KV Cache Management

**Target: 1 week**

Right now memory is unmanaged — you're implicitly relying on `llama.cpp`'s internal cache. Make it explicit.

- Track memory consumed per in-flight request (sequence length × KV cache size per token)  
- Implement admission control: reject or queue new requests when memory would be exceeded  
- Expose cache utilization as a metric

**Exit criterion:** Under a load that would previously cause OOM or severe degradation, the server degrades gracefully — new requests queue rather than crashing the process. Memory utilization stays bounded and visible.

**What you'll learn:** Why KV cache management is the central memory problem in inference serving. Why PagedAttention (vLLM's core contribution) is a big deal — it solves KV cache fragmentation the same way virtual memory solved physical memory fragmentation.

---

### Phase 5 — Streaming \+ Cancellation

**Target: 1 week**

Production inference servers stream tokens as they're generated rather than waiting for the full response.

- Server-Sent Events (SSE) on the `/v1/completions` endpoint with `stream=true`  
- Client cancellation: when a client disconnects mid-generation, stop generating and free the slot  
- Request timeout handling

**Exit criterion:** A streaming client receives the first token within one forward-pass latency (time-to-first-token, TTFT) rather than waiting for the full response. Disconnected clients immediately free their resources.

**What you'll learn:** TTFT vs throughput as distinct metrics. Why streaming changes the user experience more than it changes the server architecture — and the one place it does matter (cancellation/resource cleanup).

---

### Phase 6 — Benchmarking Dashboard

**Target: 1 week**

The project isn't done until the results are legible.

- A `/metrics` endpoint (or Prometheus-compatible output) exposing: queue depth, active requests, cache utilization, tokens per second, p50/p95/p99 latency  
- A benchmark script that drives load and produces charts: throughput vs concurrency, latency distribution, TTFT vs total latency  
- A written comparison: naive (Phase 1\) vs static batching (Phase 2\) vs continuous batching (Phase 3), with actual numbers

**Exit criterion:** You can show someone a graph and explain in one sentence why each architectural change improved (or didn't improve) the numbers. The written comparison becomes the project narrative on your resume and in interviews.

**What you'll learn:** How to measure systems, not just build them. The ability to instrument and explain a performance result is rarer than the ability to implement the optimization.

---

## Project structure

llm-inference-server/

├── models/

│   └── qwen2.5-1.5b-instruct-q4\_k\_m.gguf

├── server/

│   ├── \_\_init\_\_.py

│   ├── main.py          \# FastAPI app, routes

│   ├── engine.py        \# llama-cpp wrapper, model loading

│   └── schemas.py       \# Pydantic request/response types

├── tests/

│   └── test\_api.py

├── bench.py             \# load generation and measurement

├── requirements.txt

├── .gitignore

└── README.md

---

## Getting started

\# Clone and set up environment

git clone \<your-repo\>

cd llm-inference-server

python3 \-m venv venv

source venv/bin/activate

pip install \-r requirements.txt

\# Download the model

huggingface-cli download \\

  Qwen/Qwen2.5-1.5B-Instruct-GGUF \\

  qwen2.5-1.5b-instruct-q4\_k\_m.gguf \\

  \--local-dir ./models

\# Verify inference works before touching the server

python \-c "

from llama\_cpp import Llama

llm \= Llama(model\_path='./models/qwen2.5-1.5b-instruct-q4\_k\_m.gguf', n\_ctx=2048)

out \= llm('Q: What is 2+2? A:', max\_tokens=16)

print(out\['choices'\]\[0\]\['text'\])

"

\# Start the server (Phase 1+)

uvicorn server.main:app \--reload

---

## Resources

- [llama-cpp-python docs](https://llama-cpp-python.readthedocs.io/)  
- [vLLM paper](https://arxiv.org/abs/2309.06180) — read this after Phase 3; it will make much more sense  
- [Continuous Batching blog post (Anyscale)](https://www.anyscale.com/blog/continuous-batching-llm-inference) — read before Phase 3  
- [FastAPI docs — Background Tasks \+ SSE](https://fastapi.tiangolo.com/)

