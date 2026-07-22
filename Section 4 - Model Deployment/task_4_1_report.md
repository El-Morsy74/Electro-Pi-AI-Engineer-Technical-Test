# Section 4 — Task 4.1: Model Deployment & Production Scaling Report

## 1. API Architecture & Justification

The `Qwen2.5-0.5B-Instruct` model is deployed as a REST API service using **FastAPI** & **Uvicorn** with **4-bit NF4 Quantization** (`bitsandbytes`) and **Server-Sent Events (SSE)** token streaming.

- **Justification (FastAPI vs. vLLM/TGI)**: **FastAPI** was chosen for base service delivery due to its low runtime memory overhead, cross-platform portability (CPU & GPU support without CUDA compute capability locking), and full compatibility with OpenAI API specifications. While **vLLM** offers superior continuous batching throughput, it requires specialized Linux environments with high VRAM allocations.

---

## 2. Load & Latency Test Results (10 Concurrent Requests)

An asynchronous load test (`load_test.py`) dispatched 10 simultaneous streaming requests to measure initial responsiveness and throughput:

| Metric | Result | Analysis |
| :--- | :--- | :--- |
| **Total Concurrent Requests** | **10** | 100% Success Rate (10/10 completed) |
| **Avg Time-To-First-Token (TTFT)** | **3,969.18 ms** (~3.97s) | Min: 3.86s, Max: 4.02s |
| **Avg Total Request Latency** | **12.65 seconds** | End-to-end average per stream |
| **Total Test Wall Duration** | **17.23 seconds** | All 10 requests completed in parallel |

---

## 3. Production Write-Up: Scaling to 50 Concurrent Users

To scale an LLM serving infrastructure from 10 concurrent requests to **50+ active concurrent users in production** while maintaining sub-100ms Time-To-First-Token (TTFT) and high stream throughput, the baseline single-worker FastAPI setup must evolve into an enterprise-grade inference pipeline:

```
                      ┌───────────────────────────┐
                      │   NGINX / Ingress Router  │
                      └─────────────┬─────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
          ┌────────────────────┐        ┌────────────────────┐
          │  Redis / RabbitMQ  │        │   Prefix / Prompt  │
          │   Request Queue    │        │    RadixCache      │
          └─────────┬──────────┘        └──────────┬─────────┘
                    │                              │
     ┌──────────────┴──────────────┬───────────────┴──────────────┐
     ▼                             ▼                              ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ vLLM Engine Pod 1│          │ vLLM Engine Pod 2│          │ vLLM Engine Pod N│
│ Continuous Batch │          │ Continuous Batch │          │ Continuous Batch │
│   (PagedAttention│          │   (PagedAttention│          │   (PagedAttention│
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

1. **Continuous Batching (vLLM / PagedAttention)**: Replace PyTorch native serving with vLLM. Continuous iteration-level batching dynamically merges new prompts into active GPU passes, increasing hardware utilization and boosting throughput by **10x–15x**.
2. **KV-Cache Prefix Caching (RadixTree)**: Cache common system prompts and RAG contexts in GPU memory. Subsequent requests bypass redundant prefill computations, cutting Time-To-First-Token (TTFT) from **~3.97s to under 50ms**.
3. **Queueing & Rate-Limiting (Redis / RabbitMQ)**: Front the inference nodes with an asynchronous message broker. Enforce a token bucket rate limiter to prevent CUDA Out-Of-Memory (OOM) crashes from sudden traffic surges.
4. **Horizontal Autoscaling (Kubernetes + KEDA)**: Run workers inside Kubernetes. Scale pod replicas dynamically using KEDA triggers based on GPU KV-Cache pressure (>80%) or pending queue depth (>15).
