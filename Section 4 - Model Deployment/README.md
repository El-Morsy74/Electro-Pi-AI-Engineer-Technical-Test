# Section 4 — Model Deployment: REST API Service & Production Scaling

An enterprise-grade, lightweight REST API service deploying `Qwen2.5-0.5B-Instruct` with **4-bit NF4 Quantization**, real-time **Server-Sent Events (SSE) token streaming**, **Docker containerization**, asynchronous **load testing**, and an architectural roadmap for scaling to **50+ concurrent users** in production.

---

## 📌 Project Overview & Key Features

- **Framework**: **FastAPI** & **Uvicorn** for asynchronous HTTP execution.
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` (0.5 Billion Parameter Open-Weight LLM).
- **Quantization**: **4-bit NF4 Quantization** via `bitsandbytes` (Reduces VRAM footprint down to ~650MB).
- **Real-Time Streaming**: Asynchronous token-by-token streaming via Server-Sent Events (`text/event-stream`) using `transformers.TextIteratorStreamer`.
- **Load Testing**: Asynchronous multi-connection benchmark script (`load_test.py`) measuring Time-To-First-Token (TTFT) and Total Latency across 10 parallel streams.
- **Containerization**: Minimal multi-stage `Dockerfile` (`python:3.10-slim`) for reproducible deployment.

---

## 📁 Repository Structure

```text
Section 4 - Model Deployment/
├── app.py                # Main FastAPI model serving application
├── load_test.py          # Asynchronous 10-connection load testing script
├── load_test_results.json# Exported benchmark metrics JSON
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build specification
├── task_4_1_report.md    # Detailed technical evaluation report
└── README.md             # Project documentation
```

---

## 🚀 Quick Start Guide (Run within < 5 Minutes)

### 1. Local Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the FastAPI model server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker Container Deployment

```bash
# Build Docker image
docker build -t qwen-api-service .

# Run containerized service
docker run -d -p 8000:8000 --name qwen-service qwen-api-service
```

---

## 📡 API Usage & Endpoints

### 1. Health Check
`GET http://localhost:8000/health`

### 2. Text Generation & Streaming
`POST http://localhost:8000/generate` 

#### A. Standard Response (`stream: false`)
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is Retrieval-Augmented Generation (RAG)?",
       "max_tokens": 150,
       "stream": false
     }'
```

#### B. Token-by-Token Streaming (`stream: true`)
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Explain RAG in 15 words.",
       "max_tokens": 50,
       "stream": true
     }'
```

---

## 📊 Benchmark & Load Test Results (10 Concurrent Requests)

Run the load test script:
```bash
python load_test.py
```

### Summary Metrics

```text
===========================================================================
BENCHMARK SUMMARY REPORT (10 CONCURRENT REQUESTS)
===========================================================================
  * Total Concurrent Requests     : 10
  * Successful Requests            : 10/10 (100% Success Rate)
  * Avg Time-To-First-Token (TTFT) : 3969.18 ms (3.97 s)
  * Min TTFT                      : 3865.00 ms
  * Max TTFT                      : 4017.49 ms
  * Avg Total Request Latency     : 12.65 s
  * Total Test Wall Duration      : 17.23 s
===========================================================================
```
