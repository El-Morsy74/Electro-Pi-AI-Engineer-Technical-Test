import asyncio
import json
import time
import httpx
import statistics

# ----------------------------------------------------
# 1. API Configuration & Target Endpoint
# ----------------------------------------------------
API_URL = "http://localhost:8000/generate"

# ----------------------------------------------------
# 2. Async Function to Measure Streaming TTFT & Latency
# ----------------------------------------------------
async def send_streaming_request(client, req_id):
    payload = {"prompt": "Explain RAG in 15 words", "max_tokens": 50, "stream": True}
    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        # Open streaming connection with server
        async with client.stream("POST", API_URL, json=payload, timeout=300.0) as response:
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    # Record timestamp of first token arrival (TTFT)
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
    except Exception as e:
        print(f"Request {req_id:<2}: Failed ({e})")
        return {"id": req_id, "success": False, "error": str(e)}

    end = time.perf_counter()
    total_latency = end - start
    ttft_sec = (first_token_time - start) if first_token_time else 0.0
    ttft_ms = ttft_sec * 1000.0

    # Print live per-request completion details with TTFT
    print(f"Request {req_id:<2}: TTFT = {ttft_ms:.2f} ms ({ttft_sec:.2f}s) | Total Latency = {total_latency:.2f}s | Tokens = {token_count}")

    return {
        "id": req_id,
        "success": True,
        "ttft_ms": ttft_ms,
        "total_latency_sec": total_latency,
        "tokens": token_count
    }

# ----------------------------------------------------
# 3. Main Execution & Parallel Request Dispatch
# ----------------------------------------------------
async def main():
    print("=" * 75)
    print("STARTING LOAD & LATENCY TEST: 10 CONCURRENT REQUESTS")
    print(f"Target Endpoint: {API_URL}")
    print("=" * 75)

    # Set connection limits for concurrent requests
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=10)
    async with httpx.AsyncClient(limits=limits, timeout=300.0) as client:
        start_wall = time.perf_counter()
        # Dispatch 10 parallel requests simultaneously
        results = await asyncio.gather(*[send_streaming_request(client, i + 1) for i in range(10)])
        total_wall = time.perf_counter() - start_wall

    # Aggregate and compute performance metrics
    successful = [r for r in results if r.get("success")]
    ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms")]
    latencies = [r["total_latency_sec"] for r in successful]

    avg_ttft = statistics.mean(ttfts) if ttfts else 0
    avg_latency = statistics.mean(latencies) if latencies else 0

    # Print final summary benchmark report
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY REPORT (10 CONCURRENT REQUESTS)")
    print("=" * 75)
    print(f"  * Total Concurrent Requests     : {len(results)}")
    print(f"  * Successful Requests            : {len(successful)}/10")
    print(f"  * Avg Time-To-First-Token (TTFT) : {avg_ttft:.2f} ms ({avg_ttft/1000:.2f} s)")
    print(f"  * Avg Total Request Latency     : {avg_latency:.2f} s")
    print(f"  * Total Test Wall Duration      : {total_wall:.2f} s")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(main())