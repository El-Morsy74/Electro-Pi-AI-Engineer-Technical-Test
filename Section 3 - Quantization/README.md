# ⚖️ Model Quantization & Trade-off Analysis (Section 3 — Quantization)

This section evaluates the practical performance, file size, memory utilization, throughput speed, and qualitative response quality of full-precision vs. 4-bit quantized Large Language Models (LLMs). The benchmark evaluates the `Qwen/Qwen2.5-1.5B-Instruct` model, comparing HuggingFace Transformers (FP16 CPU baseline) against a 4-bit GGUF (`Q4_K_M`) version run locally via Ollama (`llama.cpp` inference engine).

---

## 📂 Project Structure

```text
Section 3 - Quantization/
├── benchmark_model_comparison_results.txt # Full precision vs. GGUF log
├── simple_ollama_gguf.py                  # Benchmarking evaluation pipeline
├── simple_quantization_demo.py            # NF4 dynamic loader demo script
├── requirements.txt                       # Python dependencies list
├── task_3_1_report.md                     # Quantization benchmarks report
└── README.md                              # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
Ensure Python 3.10+ is installed along with CUDA toolkits if GPU acceleration is desired.
```bash
pip install torch transformers bitsandbytes accelerate requests
```

### 2. Install and Configure Ollama (For GGUF execution)
1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Pull the target GGUF model:
   ```bash
   ollama pull qwen2.5:1.5b
   ```

---

## 🚀 How to Run the Benchmark

1. Start your local Ollama server (usually runs automatically in the background).
2. Execute the benchmark script:
   ```bash
   python simple_ollama_gguf.py
   ```
3. The benchmark will run the 5 test prompts on both models, verify the qualitative answers, print metrics to the console, and log results to `benchmark_model_comparison_results.txt`.
