import sys
import os

# Bypass torchaudio/torchvision conflicts for text-only LLMs
sys.modules["torchaudio"] = None
sys.modules["torchvision"] = None

import time
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set UTF-8 encoding for standard console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Model configurations
OLLAMA_URL = "http://localhost:11434/api/generate"
GGUF_MODEL_NAME = "qwen2.5:1.5b"
HF_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_TXT_FILE = os.path.join(os.path.dirname(__file__), "benchmark_model_comparison_results.txt")

# List of 5 benchmark prompts
prompts = [
    {
        "id": 1,
        "title": "Logical Reasoning & Math",
        "text": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show step-by-step reasoning."
    },
    {
        "id": 2,
        "title": "Code Generation",
        "text": "Write a Python function to check if a given string is a palindrome, ignoring spaces, punctuation, and capitalization. Include docstrings and test cases."
    },
    {
        "id": 3,
        "title": "Summarization",
        "text": "Extract key insights into 3 concise bullet points from the following text:\n'Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computation. While classical computers process bits (0s and 1s), quantum computers use qubits which can exist in multiple states simultaneously. This allows quantum algorithms to solve specific complex problems—like integer factorization and molecular simulation—exponentially faster than classical algorithms.'"
    },
    {
        "id": 4,
        "title": "Creative Story Writing",
        "text": "Write a short sci-fi story in exactly 3 sentences about a rogue AI on a sleeper colony ship that secretly protects humans. The last sentence must contain exactly 10 words."
    },
    {
        "id": 5,
        "title": "Machine Learning Concept",
        "text": "Explain the difference between overfitting and underfitting in machine learning, and list two distinct techniques to mitigate each."
    }
]

print("=================================================================================")
print("          FULL COMPARISON BENCHMARK: TIME & THROUGHPUT (FP16 vs GGUF)            ")
print("=================================================================================\n")

# Initialize HF Tokenizer for FP16
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

def format_prompt(p_text):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": p_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {p_text}\nAssistant:"

# -----------------------------------------------------------------------------
# 1. Run FP16 Baseline (PyTorch CPU / FP16)
# -----------------------------------------------------------------------------
print("--> [1/2] Measuring Full Precision FP16 Baseline...")
model_fp16 = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.float32)

fp16_metrics = []
fp16_responses = []

for item in prompts:
    formatted = format_prompt(item["text"])
    inputs = tokenizer(formatted, return_tensors="pt").to(model_fp16.device)
    input_len = inputs.input_ids.shape[1]
    
    t0 = time.time()   
    outputs = model_fp16.generate(**inputs, max_new_tokens=512)
    t1 = time.time()
    
    gen_tokens = outputs[0].shape[0] - input_len
    elapsed = t1 - t0
    speed = gen_tokens / elapsed if elapsed > 0 else 0.0
    
    response_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    fp16_responses.append(response_text)
    fp16_metrics.append({"tokens": gen_tokens, "time": elapsed, "speed": speed})

# Free FP16 model from RAM
del model_fp16
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 2. Run 4-Bit GGUF Model (Ollama Engine on GPU)
# -----------------------------------------------------------------------------
print("--> [2/2] Measuring 4-Bit GGUF Model (Ollama GPU Engine)...\n")

gguf_metrics = []
gguf_responses = []

for item in prompts:
    payload = {
        "model": GGUF_MODEL_NAME,
        "prompt": item["text"],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512}
    }
    
    t0 = time.time()
    response = requests.post(OLLAMA_URL, json=payload).json()
    t1 = time.time()
    
    eval_count = response.get("eval_count", 0)
    eval_duration_sec = response.get("eval_duration", 1) / 1e9
    speed = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0.0
    elapsed = t1 - t0
    
    res_text = response.get("response", "").strip()
    gguf_responses.append(res_text)
    gguf_metrics.append({"tokens": eval_count, "time": elapsed, "speed": speed})

# -----------------------------------------------------------------------------
# 3. Print & Save Comprehensive Comparison Results
# -----------------------------------------------------------------------------
table_lines = []
table_lines.append("=========================================================================================================")
table_lines.append("                               FULL PRECISION (FP16) vs 4-BIT GGUF COMPARISON                            ")
table_lines.append("=========================================================================================================")
header = f"{'Prompt Task':<25} | {'FP16 Time':<10} | {'GGUF Time':<10} | {'FP16 Speed':<12} | {'GGUF Speed':<12} | {'Time Reduction'}"
table_lines.append(header)
table_lines.append("-" * 105)

total_fp16_time = 0.0
total_gguf_time = 0.0
total_fp16_speed = 0.0
total_gguf_speed = 0.0

for i, item in enumerate(prompts):
    fp16_t = fp16_metrics[i]["time"]
    gguf_t = gguf_metrics[i]["time"]
    fp16_sp = fp16_metrics[i]["speed"]
    gguf_sp = gguf_metrics[i]["speed"]
    
    time_saved = ((fp16_t - gguf_t) / fp16_t) * 100 if fp16_t > 0 else 0.0
    
    total_fp16_time += fp16_t
    total_gguf_time += gguf_t
    total_fp16_speed += fp16_sp
    total_gguf_speed += gguf_sp
    
    row = f"{item['title']:<25} | {fp16_t:>8.2f}s | {gguf_t:>8.2f}s | {fp16_sp:>8.2f} t/s | {gguf_sp:>8.2f} t/s | {time_saved:>12.1f}% faster"
    table_lines.append(row)

avg_fp16_speed = total_fp16_speed / len(prompts)
avg_gguf_speed = total_gguf_speed / len(prompts)
total_time_saved = ((total_fp16_time - total_gguf_time) / total_fp16_time) * 100 if total_fp16_time > 0 else 0.0

table_lines.append("-" * 105)
summary_row = f"{'TOTAL / AVERAGE METRICS':<25} | {total_fp16_time:>8.2f}s | {total_gguf_time:>8.2f}s | {avg_fp16_speed:>8.2f} t/s | {avg_gguf_speed:>8.2f} t/s | {total_time_saved:>12.1f}% faster"
table_lines.append(summary_row)
table_lines.append("=========================================================================================================")

# Print to console
for line in table_lines:
    print(line)

# Save to TXT file on device
print(f"\n--> Saving complete untruncated comparison and model answers to local file: {OUTPUT_TXT_FILE}")
with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(table_lines) + "\n\n")
    f.write("=================================================================================\n")
    f.write("                   DETAILED SIDE-BY-SIDE MODEL ANSWERS                           \n")
    f.write("=================================================================================\n\n")
    
    for i, item in enumerate(prompts):
        f.write("=" * 80 + "\n")
        f.write(f"PROMPT {item['id']}: [{item['title']}]\n")
        f.write("=" * 80 + "\n")
        f.write(f"PROMPT TEXT:\n{item['text']}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"[MODEL 1: Full Precision FP16 Response] (Time: {fp16_metrics[i]['time']:.2f}s | Speed: {fp16_metrics[i]['speed']:.2f} t/s)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{fp16_responses[i]}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"[MODEL 2: 4-Bit GGUF Quantized Response] (Time: {gguf_metrics[i]['time']:.2f}s | Speed: {gguf_metrics[i]['speed']:.2f} t/s)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{gguf_responses[i]}\n\n")
        f.write("\n" + "=" * 80 + "\n\n")

print("--> Local TXT File Saved Successfully with Complete Untruncated Responses!\n")
