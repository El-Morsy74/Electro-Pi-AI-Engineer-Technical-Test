# 🚀 Electro Pi — AI Engineer Technical Test

This repository contains the assessment tasks for the Mid-Level AI Engineer position at Electro Pi, organized into four sections.

---

## 📂 Project Organization

1. **[Section 1: LiveKit Voice Agent](file:///d:/Task/Section%201%20-%20LiveKit%20Agents)**
   - Real-time order-status voice assistant.
2. **[Section 2: LangChain RAG](file:///d:/Task/Section%202%20-%20LangChain)**
   - Context-grounded retrieval system.
3. **[Section 3: Quantization](file:///d:/Task/Section%203%20-%20Quantization)**
   - FP16 vs. 4-bit GGUF model benchmarks.
4. **[Section 4: Model Serving & Deployment](file:///d:/Task/Section%204%20-%20Model%20Deployment)**
   - FastAPI server with concurrent load testing.

---

## ⚙️ Fast Run Guide

### 1. Install Dependencies
```bash
pip install -r "Section 1 - LiveKit Agents/requirements.txt"
pip install -r "Section 2 - LangChain/requirements.txt"
pip install -r "Section 3 - Quantization/requirements.txt"
pip install -r "Section 4 - Model Deployment/requirements.txt"
```

### 2. Run Commands

#### Section 1: LiveKit Voice Agent
```bash
cd "Section 1 - LiveKit Agents"
python agent.py start
```

#### Section 2: LangChain RAG
```bash
cd "Section 2 - LangChain"
python main.py
```

#### Section 3: Quantization
```bash
cd "Section 3 - Quantization"
python simple_ollama_gguf.py
```

#### Section 4: Model Serving
```bash
cd "Section 4 - Model Deployment"
uvicorn app:app --host 0.0.0.0 --port 8000
# Run load test: python load_test.py
```


