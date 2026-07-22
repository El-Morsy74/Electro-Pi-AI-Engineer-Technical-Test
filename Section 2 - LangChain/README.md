# 🔍 Knowledge-Grounding RAG Pipeline (Section 2 — LangChain)

A retrieval-augmented generation (RAG) pipeline built using LangChain, HuggingFace embeddings, and Chroma vector database. The system retrieves relevant context chunks from technical documents, answers user queries with strict source-level citations, and enforces similarity guardrails to prevent hallucinations.

---

## 📂 Project Structure

```text
Section 2 - LangChain/
├── domain_docs/          # Technical knowledge corpus directory
│   └── RAG.txt
├── Chunker.py            # Sentence-aware text splitter wrapper
├── EmbeddingManager.py   # Manages HuggingFace local embedding instance
├── VectorStoreManager.py # Handles database creation & search logic
├── OpenRouterGenerator.py# Connects to OpenRouter APIs for generation
├── loader.py             # File reader (text/markdown/pdf support)
├── main.py               # Modular RAG pipeline orchestrator
├── requirements.txt      # Python dependencies list
├── task_2_1_report.md    # RAG chunking and retrieval technical write-up
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file in the `Section 2 - LangChain/` folder containing your OpenRouter key:
```ini
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

---

## 🚀 How to Run

Run the modular pipeline orchestrator:
```bash
# Set environment variables for console print encoding safety on Windows
$env:PYTHONIOENCODING="utf-8"
python main.py
```
