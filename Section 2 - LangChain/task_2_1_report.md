# Section 2 Write-Up: RAG Chunking & Retrieval Optimizations

If answer quality on longer documents (such as extensive manuals or PDFs) is poor, the following optimizations should be implemented to replace standard character-based splitters and basic vector search:

### 1. Advanced Chunking Strategies
- **Semantic Chunking**: Instead of rigid character limits, calculate embeddings for each sentence and measure cosine similarities between adjacent sentences. Split chunks only when similarity drops below a threshold (e.g., the 85th percentile), ensuring each chunk represents a single coherent topic.
- **Parent-Child (Hierarchical) Chunking**: Split documents into small child chunks (e.g., 100–150 tokens) for highly focused semantic search indexing. Link each child to a larger parent chunk (e.g., 600–800 tokens). When a child chunk matches during search, retrieve and pass the *parent* chunk to the LLM. This provides rich context without diluting search query precision.
- **Header-Aware Splitting**: Utilize `MarkdownHeaderTextSplitter` to group sections according to structural headers (`#`, `##`), keeping tables, lists, and code blocks intact.

### 2. Retrieval & Query Improvements
- **Hybrid Search (Dense + Sparse)**: Combine semantic dense vector search (using HuggingFace embeddings) with exact-match sparse keyword retrieval (BM25). Merge their rankings using **Reciprocal Rank Fusion (RRF)** to handle both conceptual queries and exact keyword/ID lookups.
- **Cross-Encoder Re-ranking**: Retrieve a larger window of candidate chunks (e.g., top 15) using fast vector search, then pass them through a **Cross-Encoder Re-ranker** (like Cohere Rerank or local `BAAI/bge-reranker-large`). Cross-encoders analyze query and document text together to generate highly accurate relevance scores. Only feed the top 3–5 re-ranked chunks to the LLM, reducing context noise.
- **Query Rewriting (HyDE)**: Use a query expansion model or Hypothetical Document Embeddings (HyDE) to generate a mock response, which is then embedded to query the database, improving query-to-document matching.
