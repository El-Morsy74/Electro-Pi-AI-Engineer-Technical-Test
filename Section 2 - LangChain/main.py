import sys
from pathlib import Path

# Force UTF-8 encoding for standard output streams on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from loader import Loader
from Chunker import Chunker
from EmbeddingManager import EmbeddingManager
from VectorStoreManager import VectorStoreManager
from OpenRouterGenerator import OpenRouterGenerator

class RAGPipeline:
    def __init__(self, docs_dir: str = "domain_docs", persist_dir: str = "chroma_db_rag"):
        self.docs_dir = Path(__file__).parent / docs_dir
        if not self.docs_dir.exists():
            self.docs_dir = Path("domain_docs")

        self.loader = Loader(str(self.docs_dir))
        self.chunker = Chunker()
        self.vector_manager = VectorStoreManager(
            embedding_model=EmbeddingManager().get_model(),
            persist_directory=persist_dir
        )
        self.llm_generator = OpenRouterGenerator()

    def ingest(self):
        docs = self.loader.load()
        if not docs:
            return 0
        chunks = self.chunker.split(docs)
        return self.vector_manager.create(chunks)

    def ask(self, question: str, k: int = 3, min_similarity: float = 0.35):
        print(f"\nQUESTION: {question}")
        relevant, has_context = self.vector_manager.search_with_guardrail(question, k, min_similarity)
        if not has_context:
            ans = "No relevant context found in the knowledge base."
            print(f"ANSWER:\n{ans}")
            return ans

        context, sources = self.vector_manager.format_context(relevant)
        print(f"RETRIEVED SOURCES: {', '.join(sources)}")
        ans = self.llm_generator.generate_answer(question, context)
        print(f"ANSWER:\n{ans}")
        return ans

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.ingest()
    for q in [
        "What is the capital city of Australia and what is its annual GDP growth rate?",
        "What is an LLM and How Does RAG Make It Useful?"
    ]:
        pipeline.ask(q)
