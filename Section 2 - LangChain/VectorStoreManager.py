from langchain_chroma import Chroma

class VectorStoreManager:
    def __init__(self, embedding_model, collection_name="domain_rag", persist_directory="chroma_db_rag"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def create(self, chunks):
        if not chunks:
            return 0
        try:
            self.vectorstore.delete_collection()
        except Exception:
            pass
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return len(chunks)

    def search_with_guardrail(self, query: str, k: int = 3, min_similarity: float = 0.35):
        if not query:
            return [], False
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Cosine distance similarity: 1.0 - distance
        relevant = [doc for doc, dist in results if (1.0 - dist) >= min_similarity]
        return relevant, len(relevant) > 0

    def format_context(self, chunks):
        sources = [f"[Source: {c.metadata.get('file_name', 'unknown')}, Chunk: {c.metadata.get('chunk_id', 0)}]" for c in chunks]
        context = "\n\n".join(f"{src}\n{c.page_content}" for src, c in zip(sources, chunks))
        return context, sources
