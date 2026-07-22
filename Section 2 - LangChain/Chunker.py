from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, docs):
        chunks = self.splitter.split_documents(docs)
        counters = {}
        for c in chunks:
            fn = c.metadata.get("file_name", "doc")
            counters[fn] = counters.get(fn, 0) + 1
            c.metadata["chunk_id"] = counters[fn]
        return chunks
