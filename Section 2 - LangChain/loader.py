import os
import glob
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader

class Loader:
    def __init__(self, target_path: str):
        self.target_path = Path(target_path)

    def load(self):
        docs = []
        if self.target_path.is_file():
            docs.extend(self._load_file(str(self.target_path)))
        elif self.target_path.is_dir():
            files = []
            for ext in ("*.md", "*.txt", "*.pdf"):
                files.extend(glob.glob(str(self.target_path / ext)))
            for f in sorted(files):
                docs.extend(self._load_file(f))
        return docs

    def _load_file(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext not in (".pdf", ".txt", ".md"):
            return []
        loader = PyPDFLoader(file_path) if ext == ".pdf" else TextLoader(file_path, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata.update({"file_name": os.path.basename(file_path), "source": file_path})
        return loaded
