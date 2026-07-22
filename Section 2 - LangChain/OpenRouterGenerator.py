import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(Path(__file__).parent / ".env")

class OpenRouterGenerator:
    def __init__(self, temperature: float = 0.1):
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip(' "')
        models = ["cohere/north-mini-code:free", "google/gemini-2.5-flash:free", "meta-llama/llama-3-8b-instruct:free", "openrouter/auto"]
        self.model = None
        for m in models:
            try:
                self.model = ChatOpenAI(
                    model=m,
                    temperature=temperature,
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    timeout=20
                )
                self.model.invoke("Hi")
                print(f"[INFO] Using LLM Model: {m}")
                break
            except Exception:
                pass
        if not self.model:
            raise RuntimeError("No LLMs available on OpenRouter.")

    def generate_answer(self, question: str, context: str) -> str:
        template = (
            "You are a strict technical assistant. Answer the user's question using ONLY the provided context chunks.\n\n"
            "Rules:\n"
            "1. Every claim MUST cite its source in the format [Source: <filename>, Chunk: <id>].\n"
            "2. If the context is insufficient, state: 'No relevant context found in the knowledge base.' and do not guess.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = ChatPromptTemplate.from_template(template) | self.model | StrOutputParser()
        ans = chain.invoke({"context": context, "question": question})
        return ans.split("</think>")[-1].strip() if "</think>" in ans else ans
