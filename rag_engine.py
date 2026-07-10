import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai

GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-2"
GOOGLE_LLM_MODEL = "models/gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
INDEX_DIR = "faiss_index"


class RAGEngine:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(GOOGLE_LLM_MODEL)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[str] = []

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def load_pdf(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    def chunk_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #

    def _embed(self, text: str, task_type: str) -> np.ndarray:
        result = genai.embed_content(
            model=GOOGLE_EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        return np.array(result["embedding"], dtype=np.float32)

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(text, task_type="retrieval_document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text, task_type="retrieval_query")

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def build_index(self, chunks: list[str]) -> None:
        self.chunks = chunks

        embeddings = np.vstack([self.embed_document(c) for c in self.chunks])
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def save_index(self, save_dir: str = INDEX_DIR) -> None:
        Path(save_dir).mkdir(exist_ok=True)
        faiss.write_index(self.index, f"{save_dir}/index.faiss")
        with open(f"{save_dir}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self, load_dir: str = INDEX_DIR) -> bool:
        index_path = Path(load_dir) / "index.faiss"
        chunks_path = Path(load_dir) / "chunks.pkl"
        if not (index_path.exists() and chunks_path.exists()):
            return False
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        return True

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        q = self.embed_query(query).reshape(1, -1)
        faiss.normalize_L2(q)
        _, indices = self.index.search(q, top_k)
        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate_answer(
        self,
        query: str,
        context_chunks: list[str],
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        context = "\n\n---\n\n".join(context_chunks)

        history_block = ""
        if chat_history:
            lines = []
            for msg in chat_history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_block = "\nPrevious conversation:\n" + "\n".join(lines) + "\n"

        prompt = (
            "You are a helpful HR assistant for Flairminds Software. "
            "Answer questions strictly based on the leave policy context provided. "
            "If the answer is not in the context, say so clearly.\n\n"
            f"Leave Policy Context:\n{context}"
            f"{history_block}\n"
            f"User: {query}\nAssistant:"
        )

        response = self.llm.generate_content(prompt)
        return response.text.strip()

    # ------------------------------------------------------------------ #
    # Main entry point — call this from Streamlit or FastAPI
    # ------------------------------------------------------------------ #

    def chat(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
    ) -> str:
        if not self.is_ready():
            raise RuntimeError("Index not loaded. Call build_index() or load_index() first.")
        context_chunks = self.retrieve(query)
        print("context_chunks: ", context_chunks)
        return self.generate_answer(query, context_chunks, chat_history)
