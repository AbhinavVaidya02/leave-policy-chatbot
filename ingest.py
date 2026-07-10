"""
Offline indexing script — run this once before starting the app,
and again whenever the source PDF changes.

Usage:
    python ingest.py
"""
import os
import sys
from dotenv import load_dotenv
from rag_engine import RAGEngine, INDEX_DIR

load_dotenv()

PDF_PATH = "Leave Policy.pdf"


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    engine = RAGEngine(api_key=api_key)

    print(f"[1/4] Loading PDF: {PDF_PATH}")
    text = engine.load_pdf(PDF_PATH)
    print(f"      {len(text):,} characters loaded.")

    print("[2/4] Splitting into chunks...")
    chunks = engine.chunk_text(text)
    print(f"      {len(chunks)} chunks created.")

    print("[3/4] Generating embeddings and building FAISS index...")
    engine.build_index(chunks)

    print(f"[4/4] Saving index to '{INDEX_DIR}/'...")
    engine.save_index(INDEX_DIR)

    print(f"\nDone. {len(chunks)} chunks indexed. You can now run the app with:")
    print("    streamlit run main.py")


if __name__ == "__main__":
    main()
