"""
index_docs.py — Run this ONLY when you add new documents to claude/docs/

Usage:
    cd rag_pipeline
    python index_docs.py
"""
from __future__ import annotations
import logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import RAGPipeline, PipelineConfig
from document_loader import DocumentLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

HERE      = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR  = os.path.join(HERE, "..","..", "data", "raw")
INDEX_DIR = os.path.join(HERE, "..","..", "data", "vector_store")

def main():
    print("\n" + "="*60)
    print("  Indexing Documents")
    print("="*60 + "\n")

    cfg = PipelineConfig(
        vector_backend     = "faiss",
        vector_persist_dir = INDEX_DIR,
        llm_backend        = "ollama",
    )
    pipeline = RAGPipeline.from_config(cfg)

    print(f"Scanning: {os.path.abspath(DOCS_DIR)}\n")
    loader    = DocumentLoader(docs_dir=DOCS_DIR, recursive=True)
    documents = loader.load_all()

    if not documents:
        print("No documents found in docs/ folder.")
        return

    print(f"Indexing {len(documents)} document(s)...\n")
    for doc in documents:
        # Delete old index for this doc first (re-index cleanly)
        pipeline.delete_document(doc["doc_id"])
        n = pipeline.index_pages(
            pages       = doc["pages"],
            doc_id      = doc["doc_id"],
            source_path = doc["source_path"],
        )
        print(f"  {doc['source_path']}  -> {n} chunks")

    # Save index to disk
    pipeline.store.persist()
    print(f"\nTotal chunks saved: {pipeline.stats()['total_indexed_chunks']}")
    print("\nDone! Now run chat.py to ask questions.")

if __name__ == "__main__":
    main()