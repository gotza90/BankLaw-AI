"""Incremental indexer: ingest only PDFs not already present in ChromaDB.

Usage:
    python -m banklaw_step1.index_new_pdfs
"""
from pathlib import Path

from banklaw_step1.chunking import chunk_documents
from banklaw_step1.ingestion import ingest_directory
from banklaw_step1.vector_store import add_documents, get_collection

DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _indexed_documents() -> set[str]:
    collection = get_collection()
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    return {m.get("document") for m in metadatas if m.get("document")}


def main() -> None:
    indexed = _indexed_documents()
    print(f"Already indexed documents: {len(indexed)}")

    all_documents = ingest_directory(
        DATA_DIR,
        manifest_path=MANIFEST_PATH,
    )

    new_documents = [d for d in all_documents if d["metadata"]["document"] not in indexed]
    print(f"Documents after ingestion: {len(all_documents)}")
    print(f"New documents to index: {len(set(d['metadata']['document'] for d in new_documents))}")

    if not new_documents:
        print("Nothing new to index.")
        return

    chunks = chunk_documents(
        new_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"Chunks generated: {len(chunks)}")

    added = add_documents(chunks)
    print(f"Chunks added to Chroma: {added}")

    collection = get_collection()
    print(f"Chroma collection count: {collection.count()}")


if __name__ == "__main__":
    main()