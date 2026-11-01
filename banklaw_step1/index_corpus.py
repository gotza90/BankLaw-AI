from pathlib import Path

from banklaw_step1.chunking import chunk_documents
from banklaw_step1.ingestion import ingest_directory
from banklaw_step1.vector_store import add_documents, get_collection


DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main() -> None:
    print("Starting corpus indexing...")

    documents = ingest_directory(
        DATA_DIR,
        manifest_path=MANIFEST_PATH,
    )

    print(f"Documents after ingestion: {len(documents)}")

    chunks = chunk_documents(
        documents,
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