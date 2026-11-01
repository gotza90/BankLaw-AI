from pathlib import Path

import chromadb

from banklaw_step1.embeddings import embed_texts


PERSIST_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "chroma"
)

_client = chromadb.PersistentClient(
    path=str(PERSIST_DIRECTORY)
)


def add_documents(documents: list[dict]) -> int:
    collection = _client.get_or_create_collection(
        name="banklaw_documents"
    )

    texts = [document["page_content"] for document in documents]
    metadatas = [document["metadata"] for document in documents]

    # Deterministic unique ids: document name + section + chunk index.
    # Reusing sequential ids would overwrite existing chunks on incremental
    # indexing, so each chunk must be addressable independently.
    ids = []
    for document in documents:
        metadata = document["metadata"]
        ids.append(
            f"{metadata.get('document', 'unknown')}|"
            f"{metadata.get('section_number', 'unknown')}|"
            f"{metadata.get('chunk_index', 0)}|"
            f"{len(ids)}"
        )

    embeddings = embed_texts(texts)

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(documents)


def get_collection():
    return _client.get_or_create_collection(
        name="banklaw_documents"
    )


def delete_collection() -> None:
    _client.delete_collection(name="banklaw_documents")

def query_documents(query: str, n_results: int = 3) -> list[dict]:
    collection = get_collection()

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "page_content": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
    ]