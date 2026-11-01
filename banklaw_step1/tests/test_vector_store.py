import os
import subprocess
import sys

import chromadb
import pytest

from banklaw_step1 import vector_store


@pytest.fixture(autouse=True)
def isolated_vector_store(tmp_path, monkeypatch):
    test_chroma_path = tmp_path / "chroma"

    test_client = chromadb.PersistentClient(
        path=str(test_chroma_path)
    )

    monkeypatch.setattr(vector_store, "_client", test_client)

    monkeypatch.setenv(
        "BANKLAW_TEST_CHROMA_PATH",
        str(test_chroma_path),
    )


def test_add_documents_uses_bge_m3_embeddings():
    documents = [
        {
            "page_content": "حقوق عملاء البنوك",
            "metadata": {
                "document": "customer_rights.pdf",
                "section_number": "4-1",
            },
        },
        {
            "page_content": "إجراءات تقديم الشكاوى",
            "metadata": {
                "document": "complaints.pdf",
                "section_number": "2-1",
            },
        },
    ]

    vector_store.add_documents(documents)

    collection = vector_store.get_collection()
    result = collection.get(include=["embeddings"])

    embeddings = result["embeddings"]

    assert len(embeddings) == 2
    assert all(len(vector) == 1024 for vector in embeddings)


def test_query_documents_returns_relevant_legal_chunk():
    documents = [
        {
            "page_content": "يتعين على البنك الإفصاح عن البيانات.",
            "metadata": {
                "document": "customer_rights.pdf",
                "section_number": "4-1",
            },
        },
        {
            "page_content": "يحق للعميل تقديم شكوى إلى البنك.",
            "metadata": {
                "document": "complaints.pdf",
                "section_number": "2-1",
            },
        },
    ]

    vector_store.add_documents(documents)

    results = vector_store.query_documents("ما هي حقوق العميل؟")

    assert len(results) > 0
    assert any(
        result["metadata"]["document"] == "complaints.pdf"
        for result in results
    )


def test_vector_store_persists_documents():
    documents = [
        {
            "page_content": "حقوق عملاء البنوك",
            "metadata": {
                "document": "customer_rights.pdf",
                "section_number": "4-1",
            },
        }
    ]

    vector_store.add_documents(documents)

    collection = vector_store.get_collection()

    result = collection.get(
        include=["documents", "metadatas"]
    )

    assert "حقوق عملاء البنوك" in result["documents"]
    assert result["metadatas"][0]["document"] == "customer_rights.pdf"


def test_vector_store_survives_process_restart(monkeypatch):
    chroma_path = os.environ["BANKLAW_TEST_CHROMA_PATH"]

    add_script = """
import os
import chromadb
from banklaw_step1.embeddings import embed_texts

path = os.environ["BANKLAW_TEST_CHROMA_PATH"]

client = chromadb.PersistentClient(path=path)
collection = client.get_or_create_collection(name="banklaw_documents")

text = "حقوق عملاء البنوك"
embedding = embed_texts([text])[0]

collection.upsert(
    ids=["restart-test"],
    documents=[text],
    metadatas=[{
        "document": "customer_rights.pdf",
        "section_number": "4-1",
    }],
    embeddings=[embedding],
)
"""

    query_script = """
import os
import chromadb
from banklaw_step1.embeddings import embed_texts

path = os.environ["BANKLAW_TEST_CHROMA_PATH"]

client = chromadb.PersistentClient(path=path)
collection = client.get_or_create_collection(name="banklaw_documents")

query_embedding = embed_texts(["حقوق عملاء البنوك"])[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1,
    include=["documents", "metadatas"],
)

assert len(results["documents"][0]) > 0
assert results["metadatas"][0][0]["document"] == "customer_rights.pdf"
"""

    monkeypatch.setenv(
        "BANKLAW_TEST_CHROMA_PATH",
        chroma_path,
    )

    subprocess.run(
        [sys.executable, "-c", add_script],
        check=True,
    )

    subprocess.run(
        [sys.executable, "-c", query_script],
        check=True,
    )


def test_delete_collection_removes_all_documents():
    documents = [
        {
            "page_content": "اختبار بيانات Chroma",
            "metadata": {
                "document": "test.pdf",
                "section_number": "1",
            },
        }
    ]

    vector_store.add_documents(documents)

    assert vector_store.get_collection().count() > 0

    vector_store.delete_collection()

    assert vector_store.get_collection().count() == 0