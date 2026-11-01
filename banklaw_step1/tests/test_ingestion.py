from pathlib import Path

from banklaw_step1.ingestion import ingest_directory


SOURCE_PDF = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "الفصل الأول - الباب الرابع عشر - حقوق العملاء.pdf"
)


def test_ingest_directory_processes_all_pdfs(tmp_path):
    (tmp_path / "law_a.pdf").write_bytes(SOURCE_PDF.read_bytes())
    (tmp_path / "law_b.pdf").write_bytes(SOURCE_PDF.read_bytes())

    documents = ingest_directory(tmp_path)

    assert len(documents) > 2
    assert documents[0]["metadata"]["document"] == "law_a.pdf"
    assert any(
        document["metadata"]["document"] == "law_b.pdf"
        for document in documents
    )


def test_ingest_directory_extracts_pdf_content(tmp_path):
    target_pdf = tmp_path / "customer_rights.pdf"
    target_pdf.write_bytes(SOURCE_PDF.read_bytes())

    documents = ingest_directory(
        tmp_path,
        topic="حماية حقوق العملاء",
    )

    assert len(documents) > 0

    document = documents[0]

    assert document["page_content"]
    assert len(document["page_content"]) > 100

    assert document["metadata"]["document"] == "customer_rights.pdf"
    assert document["metadata"]["topic"] == "حماية حقوق العملاء"

def test_ingest_directory_uses_manifest_metadata(tmp_path):
    source_pdf = SOURCE_PDF
    target_pdf = tmp_path / source_pdf.name
    target_pdf.write_bytes(source_pdf.read_bytes())

    manifest = [
        {
            "filename": source_pdf.name,
            "category": "customer_protection",
            "source": "Central Bank of Egypt",
            "document_type": "customer_protection_instructions",
        }
    ]

    import json

    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    documents = ingest_directory(
        tmp_path,
        manifest_path=manifest_path,
    )

    assert len(documents) > 0

    metadata = documents[0]["metadata"]

    assert metadata["category"] == "customer_protection"
    assert metadata["source"] == "Central Bank of Egypt"
    assert metadata["document_type"] == "customer_protection_instructions"


def test_ingest_real_corpus_processes_all_19_pdfs():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    manifest_path = data_dir / "corpus_manifest.json"

    documents = ingest_directory(
        data_dir,
        manifest_path=manifest_path,
    )

    source_documents = {
        document["metadata"]["document"]
        for document in documents
    }

    assert len(source_documents) == 19

    assert all(
        document["page_content"].strip()
        for document in documents
    )

    assert all(
        document["metadata"]["source"] == "Central Bank of Egypt"
        for document in documents
    )
