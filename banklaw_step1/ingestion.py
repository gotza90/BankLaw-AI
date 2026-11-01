import json
from pathlib import Path

from banklaw_step1.clean_text import clean_text
from banklaw_step1.documents import build_documents
from banklaw_step1.extract_text import extract_pdf_text
from banklaw_step1.metadata import add_metadata
from banklaw_step1.structure import extract_sections


def ingest_directory(
    directory: str | Path,
    topic: str | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict]:
    """
    Ingest all PDFs in a directory through the legal document pipeline.
    """

    directory = Path(directory)

    manifest = {}

    if manifest_path is not None:
        manifest_file = Path(manifest_path)

        with manifest_file.open("r", encoding="utf-8-sig") as file:
            manifest_data = json.load(file)

        manifest = {
            item["filename"]: item
            for item in manifest_data
        }

    documents = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        raw_text = extract_pdf_text(pdf_path)
        cleaned_text = clean_text(raw_text)
        sections = extract_sections(cleaned_text)

        corpus_metadata = manifest.get(pdf_path.name, {})

        sections = [
            add_metadata(
                section,
                document=pdf_path.name,
                topic=topic or "",
                category=corpus_metadata.get("category"),
                source=corpus_metadata.get("source"),
                document_type=corpus_metadata.get("document_type"),
            )
            for section in sections
        ]

        documents.extend(build_documents(sections))

    return documents