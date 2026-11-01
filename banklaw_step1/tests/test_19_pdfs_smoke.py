from pathlib import Path

from pypdf import PdfReader

from banklaw_step1.clean_text import clean_text
from banklaw_step1.extract_text import extract_pdf_text
from banklaw_step1.metadata import add_metadata
from banklaw_step1.structure import extract_sections


DATA_DIR = Path("data")


def main() -> None:
    pdfs = sorted(DATA_DIR.glob("*.pdf"))

    print(f"PDF count = {len(pdfs)}")
    print("=" * 100)

    if len(pdfs) != 19:
        raise AssertionError(
            f"Expected 19 PDFs, found {len(pdfs)}"
        )

    passed = 0

    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"\n[{index}/19] {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        raw_text = extract_pdf_text(pdf_path)

        if not raw_text.strip():
            raise AssertionError("Raw extraction returned empty text")

        cleaned_text = clean_text(raw_text)

        if not cleaned_text.strip():
            raise AssertionError("Cleaning produced empty text")

        sections = extract_sections(cleaned_text)

        if not sections:
            raise AssertionError("Structure extraction returned 0 sections")

        enriched_sections = [
            add_metadata(
                section,
                document=pdf_path.name,
                topic="",
                category=None,
                source="Central Bank of Egypt",
                document_type=(
                    "law"
                    if pdf_path.name.startswith("CBE Law")
                    else None
                ),
            )
            for section in sections
        ]

        print(f"  pages       = {len(reader.pages)}")
        print(f"  raw_chars   = {len(raw_text):,}")
        print(f"  clean_chars = {len(cleaned_text):,}")
        print(f"  sections    = {len(sections):,}")
        print(f"  metadata    = {len(enriched_sections):,}")
        print("  STATUS      = PASS")

        passed += 1

    print("\n" + "=" * 100)
    print(f"RESULT: {passed}/19 PDFs passed")


if __name__ == "__main__":
    main()