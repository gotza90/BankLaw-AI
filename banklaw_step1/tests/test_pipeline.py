from banklaw_step1.extract_text import extract_pdf_text
from banklaw_step1.clean_text import clean_text
from banklaw_step1.structure import extract_sections
from banklaw_step1.metadata import add_metadata


def test_build_structured_document():
    raw = """
4- قواعد حماية حقوق عملاء البنوك
4-1 التعامل مع العملاء
4-1-8 العقود والنماذج
"""

    cleaned = clean_text(raw)
    sections = extract_sections(cleaned)

    structured = [
        add_metadata(
            section,
            document="customer_rights.pdf",
            topic="حماية حقوق العملاء",
        )
        for section in sections
    ]

    assert len(structured) == 3

    assert structured[0]["number"] == "4"
    assert structured[0]["level"] == 1
    assert structured[0]["parent"] is None

    assert structured[1]["number"] == "4-1"
    assert structured[1]["level"] == 2
    assert structured[1]["parent"] == "4"

    assert structured[2]["number"] == "4-1-8"
    assert structured[2]["level"] == 3
    assert structured[2]["parent"] == "4-1"

    assert structured[2]["document"] == "customer_rights.pdf"
    assert structured[2]["topic"] == "حماية حقوق العملاء"