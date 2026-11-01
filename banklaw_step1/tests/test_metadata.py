from banklaw_step1.metadata import add_metadata


def test_metadata_contains_retrieval_fields():
    section = {
        "number": "4-1-8",
        "level": 3,
        "parent": "4-1",
        "text": "يتعين على البنك الإفصاح عن البيانات.",
    }

    result = add_metadata(
        section,
        document="customer_rights.pdf",
        topic="حماية حقوق العملاء",
    )

    assert result["document"] == "customer_rights.pdf"
    assert result["topic"] == "حماية حقوق العملاء"

    assert result["section_number"] == "4-1-8"
    assert result["section_level"] == 3
    assert result["section_parent"] == "4-1"

def test_metadata_contains_corpus_fields():
    section = {
        "number": "4-1-8",
        "level": 3,
        "parent": "4-1",
        "text": "يتعين على البنك الإفصاح عن البيانات.",
    }

    result = add_metadata(
        section,
        document="customer_rights.pdf",
        topic="حماية حقوق العملاء",
        category="customer_protection",
        source="Central Bank of Egypt",
        document_type="customer_protection_instructions",
    )

    assert result["category"] == "customer_protection"
    assert result["source"] == "Central Bank of Egypt"
    assert result["document_type"] == "customer_protection_instructions"