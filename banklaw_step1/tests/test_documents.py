from banklaw_step1.documents import build_documents


def test_build_documents_from_sections():
    sections = [
        {
            "number": "4-1",
            "level": 2,
            "parent": "4",
            "text": "يتعين على البنك الإفصاح عن البيانات.",
            "document": "customer_rights.pdf",
            "topic": "حماية حقوق العملاء",
        }
    ]

    documents = build_documents(sections)

    assert len(documents) == 1

    doc = documents[0]

    assert doc["page_content"] == "يتعين على البنك الإفصاح عن البيانات."

    assert doc["metadata"]["document"] == "customer_rights.pdf"
    assert doc["metadata"]["topic"] == "حماية حقوق العملاء"
    assert doc["metadata"]["section_number"] == "4-1"
    assert doc["metadata"]["section_level"] == 2
    assert doc["metadata"]["section_parent"] == "4"