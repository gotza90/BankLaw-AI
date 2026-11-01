from banklaw_step1.chunking import chunk_documents


def test_chunk_documents_preserves_legal_metadata():
    documents = [
        {
            "page_content": (
                "4-1 التعامل مع العملاء.\n"
                "يتعين على البنك الإفصاح عن البيانات."
            ),
            "metadata": {
                "document": "customer_rights.pdf",
                "topic": "حماية حقوق العملاء",
                "section_number": "4-1",
                "section_level": 2,
                "section_parent": "4",
            },
        }
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) >= 1

    chunk = chunks[0]

    assert "يتعين على البنك الإفصاح عن البيانات." in chunk["page_content"]

    assert chunk["metadata"]["document"] == "customer_rights.pdf"
    assert chunk["metadata"]["topic"] == "حماية حقوق العملاء"
    assert chunk["metadata"]["section_number"] == "4-1"
    assert chunk["metadata"]["section_level"] == 2
    assert chunk["metadata"]["section_parent"] == "4"

def test_chunk_documents_splits_long_text():
    documents = [
        {
            "page_content": "أ" * 1200,
            "metadata": {
                "document": "customer_rights.pdf",
                "topic": "حماية حقوق العملاء",
                "section_number": "4-1",
                "section_level": 2,
                "section_parent": "4",
            },
        }
    ]

    chunks = chunk_documents(documents, chunk_size=500)

    assert len(chunks) == 3

    assert all(len(chunk["page_content"]) <= 500 for chunk in chunks)

    assert all(
        chunk["metadata"]["section_number"] == "4-1"
        for chunk in chunks
    )

def test_chunk_documents_supports_overlap():
    documents = [
        {
            "page_content": "0123456789" * 100,
            "metadata": {
                "document": "customer_rights.pdf",
                "topic": "حماية حقوق العملاء",
                "section_number": "4-1",
                "section_level": 2,
                "section_parent": "4",
            },
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 13

    assert chunks[0]["page_content"][-20:] == chunks[1]["page_content"][:20]


def test_chunk_documents_preserves_legal_hierarchy_context():
    documents = [
        {
            "page_content": (
                "4-1 التعامل مع العملاء\n"
                "يتعين على البنك الإفصاح عن البيانات."
            ),
            "metadata": {
                "document": "customer_rights.pdf",
                "topic": "حماية حقوق العملاء",
                "section_number": "4-1",
                "section_level": 2,
                "section_parent": "4",
                "category": "customer_protection",
                "source": "Central Bank of Egypt",
                "document_type": "customer_protection_instructions",
            },
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        chunk_overlap=0,
    )

    assert len(chunks) >= 1

    for chunk in chunks:
        metadata = chunk["metadata"]

        assert metadata["section_number"] == "4-1"
        assert metadata["section_level"] == 2
        assert metadata["section_parent"] == "4"
        assert metadata["category"] == "customer_protection"
        assert metadata["source"] == "Central Bank of Egypt"


def test_chunk_documents_adds_chunk_position_metadata():
    documents = [
        {
            "page_content": "أ" * 120,
            "metadata": {
                "document": "customer_rights.pdf",
                "topic": "حماية حقوق العملاء",
                "section_number": "4-1",
                "section_level": 2,
                "section_parent": "4",
            },
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        chunk_overlap=0,
    )

    assert len(chunks) == 3

    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["metadata"]["chunk_index"] == 1
    assert chunks[2]["metadata"]["chunk_index"] == 2

    assert chunks[0]["metadata"]["chunk_count"] == 3
    assert chunks[1]["metadata"]["chunk_count"] == 3
    assert chunks[2]["metadata"]["chunk_count"] == 3

def test_chunk_documents_does_not_split_legal_sentence():
    documents = [
        {
            "page_content": (
                "6. قواعد معالجة الشكاوى\n"
                "يتعين على البنك فور استلام الشكوى القيام بالآتي:\n"
                "فحص الشكوى بكفاءة وجدية وحيادية وبذل العناية الواجبة في معالجتها.\n"
                "الحصول على أية معلومات إضافية ضرورية لفحص الشكوى."
            ),
            "metadata": {
                "document": "complaints.pdf",
                "section_number": "1",
            },
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        chunk_overlap=0,
    )

    combined_text = "".join(chunk["page_content"] for chunk in chunks)

    assert "فحص الشكوى بكفاءة وجدية وحيادية وبذل العناية الواجبة في معالجتها." in combined_text
