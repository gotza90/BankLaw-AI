def build_documents(sections: list[dict]) -> list[dict]:
    """
    Convert structured legal sections into retrieval-ready documents.
    """

    documents = []

    for section in sections:
        document = {
            "page_content": section["text"],
            "metadata": {
                "document": section["document"],
                "topic": section["topic"],
                "section_number": section["number"],
                "section_level": section["level"],
                "section_parent": section["parent"],
                "category": section.get("category"),
                "source": section.get("source"),
                "document_type": section.get("document_type"),
            },
        }

        documents.append(document)

    return documents