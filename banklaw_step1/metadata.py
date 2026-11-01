def add_metadata(
    section: dict[str, str],
    document: str,
    topic: str,
    category: str | None = None,
    source: str | None = None,
    document_type: str | None = None,
) -> dict[str, str]:
    """
    Add source and corpus metadata to a legal section
    without modifying its content.
    """
    result = section.copy()

    result["document"] = document
    result["topic"] = topic
    result["section_number"] = section["number"]
    result["section_level"] = section["level"]
    result["section_parent"] = section["parent"]

    if category is not None:
        result["category"] = category

    if source is not None:
        result["source"] = source

    if document_type is not None:
        result["document_type"] = document_type

    return result