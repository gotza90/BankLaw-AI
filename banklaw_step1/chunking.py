def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[dict]:
    """
    Split documents into overlapping chunks while preserving metadata
    and recording each chunk's position within its source document.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap

    for document in documents:
        text = document["page_content"]
        metadata = document["metadata"]

        document_chunks = []

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            document_chunks.append(
                {
                    "page_content": chunk_text,
                    "metadata": metadata.copy(),
                }
            )

            if start + chunk_size >= len(text):
                break

        chunk_count = len(document_chunks)

        for chunk_index, chunk in enumerate(document_chunks):
            chunk["metadata"]["chunk_index"] = chunk_index
            chunk["metadata"]["chunk_count"] = chunk_count
            chunks.append(chunk)

    return chunks