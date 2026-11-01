import re


def recall_at_k(results, expected_document, k):
    top_results = results[:k]

    return (
        1.0
        if any(
            result["metadata"].get("document") == expected_document
            for result in top_results
        )
        else 0.0
    )


def reciprocal_rank(results, expected_document):
    for index, result in enumerate(results, start=1):
        if result["metadata"].get("document") == expected_document:
            return 1.0 / index

    return 0.0


def _normalize_arabic_text(text: str) -> list[str]:
    text = text.lower()

    # Remove punctuation while keeping Arabic letters/numbers.
    text = re.sub(r"[^\w\u0600-\u06ff]+", " ", text)

    tokens = [
        token.strip()
        for token in text.split()
        if len(token.strip()) >= 2
    ]

    return tokens


def grounding_score(answer: str, context: list[dict]) -> float:
    """
    Baseline grounding metric based on token overlap.

    Returns a value between 0.0 and 1.0:
    - 1.0 = all answer tokens are found in the retrieved context
    - 0.0 = no meaningful answer tokens are found in the context
    """

    if not answer.strip() or not context:
        return 0.0

    context_text = " ".join(
        item["text"]
        for item in context
        if item.get("text")
    )

    answer_tokens = set(_normalize_arabic_text(answer))
    context_tokens = set(_normalize_arabic_text(context_text))

    if not answer_tokens:
        return 0.0

    grounded_tokens = answer_tokens & context_tokens

    return len(grounded_tokens) / len(answer_tokens)


def claim_grounding_score(claim: str, context: list[dict]) -> float:
    """
    Measures whether a claim is directly supported by the retrieved context.

    Returns:
    - 1.0 when all meaningful claim tokens are present in context.
    - 0.0 when the claim has no meaningful overlap with context.
    """

    if not claim.strip() or not context:
        return 0.0

    context_text = " ".join(
        item["text"]
        for item in context
        if item.get("text")
    )

    claim_tokens = set(_normalize_arabic_text(claim))
    context_tokens = set(_normalize_arabic_text(context_text))

    if not claim_tokens:
        return 0.0

    grounded_tokens = claim_tokens & context_tokens

    return len(grounded_tokens) / len(claim_tokens)