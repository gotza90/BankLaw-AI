from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_model = CrossEncoder(MODEL_NAME)


def _min_max_normalize(values):
    """
    Min-max normalize a list of floats into [0, 1].

    A constant list (all values equal) maps to 0.5 for every element so the
    signal stays neutral instead of collapsing to zero.
    """
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [0.5 for _ in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def rerank_documents(
    query,
    documents,
    top_k=3,
    semantic_weight=1.0,
    crossencoder_weight=1.0,
):
    """
    Re-rank documents by fusing the semantic retrieval signal with the
    CrossEncoder signal on a normalized scale.

    Why fusion instead of CrossEncoder alone
    -----------------------------------------
    The CrossEncoder (bge-reranker-v2-m3) scores are tightly clustered near
    the top of their range. On Arabic legal text a specialized section that
    shares many keywords with the query - for example a section about
    systemic / repeated complaints - can score marginally higher than the
    general rule that actually answers the question. Ranking on the raw
    CrossEncoder score alone therefore lets a keyword-rich specialized
    section displace a clearly-relevant general rule, even though the
    semantic retrieval stage (ChromaDB + BGE-M3) ranked the general rule
    higher.

    The semantic retrieval stage already returns the documents sorted by
    cosine distance (best first), and its distances are on a stable,
    comparable scale. The CrossEncoder scores are on an unbounded logit
    scale whose spread varies from batch to batch, so comparing a raw CE
    score against a raw semantic distance is not meaningful.

    Both signals are therefore min-max normalized within the batch before
    being fused:

        fused = semantic_weight * norm(semantic_similarity)
              + crossencoder_weight * norm(crossencoder_score)

    where semantic_similarity = 1 - cosine_distance. Normalization puts
    the two signals on equal footing: a large CrossEncoder margin can still
    reorder results, but a small one cannot override a clear semantic
    preference. This is what prevents a specialized section from winning
    over the general rule for a general question, while still letting the
    CrossEncoder refine the order when it has a genuine, large preference.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if not documents:
        return []

    # 1. CrossEncoder relevance scores. Document title and section number
    #    are included in the passage so the model can use them as signals.
    pairs = [
        (
            query,
            (
                f"Document: {document['metadata'].get('document', '')}\n"
                f"Section: {document['metadata'].get('section_number', '')}\n"
                f"Text: {document['page_content']}"
            ),
        )
        for document in documents
    ]

    ce_scores = _model.predict(pairs)

    # 2. Semantic similarity. ChromaDB returns cosine distance, so
    #    similarity = 1 - distance (higher is better).
    semantic_similarities = [
        1.0 - float(document.get("distance", 0.0))
        for document in documents
    ]

    # 3. Normalize both signals within the batch so they are comparable.
    norm_semantic = _min_max_normalize(semantic_similarities)
    norm_ce = _min_max_normalize([float(score) for score in ce_scores])

    # 4. Fuse and rank.
    fused = []
    for index, document in enumerate(documents):
        fused_score = (
            semantic_weight * norm_semantic[index]
            + crossencoder_weight * norm_ce[index]
        )

        result = document.copy()
        result["score"] = float(fused_score)
        result["ce_score"] = float(ce_scores[index])
        result["semantic_similarity"] = semantic_similarities[index]
        fused.append(result)

    fused.sort(key=lambda item: item["score"], reverse=True)

    return fused[:top_k]