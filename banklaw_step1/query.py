"""
Minimal end-to-end query entry point for banklaw-ai.

ask(question) runs the full existing RAG pipeline:

    retrieval  -> vector_store.query_documents
    reranking  -> reranker.rerank_documents
    context    -> context.build_context
    prompt+LLM -> llm.generate_answer

It is deliberately a thin orchestrator: it imports and calls the existing
modules and does not reimplement any of them.
"""

from banklaw_step1.vector_store import query_documents
from banklaw_step1.reranker import rerank_documents
from banklaw_step1.context import build_context
from banklaw_step1.llm import generate_answer
from banklaw_step1.query_understanding import analyze_question, rewrite_queries


DEFAULT_N_RESULTS = 10
DEFAULT_TOP_K = 3


def ask(question, n_results=DEFAULT_N_RESULTS, top_k=DEFAULT_TOP_K,
        conversation_context=None):
    """
    Answer a natural-language question against the indexed CBE banking-law corpus.

    Returns a dict with:
      - question: the question that was asked
      - answer:   the LLM-generated answer with [Citation: ...] tags
      - context:  the citation-aware context the answer was grounded in
      - analysis: the query-understanding result (intent, topic, ...)
      - retrieval_queries: the semantic queries used for retrieval
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    # 1. Understand the question and rewrite it into semantic retrieval queries.
    analysis = analyze_question(question, conversation_context)
    queries = rewrite_queries(analysis) or [question]

    # 2. Retrieve against each reformulated query, then merge/dedup.
    retrieved = []
    for query in queries:
        retrieved.extend(query_documents(query, n_results=n_results))
    retrieved = _merge_documents(retrieved)

    # 3. Re-rank with the existing RRF reranker, driven by the ORIGINAL question.
    reranked = rerank_documents(question, retrieved, top_k=top_k)

    # 4. Citation-aware context.
    context = build_context(reranked)

    # 5. Prompt construction + Ollama generation, using the user's ORIGINAL wording.
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "analysis": analysis,
        "retrieval_queries": queries,
    }


def _merge_documents(documents):
    """De-duplicate retrieved documents by content signature.

    Different reformulated queries may retrieve the same chunk; we keep only
    one copy of each, retaining the best (smallest) distance. The existing
    reranker still runs on the ORIGINAL question after this merge.
    """
    best: dict[str, dict] = {}
    for document in documents:
        key = _doc_key(document)
        existing = best.get(key)
        if existing is None or float(document.get("distance", 1.0)) < float(
            existing.get("distance", 1.0)
        ):
            best[key] = document
    return list(best.values())


def _doc_key(document):
    metadata = document.get("metadata", {}) or {}
    return (
        metadata.get("document"),
        metadata.get("section_number"),
        metadata.get("chunk_index"),
        _normalize(document.get("page_content", "")),
    )


def _normalize(text):
    """Collapse whitespace so identical chunks match across queries."""
    return " ".join(str(text or "").split())
