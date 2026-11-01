from banklaw_step1.vector_store import query_documents
from banklaw_step1.reranker import rerank_documents
from banklaw_step1.context import build_context
from banklaw_step1.llm import generate_answer


def test_real_rag_generates_answer_from_retrieved_source():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    # 1. Semantic retrieval
    retrieved = query_documents(question, n_results=10)

    assert retrieved
    assert len(retrieved) == 10

    # 2. Reranking
    reranked = rerank_documents(
        question,
        retrieved,
        top_k=3,
    )

    assert reranked
    assert len(reranked) == 3

    # 3. Build citation-aware context
    context = build_context(reranked)

    assert context
    assert len(context) == 3

    # 4. Generate final answer with the real Ollama + Qwen3:8b
    answer = generate_answer(question, context)

    assert isinstance(answer, str)
    assert answer.strip()

    # 5. Basic grounding sanity check
    assert any(
        "شكاوى" in item["text"]
        or "شكاوى" in item["citation"]["document"]
        for item in context
    )
