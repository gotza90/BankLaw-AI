"""Focused integration test for the ask(question) entry point.

This test verifies the public contract of ask(question): that it runs the
existing retrieval -> reranker -> context -> prompt -> LLM pipeline and
returns a grounded answer with real citations. It deliberately does NOT
assert any exact Arabic document filename.
"""
import re

from banklaw_step1.evaluation import grounding_score
from banklaw_step1.query_understanding import rewrite_queries, _query_supported


def test_ask_returns_grounded_answer_with_citations():
    from banklaw_step1.query import ask

    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    result = ask(question)

    # 1. ask() must return a structured result with a non-empty answer.
    assert isinstance(result, dict), "ask() must return a structured result"
    answer = result.get("answer")
    assert isinstance(answer, str) and answer.strip(), "answer must be a non-empty string"

    # 2. The answer must carry at least one real [Citation: ...] tag.
    citations = re.findall(
        r"\[Citation:\s*(.*?)\s*\|\s*Section:\s*([^\]]+)\]",
        answer,
    )
    assert citations, "answer must contain at least one [Citation: ...] tag"

    for document, section in citations:
        assert document.strip(), "citation document must not be empty"
        assert section.strip(), "citation section must not be empty"

    # 3. The returned context must be non-empty.
    context = result.get("context")
    assert context, "ask() must expose the context it answered from"

    # 4. Each context item must carry citation fields.
    for item in context:
        assert "text" in item, "context item must have a text field"
        assert "citation" in item, "context item must have a citation"
        assert item["citation"].get("document"), "citation must name a document"
        assert item["citation"].get("section") is not None, "citation must name a section"

    # 5. The answer must be grounded in the retrieved context.
    score = grounding_score(answer, context)
    assert score > 0.30, f"grounding score too low: {score:.3f}"


_ANALYSIS = {
    "intent": "حدود/حد أقصى",
    "topic": "السحب من الصراف الآلي",
    "product_or_service": "صراف آلي",
    "requested_information_type": "قيمة محددة/مبلغ",
    "constraints": ["عملة محددة"],  # NOT present in the actual question below
    "language": "Egyptian Arabic",
    "question": "ما هي الحدود القصوى للسحب اليومي من ماكينات الصراف الآلي؟",
}


def test_rewrite_includes_original_question():
    queries = rewrite_queries(_ANALYSIS, use_llm=False)
    assert queries and queries[0] == _ANALYSIS["question"]


def test_rewrite_does_not_invent_constraints():
    queries = rewrite_queries(_ANALYSIS, use_llm=False)
    assert all("عملة محددة" not in q for q in queries)


def test_rewrite_is_small_and_non_empty():
    queries = rewrite_queries(_ANALYSIS, use_llm=False)
    assert 1 <= len(queries) <= 3
    assert all(q.strip() for q in queries)


def test_rewrite_preserves_daily_maximum_qualifier():
    queries = rewrite_queries(_ANALYSIS, use_llm=False)
    joined = " ".join(queries)
    assert "يومي" in joined and "أقصى" in joined


_ORIG_ATM = "ما هي الحدود القصوى للسحب اليومي من ماكينات الصراف الآلي؟"


def test_rewrite_rejects_invented_currency():
    rewrite = "ما هي الحدود القصوى للسحب اليومي من ماكينات الصراف الآلي بالجنيه المصري؟"
    assert _query_supported(_ORIG_ATM, rewrite) is False


def test_rewrite_rejects_invented_amount():
    rewrite = "ما هو الحد الأقصى للسحب اليومي قدره 5000 جنيه؟"
    assert _query_supported(_ORIG_ATM, rewrite) is False


def test_rewrite_accepts_legitimate_paraphrase():
    rewrite = "ما هو الحد الأقصى للسحب اليومي من الصراف الآلي؟"
    assert _query_supported(_ORIG_ATM, rewrite) is True
