from banklaw_step1.vector_store import query_documents
from banklaw_step1.reranker import rerank_documents
from banklaw_step1.context import build_context
from banklaw_step1.llm import generate_answer
import re

def test_rag_answer_is_grounded_in_retrieved_context():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    retrieved = query_documents(question, n_results=10)
    reranked = rerank_documents(question, retrieved, top_k=3)
    context = build_context(reranked)

    answer = generate_answer(question, context)

    print("\n=== RETRIEVED SOURCES ===")
    for item in context:
        print(
            f"- {item['citation']['document']} | "
            f"Section: {item['citation']['section']}"
        )

    print("\n=== ANSWER ===")
    print(answer)

    assert answer.strip()

    # The answer should stay grounded in the retrieved material.
    grounding_terms = ["شكاوى", "العملاء", "البنك"]

    assert any(term in answer for term in grounding_terms)


from banklaw_step1.evaluation import grounding_score


def test_real_rag_grounding_score():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    retrieved = query_documents(question, n_results=10)
    reranked = rerank_documents(question, retrieved, top_k=3)
    context = build_context(reranked)

    assert context

    answer = generate_answer(question, context)

    score = grounding_score(answer, context)

    print("\n=== GROUNDING SCORE ===")
    print(f"{score:.4f}")

    print("\n=== ANSWER ===")
    print(answer)

    assert answer.strip()
    assert score > 0.30


def test_answer_citations_include_document_and_section():
    answer = """
    1. يتعين على البنوك توفير آلية سريعة ومجانية للتعامل مع الشكاوى.
    [Citation: complaints.pdf | Section: 4-3]

    2. يجب تحليل المشاكل النظامية والمتكررة.
    [Citation: complaints.pdf | Section: 2005-5]
    """

    citations = re.findall(
        r"\[Citation:\s*(.*?)\s*\|\s*Section:\s*([^\]]+)\]",
        answer,
    )

    assert citations
    assert len(citations) == 2

    for document, section in citations:
        assert document.strip()
        assert section.strip()


def test_each_numbered_claim_has_citation():
    answer = """
    1. يجب على البنك توفير آلية سريعة ومجانية للتعامل مع الشكاوى.
    [Citation: complaints.pdf | Section: 4-3]

    2. يجب تحليل المشاكل النظامية والمتكررة.
    [Citation: complaints.pdf | Section: 2005-5]

    3. يجب رفع تقارير دورية إلى لجنة المراجعة.
    [Citation: complaints.pdf | Section: 2005-5]
    """

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    numbered_claims = [
    line for line in lines
    if line.split(".", 1)[0].isdigit()
]

    assert numbered_claims
    assert len(numbered_claims) == 3

    for claim in numbered_claims:
        claim_index = lines.index(claim)
        # The LLM emits citations inline ON the claim line
        # (e.g. "1. ... [Citation: doc | Section: x]"), so the citation
        # window must INCLUDE the claim line, not just the lines after it.
        following_lines = lines[claim_index:claim_index + 3]

        assert any("[Citation:" in line for line in following_lines)


def test_real_rag_simple_question_is_single_paragraph_with_one_citation():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    retrieved = query_documents(question, n_results=10)
    reranked = rerank_documents(question, retrieved, top_k=3)
    context = build_context(reranked)

    answer = generate_answer(question, context)

    lines = [line.strip() for line in answer.splitlines() if line.strip()]

    # A simple/direct question must produce ONE answer paragraph, not a
    # numbered list of claims.
    assert len(lines) <= 2, (
        f"Simple question should yield a single concise paragraph, "
        f"got {len(lines)} lines: {answer[:200]}"
    )

    citations = re.findall(
        r"\[Citation:\s*(.*?)\s*\|\s*Section:\s*([^\]]+)\]",
        answer,
    )
    assert citations, f"Answer must carry at least one inline citation: {answer[:200]}"

    # The single-paragraph contract allows exactly one citation for a simple question.
    assert len(citations) == 1, (
        f"Simple question should carry exactly one citation, got {len(citations)}"
    )


def _normalize_whitespace(value):
    """Collapse runs of whitespace to single spaces and strip."""
    return " ".join(value.split())


def test_real_rag_citations_are_complete_and_match_context():
    """Validate the REAL generate_answer() output for citation completeness.

    Every [Citation: ...] in the answer must carry a non-empty document AND
    section, and must reference one of the sources the model was actually
    given. Document names are compared after normalizing whitespace, since
    the LLM may reproduce multi-space names with single spaces.
    """
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    retrieved = query_documents(question, n_results=10)
    reranked = rerank_documents(question, retrieved, top_k=3)
    context = build_context(reranked)

    answer = generate_answer(question, context)

    citations = re.findall(
        r"\[Citation:\s*(.*?)\s*\|\s*Section:\s*([^\]]+)\]",
        answer,
    )

    assert citations, "real answer must contain at least one [Citation: ...]"

    context_documents = {
        _normalize_whitespace(item["citation"]["document"])
        for item in context
    }

    for document, section in citations:
        assert document.strip(), "citation document must not be empty"
        assert section.strip(), "citation section must not be empty"
        assert _normalize_whitespace(document.strip()) in context_documents, (
            f"citation references unknown document: {document!r}"
        )
