from banklaw_step1.evaluation import recall_at_k, reciprocal_rank, claim_grounding_score
from banklaw_step1.evaluation import grounding_score
from banklaw_step1.reranker import rerank_documents
from banklaw_step1.vector_store import query_documents


def test_recall_at_k_returns_one_when_relevant_document_is_found():
    results = [
        {"metadata": {"document": "other.pdf"}},
        {"metadata": {"document": "target.pdf"}},
        {"metadata": {"document": "another.pdf"}},
    ]

    assert recall_at_k(
        results,
        expected_document="target.pdf",
        k=3,
    ) == 1.0


def test_recall_at_k_returns_zero_when_relevant_document_is_missing():
    results = [
        {"metadata": {"document": "other.pdf"}},
        {"metadata": {"document": "another.pdf"}},
    ]

    assert recall_at_k(
        results,
        expected_document="target.pdf",
        k=2,
    ) == 0.0


def test_reciprocal_rank_returns_inverse_position():
    results = [
        {"metadata": {"document": "other.pdf"}},
        {"metadata": {"document": "target.pdf"}},
        {"metadata": {"document": "another.pdf"}},
    ]

    assert reciprocal_rank(
        results,
        expected_document="target.pdf",
    ) == 0.5


def test_reciprocal_rank_returns_zero_when_missing():
    results = [
        {"metadata": {"document": "other.pdf"}},
    ]

    assert reciprocal_rank(
        results,
        expected_document="target.pdf",
    ) == 0.0


GOLDEN_QUERIES = [
    {
        "query": "ما هي آلية التعامل مع الشكاوى بالبنوك؟",
        "expected_document": "الباب الرابع عشر   الفصل الثاني   التعليمات المنظمة لآلية التعامل مع الشكاوى بالبنوك.pdf",
    },
    {
        "query": "ما هي حقوق العملاء لدى البنوك؟",
        "expected_document": "الفصل الأول - الباب الرابع عشر - حقوق العملاء.pdf",
    },
    {
        "query": "ما هي التعليمات الرقابية المنظمة للحسابات الراكدة؟",
        "expected_document": "الباب الخامس الفصل الثالث عشر- التعليمات الرقابية المنظمة للحسابات الراكدة.pdf",
    },
    {
        "query": "ما هي ضوابط مزاولة نشاط أمناء الحفظ؟",
        "expected_document": "الباب الخامس الفصل الخامس- ضوابط مزاولة نشاط أمناء الحفظ.pdf",
    },
    {
        "query": "ما هي القواعد المنظمة للتعامل مع صناديق الاستثمار وأسواق النقد؟",
        "expected_document": "الباب الخامس الفصل الثانى عشر- صناديق الاستثمار وأسواق النقد.pdf",
    },
    {
        "query": "ما هي ضوابط مزاولة البنوك نشاط التأمين المصرفي؟",
        "expected_document": "الباب الخامس الفصل التاسع- ضوابط مزاولة البنوك نشاط التأمين المصرفي.pdf",
    },
    {
        "query": "ما هي ضوابط مزاولة نشاط بنوك الإيداع؟",
        "expected_document": "الباب الخامس الفصل السابع- ضوابط مزاولة نشاط بنوك الإيداع.pdf",
    },
    {
        "query": "ما هي ضوابط نشاط المتعاملين الرئيسيين لأذون الخزانة والسندات الحكومية؟",
        "expected_document": "الباب الخامس الفصل السادس- ضوابط مزاولة نشاط المتعاملين الرئيسيين لأذون الخزانة والسندات الحكومية.pdf",
    },
    {
        "query": "ما هي القواعد الخاصة بالإفصاح عن مواصفات وشروط منتجات التجزئة المصرفية؟",
        "expected_document": "الفصل الثانى-الإفصاح عن مواصفات وشروط منتجات التجزئة المصرفية.pdf",
    },
    {
        "query": "ما هو قانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020؟",
        "expected_document": "CBE Law No. 194 of 2020.pdf",
    },
]


from banklaw_step1.vector_store import query_documents


def test_golden_queries_retrieve_expected_documents_in_top_5():
    for case in GOLDEN_QUERIES:
        results = query_documents(case["query"], n_results=5)

        assert recall_at_k(
            results,
            expected_document=case["expected_document"],
            k=5,
        ) == 1.0, (
            f"Expected document not found in top-5 for query: "
            f"{case['query']}"
        )

from banklaw_step1.vector_store import query_documents


def test_golden_queries_report_retrieval_metrics():
    recall_3_scores = []
    recall_5_scores = []
    mrr_scores = []

    for case in GOLDEN_QUERIES:
        results = query_documents(
            case["query"],
            n_results=5,
        )

        recall_3_scores.append(
            recall_at_k(
                results,
                expected_document=case["expected_document"],
                k=3,
            )
        )

        recall_5_scores.append(
            recall_at_k(
                results,
                expected_document=case["expected_document"],
                k=5,
            )
        )

        mrr_scores.append(
            reciprocal_rank(
                results,
                expected_document=case["expected_document"],
            )
        )

    recall_3 = sum(recall_3_scores) / len(recall_3_scores)
    recall_5 = sum(recall_5_scores) / len(recall_5_scores)
    mrr = sum(mrr_scores) / len(mrr_scores)

    print(f"\nRecall@3: {recall_3:.4f}")
    print(f"Recall@5: {recall_5:.4f}")
    print(f"MRR:      {mrr:.4f}")

    assert 0.0 <= recall_3 <= 1.0
    assert recall_5 == 1.0
    assert 0.0 <= mrr <= 1.0


def test_golden_queries_report_reranked_metrics():
    recall_3_scores = []
    recall_5_scores = []
    mrr_scores = []

    for case in GOLDEN_QUERIES:
        retrieved = query_documents(
            case["query"],
            n_results=10,
        )

        reranked = rerank_documents(
            case["query"],
            retrieved,
            top_k=5,
        )

        recall_3_scores.append(
            recall_at_k(
                reranked,
                expected_document=case["expected_document"],
                k=3,
            )
        )

        recall_5_scores.append(
            recall_at_k(
                reranked,
                expected_document=case["expected_document"],
                k=5,
            )
        )

        mrr_scores.append(
            reciprocal_rank(
                reranked,
                expected_document=case["expected_document"],
            )
        )

    recall_3 = sum(recall_3_scores) / len(recall_3_scores)
    recall_5 = sum(recall_5_scores) / len(recall_5_scores)
    mrr = sum(mrr_scores) / len(mrr_scores)

    print(f"\nReranked Recall@3: {recall_3:.4f}")
    print(f"Reranked Recall@5: {recall_5:.4f}")
    print(f"Reranked MRR:      {mrr:.4f}")

    assert 0.0 <= recall_3 <= 1.0
    assert 0.0 <= recall_5 <= 1.0
    assert 0.0 <= mrr <= 1.0


def test_debug_reranked_golden_queries():
    from banklaw_step1.reranker import rerank_documents
    from banklaw_step1.vector_store import query_documents

    for case in GOLDEN_QUERIES:
        retrieved = query_documents(
            case["query"],
            n_results=10,
        )

        reranked = rerank_documents(
            case["query"],
            retrieved,
            top_k=5,
        )

        print("\n" + "=" * 80)
        print("QUERY:", case["query"])
        print("EXPECTED:", case["expected_document"])

        for i, result in enumerate(reranked, start=1):
            print(
                i,
                result["metadata"].get("document"),
                "| score:",
                round(result["score"], 4),
                "| section:",
                result["metadata"].get("section_number"),
            )


def test_debug_reranker_problem_cases():
    from banklaw_step1.reranker import rerank_documents
    from banklaw_step1.vector_store import query_documents

    queries = [
        (
            "ما هي آلية التعامل مع الشكاوى بالبنوك؟",
            "الباب الرابع عشر   الفصل الثاني   التعليمات المنظمة لآلية التعامل مع الشكاوى بالبنوك.pdf",
        ),
        (
            "ما هو قانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020؟",
            "CBE Law No. 194 of 2020.pdf",
        ),
    ]

    for query, expected_document in queries:
        retrieved = query_documents(query, n_results=10)

        reranked = rerank_documents(
            query,
            retrieved,
            top_k=5,
        )

        print("\n" + "=" * 80)
        print("QUERY:", query)
        print("EXPECTED:", expected_document)

        print("\nBEFORE RERANKING:")
        for i, result in enumerate(retrieved, start=1):
            print(
                i,
                "|",
                result["metadata"].get("document"),
                "|",
                result["metadata"].get("section_number"),
                "| distance:",
                round(result["distance"], 4),
            )
            print(result["page_content"][:500])

        print("\nAFTER RERANKING:")
        for i, result in enumerate(reranked, start=1):
            print(
                i,
                "|",
                result["metadata"].get("document"),
                "|",
                result["metadata"].get("section_number"),
                "| score:",
                round(result["score"], 4),
            )
            print(result["page_content"][:500])




def test_grounding_score_is_one_when_answer_matches_context():
    answer = "البنك يلتزم بالتعامل مع شكاوى العملاء."
    context = [
        {
            "text": "يلتزم البنك بالتعامل مع شكاوى العملاء.",
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    score = grounding_score(answer, context)

    assert score == 1.0


def test_grounding_score_is_zero_for_unrelated_answer():
    answer = "أسعار الذهب ترتفع في الأسواق."
    context = [
        {
            "text": "يلتزم البنك بالتعامل مع شكاوى العملاء.",
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    score = grounding_score(answer, context)

    assert score == 0.0


def test_grounding_score_is_between_zero_and_one_for_partial_overlap():
    answer = "البنك يتعامل مع شكاوى العملاء وفق الإجراءات."
    context = [
        {
            "text": "يلتزم البنك بالتعامل مع شكاوى العملاء.",
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    score = grounding_score(answer, context)

    assert 0.0 < score < 1.0


def test_claim_grounding_detects_supported_claim():
    context = [
        {
            "text": "يلتزم البنك بتوفير آلية سريعة ومجانية للتعامل مع شكاوى العملاء.",
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    claim = "يلتزم البنك بتوفير آلية سريعة ومجانية للتعامل مع شكاوى العملاء."

    assert claim_grounding_score(claim, context) == 1.0



def test_claim_grounding_does_not_treat_shared_words_as_support():
    context = [
        {
            "text": "يلتزم البنك بتوفير آلية سريعة ومجانية للتعامل مع شكاوى العملاء.",
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    claim = "يجب على البنك فرض رسوم على تقديم الشكاوى."

    assert claim_grounding_score(claim, context) < 0.5

def test_claim_grounding_detects_contradictory_claim():
    context = [
        {
            "text": "يلتزم البنك بتوفير آلية سريعة ومجانية للتعامل مع شكاوى العملاء.",
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    claim = "التعامل مع الشكاوى يكون بمقابل مالي."

    assert claim_grounding_score(claim, context) < 0.5



def test_general_complaint_query_retrieves_general_rule():
    query = "ماذا يجب على البنك أن يفعل فور استلام شكوى من العميل؟"

    retrieved = query_documents(query, n_results=10)
    reranked = rerank_documents(query, retrieved, top_k=5)

    top_result = reranked[0]

    assert top_result["metadata"].get("document") == (
        "الباب الرابع عشر   الفصل الثاني   التعليمات المنظمة لآلية التعامل مع الشكاوى بالبنوك.pdf"
    )

    assert "فور استلام الشكوى" in top_result["page_content"]


# ---------------------------------------------------------------------------
# Regression test: general questions must not be displaced by specialized
# sections that merely share more keywords with the query.
#
# The complaints instructions document contains a general rule describing
# the mechanism for handling complaints ("يتعين على البنك فور استلام
# الشكوى ...") and a specialized section ("2005-5 تحديد المشاكل النظامية
# والمتكررة ...") that discusses systemic / repeated cases.
#
# For a general question about how banks handle complaints, the general rule
# must rank above the specialized section. A CrossEncoder that scores on
# keyword overlap alone promotes the specialized section over the general
# rule, which is the wrong answer for a general question.
# ---------------------------------------------------------------------------
COMPLAINTS_GENERAL_QUERY = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"
COMPLAINTS_SPECIALIZED_SECTION = "2005-5"
COMPLAINTS_GENERAL_RULE_MARKER = "يتعين على البنك فور استلام"
COMPLAINTS_SPECIALIZED_RULE_MARKER = "تحديد المشاكل النظامية والمتكررة"


def _sections(reranked):
    return [result["metadata"].get("section_number") for result in reranked]


def test_general_complaint_query_ranks_general_rule_above_specialized():
    retrieved = query_documents(COMPLAINTS_GENERAL_QUERY, n_results=10)
    reranked = rerank_documents(COMPLAINTS_GENERAL_QUERY, retrieved, top_k=5)

    general_results = [
        result
        for result in reranked
        if COMPLAINTS_GENERAL_RULE_MARKER in result["page_content"]
    ]
    specialized_results = [
        result
        for result in reranked
        if COMPLAINTS_SPECIALIZED_RULE_MARKER in result["page_content"]
    ]

    assert general_results, (
        "General complaint-handling rule was not returned by the reranker"
    )
    assert specialized_results, (
        "Specialized systemic-complaints section was not returned by the reranker"
    )

    general_index = reranked.index(general_results[0])
    specialized_index = reranked.index(specialized_results[0])

    assert general_index < specialized_index, (
        f"General rule ranked at {general_index + 1} but specialized section "
        f"(2005-5) ranked at {specialized_index + 1}; the general rule must "
        "rank above a specialized section for a general question"
    )


def test_general_complaint_query_top_result_is_not_specialized_section():
    retrieved = query_documents(COMPLAINTS_GENERAL_QUERY, n_results=10)
    reranked = rerank_documents(COMPLAINTS_GENERAL_QUERY, retrieved, top_k=5)

    top_result = reranked[0]

    assert COMPLAINTS_SPECIALIZED_RULE_MARKER not in top_result["page_content"], (
        "Top reranked result is the specialized systemic-complaints section, "
        "which should not displace the general rule for a general question"
    )
    assert COMPLAINTS_GENERAL_RULE_MARKER in "\n".join(
        result["page_content"] for result in reranked
    ), "General complaint-handling rule was not returned by the reranker"
