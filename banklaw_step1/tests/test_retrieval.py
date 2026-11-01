from banklaw_step1.vector_store import query_documents


def test_retrieval_returns_relevant_legal_chunks():
    results = query_documents(
        "ما هي حقوق العميل عند تقديم شكوى إلى البنك؟",
        n_results=5,
    )

    assert len(results) == 5

    for result in results:
        assert result["page_content"]
        assert result["metadata"]
        assert "document" in result["metadata"]
        assert "section_number" in result["metadata"]
        assert "distance" in result


def test_retrieval_finds_customer_protection_content():
    results = query_documents(
        "ما هي حقوق العميل عند تقديم شكوى إلى البنك؟",
        n_results=5,
    )

    relevant_results = [
        result
        for result in results
        if result["metadata"].get("category") == "customer_protection"
    ]

    assert relevant_results, (
        "Retrieval did not return any customer_protection chunk "
        "for a customer complaint/rights query."
    )

def test_retrieval_places_relevant_customer_protection_in_top_3():
    results = query_documents(
        "ما هي حقوق العميل عند تقديم شكوى إلى البنك؟",
        n_results=3,
    )

    assert any(
        result["metadata"].get("category") == "customer_protection"
        for result in results
    )




def test_reranker_returns_ranked_results_with_scores():
    from banklaw_step1.reranker import rerank_documents

    results = [
        {
            "page_content": "النص الأول عن الشكاوى وحقوق العملاء.",
            "metadata": {"document": "complaints.pdf"},
        },
        {
            "page_content": "نص عام عن خدمات البنوك.",
            "metadata": {"document": "banking.pdf"},
        },
    ]

    ranked = rerank_documents(
        "ما هي حقوق العميل عند تقديم شكوى إلى البنك؟",
        results,
        top_k=2,
    )

    assert len(ranked) == 2
    assert all("score" in result for result in ranked)
    assert ranked[0]["score"] >= ranked[1]["score"]
    assert ranked[0]["metadata"]["document"] == "complaints.pdf"


def test_reranker_improves_real_customer_complaint_results():
    from banklaw_step1.reranker import rerank_documents
    from banklaw_step1.vector_store import query_documents

    query = "ما هي حقوق العميل عند تقديم شكوى إلى البنك؟"

    retrieved = query_documents(query, n_results=10)
    reranked = rerank_documents(query, retrieved, top_k=3)

    assert len(reranked) == 3

    assert all("score" in result for result in reranked)

    assert reranked[0]["score"] >= reranked[1]["score"]
    assert reranked[1]["score"] >= reranked[2]["score"]

    assert reranked[0]["metadata"].get("category") == "customer_protection"


def test_document_aware_reranker_preserves_top_semantic_source():
    from banklaw_step1.reranker import rerank_documents

    query = "ما هو قانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020؟"

    results = [
        {
            "page_content": "ورد في المستند قانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020.",
            "metadata": {
                "document": "CBE Law No. 194 of 2020.pdf",
            },
        },
        {
            "page_content": "خدمات السجل التجاري والتوثيق داخل فروع البنوك وفقاً لأحكام قانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020.",
            "metadata": {
                "document": "السجل التجاري.pdf",
            },
        },
    ]

    ranked = rerank_documents(
        query,
        results,
        top_k=2,
    )

    assert ranked[0]["metadata"]["document"] == "CBE Law No. 194 of 2020.pdf"



def test_reranker_prefers_exact_document_match_for_named_law():
    from banklaw_step1.reranker import rerank_documents

    query = "ما هو قانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020؟"

    documents = [
        {
            "page_content": (
                "خدمات السجل التجاري والتوثيق داخل فروع البنوك "
                "وفقًا لقانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020."
            ),
            "metadata": {
                "document": (
                    "الباب الخامس الفصل العاشر- ضوابط تقديم خدمات "
                    "السجل التجاري والشهر العقاري (التوثيق) داخل فروع البنوك.pdf"
                ),
            },
        },
        {
            "page_content": (
                "قانون البنك المركزي والجهاز المصرفي رقم 194 لسنة 2020 "
                "وينظم أعمال البنك المركزي والجهاز المصرفي."
            ),
            "metadata": {
                "document": "CBE Law No. 194 of 2020.pdf",
            },
        },
    ]

    ranked = rerank_documents(query, documents, top_k=2)

    assert ranked[0]["metadata"]["document"] == "CBE Law No. 194 of 2020.pdf"
