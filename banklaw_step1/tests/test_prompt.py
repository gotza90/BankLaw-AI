from banklaw_step1.prompt import build_prompt


def test_build_prompt_includes_question_context_and_citation_rules():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

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

    prompt = build_prompt(question, context)

    assert question in prompt
    assert "يلتزم البنك بتوفير آلية سريعة ومجانية" in prompt
    assert "complaints.pdf" in prompt
    assert "4-3" in prompt

    assert "يعتمد فقط على المعلومات الواردة في المصادر" in prompt
    assert "إذا لم تكن معلومات كافية" in prompt

def test_build_prompt_requires_claim_level_citations():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

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

    prompt = build_prompt(question, context)

    assert "فقرة واحدة" in prompt
    assert "document" in prompt
    assert "section" in prompt


def test_build_prompt_requires_explicit_citation_format():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

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

    prompt = build_prompt(question, context)

    assert "[Citation:" in prompt
    assert "document | Section" in prompt


def test_build_prompt_rejects_vague_citation_style():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

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

    prompt = build_prompt(question, context)

    assert "(المصدر الأول)" in prompt


def test_build_prompt_requires_concise_legal_answer():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

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

    prompt = build_prompt(question, context)

    assert "مختصرة" in prompt
    assert "مباشرة" in prompt


def test_build_prompt_forbids_numbered_claims_for_simple_questions():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

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

    prompt = build_prompt(question, context)

    assert "القاعدة الأهم: اجب فقط عن السؤال المطروح" in prompt
    assert "لا تلخص إجراءات كاملة" in prompt
    assert "فقرة واحدة فقط" in prompt