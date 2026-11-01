from banklaw_step1.context import build_context


def test_build_context_preserves_ranked_order():
    results = [
        {
            "page_content": "النص القانوني الأول.",
            "metadata": {
                "document": "law_a.pdf",
                "section_number": "1",
            },
            "score": 0.95,
        },
        {
            "page_content": "النص القانوني الثاني.",
            "metadata": {
                "document": "law_b.pdf",
                "section_number": "2",
            },
            "score": 0.80,
        },
    ]

    context = build_context(results)

    assert len(context) == 2
    assert context[0]["text"] == "النص القانوني الأول."
    assert context[0]["citation"]["document"] == "law_a.pdf"
    assert context[1]["text"] == "النص القانوني الثاني."
    assert context[1]["citation"]["document"] == "law_b.pdf"

def test_build_context_creates_llm_ready_context_with_citations():
    results = [
        {
            "page_content": "يلتزم البنك بتوفير آلية للتعامل مع الشكاوى.",
            "metadata": {
                "document": "complaints.pdf",
                "section_number": "4-3",
                "source": "Central Bank of Egypt",
            },
            "score": 0.95,
        },
        {
            "page_content": "يجب فحص الشكوى بكفاءة وجدية وحيادية.",
            "metadata": {
                "document": "complaints.pdf",
                "section_number": "6",
                "source": "Central Bank of Egypt",
            },
            "score": 0.90,
        },
    ]

    context = build_context(results)

    assert context[0]["text"] == "يلتزم البنك بتوفير آلية للتعامل مع الشكاوى."
    assert context[0]["citation"]["document"] == "complaints.pdf"
    assert context[0]["citation"]["section"] == "4-3"
    assert context[0]["citation"]["source"] == "Central Bank of Egypt"

    assert context[1]["text"] == "يجب فحص الشكوى بكفاءة وجدية وحيادية."
    assert context[1]["citation"]["document"] == "complaints.pdf"
    assert context[1]["citation"]["section"] == "6"


def test_build_context_creates_prompt_ready_text():
    results = [
        {
            "page_content": "يلتزم البنك بتوفير آلية للتعامل مع الشكاوى.",
            "metadata": {
                "document": "complaints.pdf",
                "section_number": "4-3",
                "source": "Central Bank of Egypt",
            },
            "score": 0.95,
        },
        {
            "page_content": "يجب فحص الشكوى بكفاءة وجدية وحيادية.",
            "metadata": {
                "document": "complaints.pdf",
                "section_number": "6",
                "source": "Central Bank of Egypt",
            },
            "score": 0.90,
        },
    ]

    context = build_context(results)

    assert isinstance(context, list)

    prompt_context = "\n\n".join(
        (
            f"[Source: {item['citation']['document']} | "
            f"Section: {item['citation']['section']} | "
            f"Authority: {item['citation']['source']}]\n"
            f"{item['text']}"
        )
        for item in context
    )

    assert "[Source: complaints.pdf | Section: 4-3 | Authority: Central Bank of Egypt]" in prompt_context
    assert "يلتزم البنك بتوفير آلية للتعامل مع الشكاوى." in prompt_context
    assert "يجب فحص الشكوى بكفاءة وجدية وحيادية." in prompt_context


# ---------------------------------------------------------------------------
# Regression test: OCR-noise section numbers must not leak into citations.
#
# The structure parser sometimes emits garbage section numbers for Law 194
# chunks (e.g. "؟", "2:؟", "1517/") when it cannot parse a legal article
# marker. build_context used to copy these straight into the citation, so
# the prompt received "[Citation: CBE Law No. 194 of 2020.pdf | Section: ؟]".
# The LLM then attached fabricated claims to that bogus citation.
#
# A citation section is only meaningful if it contains at least one Arabic
# letter or digit after stripping bidi marks, tatweel and whitespace.
# ---------------------------------------------------------------------------
def _has_meaningful_section(section):
    if section is None:
        return False
    cleaned = (
        section
        .replace("\u200e", "")
        .replace("\u200f", "")
        .replace("\u0640", "")
        .strip()
    )
    if not cleaned:
        return False
    return any("\u0621" <= ch <= "\u064a" or ch.isdigit() for ch in cleaned)


def test_build_context_sanitizes_ocr_noise_section_numbers():
    results = [
        {
            "page_content": "البنك المركزى جهاز رقابى مستقل.",
            "metadata": {
                "document": "CBE Law No. 194 of 2020.pdf",
                "section_number": "؟",
                "source": "Central Bank of Egypt",
            },
        },
        {
            "page_content": "الRules وال-stock market.",
            "metadata": {
                "document": "law.pdf",
                "section_number": "2:؟",
                "source": "Central Bank of Egypt",
            },
        },
        {
            "page_content": "نص قانوني.",
            "metadata": {
                "document": "law.pdf",
                "section_number": "1517/",
                "source": "Central Bank of Egypt",
            },
        },
    ]

    context = build_context(results)

    assert len(context) == 3
    for item in context:
        assert _has_meaningful_section(item["citation"]["section"]), (
            f"Garbage section leaked into citation: {item['citation']['section']!r}"
        )


def test_build_context_preserves_valid_section_numbers():
    results = [
        {
            "page_content": "يلتزم البنك بحماية العملاء.",
            "metadata": {
                "document": "customer_rights.pdf",
                "section_number": "4-3",
                "source": "Central Bank of Egypt",
            },
        },
        {
            "page_content": "نص قانوني.",
            "metadata": {
                "document": "law.pdf",
                "section_number": "2005-5",
                "source": "Central Bank of Egypt",
            },
        },
    ]

    context = build_context(results)

    assert context[0]["citation"]["section"] == "4-3"
    assert context[1]["citation"]["section"] == "2005-5"
