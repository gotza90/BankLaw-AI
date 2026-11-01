import pytest

from banklaw_step1.llm import generate_answer


def test_real_ollama_generates_answer():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    context = [
        {
            "text": (
                "يتعين على البنك تطبيق سياسات وإجراءات تضمن توفير آلية "
                "سريعة ومجانية للتعامل مع شكاوى العملاء بشكل عادل "
                "وبشفافية وفعالية."
            ),
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    try:
        answer = generate_answer(question, context)
    except ConnectionError:
        pytest.fail("Ollama is not running or is not accessible.")

    assert isinstance(answer, str)
    assert answer.strip()