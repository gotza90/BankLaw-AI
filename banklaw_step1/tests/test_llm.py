from banklaw_step1 import llm


def test_generate_answer_returns_text():
    question = "ما هي آلية التعامل مع الشكاوى بالبنوك؟"

    context = [
        {
            "text": (
                "يلتزم البنك بتوفير آلية سريعة ومجانية "
                "للتعامل مع شكاوى العملاء."
            ),
            "citation": {
                "document": "complaints.pdf",
                "section": "4-3",
                "source": "Central Bank of Egypt",
            },
        }
    ]

    class FakeResponse:
        def __init__(self, text):
            self.message = {"content": text}

    class FakeClient:
        def chat(self, model, messages):
            return FakeResponse("إجابة تجريبية")

    original_client = llm._client

    try:
        llm._client = FakeClient()

        answer = llm.generate_answer(question, context)

        assert isinstance(answer, str)
        assert answer.strip()

    finally:
        llm._client = original_client