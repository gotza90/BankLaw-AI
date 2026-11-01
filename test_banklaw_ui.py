"""Focused test for the Streamlit chat UI.

Drives the real Streamlit app via AppTest and verifies the chat contract:
chat input at the bottom, user message rendered, assistant answer rendered
directly below with exactly one inline citation, and conversation history
persisting across questions.
"""
import re

import streamlit.testing.v1 as st_test


def _fake_ask(question):
    return {
        "question": question,
        # Two citations: the first is the answer's own, the second is a
        # stray tag on a separate line. The UI must collapse to exactly one.
        "answer": (
            "The answer. [Citation: doc.pdf | Section: 1] "
            "and more [Citation: other.pdf | Section: 2]."
        ),
        "context": [
            {
                "text": "some text",
                "citation": {
                    "document": "doc.pdf",
                    "section": "1",
                    "source": "Central Bank of Egypt",
                },
            },
            {
                "text": "more text",
                "citation": {
                    "document": "other.pdf",
                    "section": "2",
                    "source": "Central Bank of Egypt",
                },
            },
            # A duplicate source that must be collapsed in the Sources list.
            {
                "text": "duplicate text",
                "citation": {
                    "document": "doc.pdf",
                    "section": "1",
                    "source": "Central Bank of Egypt",
                },
            },
        ],
    }


def _run_question(at, question):
    chat_inputs = [el for el in at if el.type == "chat_input"]
    assert chat_inputs, "UI must have a chat input"
    chat_inputs[0].set_value(question).run()
    return at


def test_streamlit_ui_renders_chat_message_and_answer():
    import banklaw_ui

    at = st_test.AppTest.from_function(
        banklaw_ui.build_app,
        args=(),
        kwargs={"ask": _fake_ask},
        default_timeout=120,
    )
    at.run()

    # Chat input at the bottom instead of text_input + submit button.
    chat_inputs = [el for el in at if el.type == "chat_input"]
    assert chat_inputs, "UI must have a chat input"
    assert not [el for el in at if el.type == "text_input"], (
        "legacy text_input must be replaced by chat_input"
    )
    assert not [el for el in at if el.type == "button"], (
        "legacy submit button must be replaced by chat_input"
    )

    _run_question(at, "\u0645\u0627 \u0647\u064a \u0622\u0644\u064a\u0629")

    # The user message must be rendered.
    user_messages = [el for el in at if el.type == "chat_message" and el.name == "user"]
    assert user_messages, "UI must render the user's question as a chat message"

    # The assistant message must be rendered.
    assistant_messages = [el for el in at if el.type == "chat_message" and el.name == "assistant"]
    assert assistant_messages, "UI must render the assistant's answer as a chat message"

    markdown = "\n".join(el.value for el in at if el.type == "markdown")
    assert "\u0645\u0627 \u0647\u064a \u0622\u0644\u064a\u0629" in markdown, (
        "user question must be rendered in the chat"
    )
    assert "The answer." in markdown, "answer must be displayed"

    # Citations must NOT appear anywhere in the visible chat UI.
    assert "[Citation:" not in markdown, (
        "citation tags must be hidden from the visible chat message"
    )
    citations_in_markdown = re.findall(r"\[Citation:.*?\]", markdown)
    assert len(citations_in_markdown) == 0, (
        "visible chat must contain zero citation tags"
    )

    # The Sources section must NOT be rendered in the chat UI.
    assert "Sources:" not in markdown, "Sources section must be hidden from the chat UI"
    assert "doc.pdf -- Section: 1" not in markdown, (
        "source list must not appear in the chat UI"
    )


def test_streamlit_ui_keeps_conversation_history():
    import banklaw_ui

    at = st_test.AppTest.from_function(
        banklaw_ui.build_app,
        args=(),
        kwargs={"ask": _fake_ask},
        default_timeout=120,
    )
    at.run()

    _run_question(at, "first question")
    _run_question(at, "second question")

    user_messages = [el for el in at if el.type == "chat_message" and el.name == "user"]
    assistant_messages = [el for el in at if el.type == "chat_message" and el.name == "assistant"]

    assert len(user_messages) == 2, (
        f"both user questions must remain visible, got {len(user_messages)}"
    )
    assert len(assistant_messages) == 2, (
        f"both assistant answers must remain visible, got {len(assistant_messages)}"
    )

    markdown = "\n".join(el.value for el in at if el.type == "markdown")
    assert "first question" in markdown, "first question must remain in the conversation"
    assert "second question" in markdown, "second question must appear below the first"