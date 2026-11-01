"""Focused test for the Streamlit UI error handling.

Verifies that a failing ask() inside the chat submit path produces a
friendly st.error message instead of a raw Streamlit script exception.
"""
import streamlit.testing.v1 as st_test


def _boom(question, n_results=10, top_k=3):
    raise ConnectionError("Ollama is not running")


def test_streamlit_ui_shows_friendly_error_when_ask_fails():
    import banklaw_ui

    at = st_test.AppTest.from_function(
        banklaw_ui.build_app,
        kwargs={"ask": _boom},
        default_timeout=120,
    )
    at.run()

    # The chat input must exist.
    chat_inputs = [el for el in at if el.type == "chat_input"]
    assert chat_inputs, "UI must have a chat input"

    chat_inputs[0].set_value("\u0645\u0627 \u0647\u064a \u0622\u0644\u064a\u0629").run()

    # A failing ask() must NOT propagate as a raw Streamlit script
    # exception. AppTest reports st.error() as an exception element, so we
    # verify none of them is a genuine unhandled script exception.
    for exc in at.exception:
        message = str(getattr(exc, "message", exc))
        assert "Traceback" not in message, (
            f"UI raised an unhandled script exception: {message[:200]}"
        )

    # A friendly error message must be displayed to the user.
    error_elements = [el for el in at if el.type == "error"]
    assert error_elements, "UI must show an st.error message when ask() fails"

    error_text = "\n".join(
        str(getattr(el, "value", getattr(el, "body", ""))) for el in error_elements
    )
    assert "Ollama is not running" in error_text, (
        "UI must show a friendly error message when ask() fails"
    )