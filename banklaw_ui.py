def build_app(ask=None):
    """The Streamlit app body.

    `ask` defaults to banklaw_step1.query.ask and is injected as a
    parameter (not a module global) so AppTest can monkeypatch it and so
    the function is self-contained when re-executed in isolation.
    """
    import streamlit as st

    if ask is None:
        from banklaw_step1.query import ask as ask

    # Page config is optional and is a no-op when run under AppTest, so it
    # is guarded to keep the app importable and testable in bare mode.
    try:
        st.set_page_config(
            page_title="banklaw-ai",
            page_icon="\u2696\ufe0f",
            layout="centered",
            initial_sidebar_state="collapsed",
        )
    except Exception:
        pass

    # RTL-friendly styling for Arabic legal text.
    st.markdown(
        """
        <style>
        div[data-testid="stMarkdownContainer"] {
            direction: rtl;
            text-align: right;
            unicode-bidi: embed;
        }
        textarea, input[type="text"] {
            direction: rtl;
            text-align: right;
            unicode-bidi: embed;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    import re

    def _keep_single_citation(text):
        """Collapse any [Citation: ...] tags in the answer body to exactly one.

        A simple question must carry exactly one inline citation. If the LLM
        emits extra tags (e.g. one at the end of the paragraph and another on
        a separate line), strip everything after the first so the rendered
        answer never shows more than one.
        """
        tags = re.findall(r"\[Citation:[^\]]*\]", text)
        if len(tags) <= 1:
            return text
        first = tags[0]
        # Remove all citation tags, then re-insert exactly the first one.
        stripped = re.sub(r"\[Citation:[^\]]*\]", "", text).strip()
        return f"{stripped}\n\n{first}"

    def _strip_citations(text):
        """Remove [Citation: ...] tags for display only.

        Citations remain in the backend answer for grounding/validation; the
        visible chat message shows only the natural Arabic text.
        """
        return re.sub(r"\[Citation:[^\]]*\]", "", text).strip()

    # Persistent conversation history: list of (role, content) tuples.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("banklaw-ai")
    st.write(
        "\u0645\u0633\u0627\u0639\u062f \u0642\u0627\u0646\u0648\u0646\u064a "
        "\u0645\u062e\u0635\u0635 \u0641\u064a \u0627\u0644\u0642\u0648\u0627\u0646\u064a\u0646 "
        "\u0648\u0627\u0644\u062a\u0639\u0644\u064a\u0645\u0627\u062a \u0627\u0644\u0645\u0635\u0631\u0641\u064a\u0629 "
        "\u0627\u0644\u0645\u0635\u0631\u064a\u0629."
    )

    # Render the full conversation history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at the bottom; returns None until the user submits.
    question = st.chat_input(
        "\u0627\u0643\u062a\u0628 \u0633\u0624\u0627\u0644\u0633 \u0628\u0627\u0644\u0639\u0631\u0628\u064a\u0629..."
    )

    if question and question.strip():
        user_question = question.strip()

        # Append and render the user message immediately.
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate the assistant answer inside a status box, then render it.
        with st.status(
            "\u062c\u0627\u0631\u064a \u0627\u0644\u0625\u062c\u0627\u0628\u0629...",
            expanded=True,
        ) as status_box:
            st.write(
                "\u064a\u062a\u0645 \u0627\u0644\u0628\u062d\u062b \u0641\u064a "
                "\u0627\u0644\u0645\u0635\u0627\u062f\u0631 \u0648\u0627\u0644\u062a\u0639\u0644\u064a\u0645..."
            )
            try:
                result = ask(user_question)
            except Exception as exc:
                status_box.update(
                    label="\u0627\u0643\u062a\u0645\u0644 \u0627\u0644\u0625\u062c\u0627\u0628\u0629",
                    state="error",
                    expanded=True,
                )
                st.error(
                    "\u064a\u062d\u062f\u062b \u062e\u0637\u0623 \u0623\u062b\u0646\u0627\u0621 "
                    "\u0627\u0644\u0625\u062c\u0627\u0628\u0629. \u064a\u0630\u0647\u0631 "
                    + str(exc)
                )
                return
            status_box.update(
                label="\u0627\u0643\u062a\u0645\u0644 \u0627\u0644\u0625\u062c\u0627\u0628\u0629",
                state="complete",
                expanded=False,
            )

        answer = _keep_single_citation(result["answer"])
        display_answer = _strip_citations(answer)
        st.session_state.messages.append({"role": "assistant", "content": display_answer})
        with st.chat_message("assistant"):
            st.markdown(display_answer)


# Streamlit runs the top-level script directly when you do
# `streamlit run banklaw_ui.py`, so the app body must be invoked at module
# top level rather than behind an `if __name__ == "__main__":` guard (which
# is False under Streamlit's script runner and would render a blank page).
build_app()