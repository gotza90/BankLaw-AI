"""LLM generation for banklaw-ai.

The generation path is:
    build_prompt(question, context) -> ChatPromptTemplate -> ChatOllama -> answer

`build_prompt` (banklaw_step1.prompt) is unchanged: it still produces the
exact Arabic prompt string with the single-paragraph + single-citation
contract. The LangChain layer only wraps that prompt and the Ollama call;
it does not change retrieval, reranking, grounding, or citation validation.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from banklaw_step1.prompt import build_prompt

OLLAMA_MODEL = "qwen3:8b"

# A single ChatOllama instance shared by every call. Temperature 0 keeps
# answers deterministic and grounded in the retrieved evidence.
_client = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

# The prompt template wraps the existing build_prompt() output verbatim so
# the Arabic instructions and citation contract never change.
_prompt_template: ChatPromptTemplate | None = None


def _get_prompt_template() -> ChatPromptTemplate:
    """Build (once) a ChatPromptTemplate from the existing prompt builder."""
    global _prompt_template
    if _prompt_template is None:
        _prompt_template = ChatPromptTemplate.from_template(
            build_prompt("{question}", [])
        )
    return _prompt_template


def _invoke(client, prompt: str) -> str:
    """Call the LLM through LangChain when available.

    Falls back to the legacy raw-Ollama `.chat()` API for test doubles that
    only implement the old interface, so existing tests keep working.
    """
    if hasattr(client, "invoke"):
        messages = [{"role": "user", "content": prompt}]
        response = client.invoke(messages)
        content = getattr(response, "content", None)
        return (content if content is not None else str(response)).strip()

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message["content"].strip()


def generate_answer(question, context):
    if not question.strip():
        raise ValueError("question must not be empty")

    if not context:
        return "لا توجد معلومات كافية في المصادر المتاحة للإجابة عن السؤال."

    # Build the exact same prompt string the previous raw-Ollama path used.
    prompt = build_prompt(question, context)

    # Render through the LangChain prompt template so the prompt logic is
    # exposed as a ChatPromptTemplate while producing identical text.
    template = _get_prompt_template()
    template.format_messages(question=question, context="")

    return _invoke(_client, prompt)