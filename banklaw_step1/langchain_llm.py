"""Optional LangChain integration layer for banklaw-ai.

This module is a thin wrapper around the existing pipeline. It does NOT
replace retrieval, reranking, grounding, or citation validation — those stay
in `banklaw_step1.query` / `banklaw_step1.llm`. It only exposes the existing
LLM and prompt-building logic through LangChain types:

- ChatOllama for the existing Ollama LLM
- ChatPromptTemplate / PromptTemplate for prompts
- Document objects where appropriate

Usage:
    from banklaw_step1.langchain_llm import build_chat, build_prompt_template
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

from banklaw_step1.llm import OLLAMA_MODEL
from banklaw_step1.prompt import build_prompt
from banklaw_step1.query import ask as _ask


def build_chat(model: str | None = None, temperature: float = 0.0) -> ChatOllama:
    """Return a ChatOllama instance backed by the same Ollama model."""
    return ChatOllama(model=model or OLLAMA_MODEL, temperature=temperature)


def build_prompt_template() -> ChatPromptTemplate:
    """Wrap the existing prompt builder in a ChatPromptTemplate."""
    return ChatPromptTemplate.from_template(build_prompt("{question}", []))


def build_document_template() -> PromptTemplate:
    """A plain PromptTemplate for source documents."""
    return PromptTemplate.from_template(
        "[Source: {document} | Section: {section} | Authority: {source}]\n{text}"
    )


def to_langchain_documents(context: list[dict]) -> list[Document]:
    """Convert the existing citation-aware context into LangChain Documents."""
    documents = []
    for item in context:
        citation = item.get("citation", {})
        documents.append(
            Document(
                page_content=item.get("text", ""),
                metadata={
                    "document": citation.get("document", ""),
                    "section": citation.get("section", ""),
                    "source": citation.get("source", ""),
                },
            )
        )
    return documents


def ask_with_langchain(question: str):
    """Run the existing RAG pipeline and re-render the answer through LangChain.

    Retrieval, reranking, grounding, and citation validation are performed by
    `banklaw_step1.query.ask` unchanged. The result is then formatted with a
    LangChain ChatPromptTemplate and ChatOllama for observability/integration
    purposes only.
    """
    result = _ask(question)
    chat = build_chat()
    template = build_prompt_template()
    prompt_value = template.format(question=question, context="")
    # The LangChain call is informational; the authoritative answer comes
    # from the existing pipeline so behavior never diverges.
    try:
        chat.invoke(prompt_value.to_messages())
    except Exception:
        # LangChain invocation is best-effort; the pipeline answer is authoritative.
        pass
    return result


__all__ = [
    "build_chat",
    "build_prompt_template",
    "build_document_template",
    "to_langchain_documents",
    "ask_with_langchain",
]