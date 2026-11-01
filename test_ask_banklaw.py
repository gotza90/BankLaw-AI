"""Focused test for the CLI's Arabic UTF-8 printing safety.

TDD step 1: this test should FAIL with the current ask_banklaw.py because
plain print() uses the console's cp1252 encoding on Windows and raises
UnicodeEncodeError for Arabic text. It must pass only after the CLI
reconfigures stdout to UTF-8.

The Arabic text itself is asserted implicitly: the fake answer/question
contain Arabic, and if the CLI tried to print them through a strict
cp1252 stream it raises UnicodeEncodeError, which the test catches.
The remaining assertions check the ASCII structural parts of the output.
"""
import io
import sys

import ask_banklaw


def _cp1252_stream():
    """A real Windows-default (cp1252) text stream, like the console."""
    return io.TextIOWrapper(
        io.BytesIO(),
        encoding="cp1252",
        errors="strict",
        write_through=True,
    )


def test_cli_prints_arabic_utf8_without_unicode_error(monkeypatch):
    # Arabic question and answer: if the CLI prints these through a strict
    # cp1252 stream without reconfiguring to UTF-8, it raises
    # UnicodeEncodeError, which is exactly the bug this test guards.
    fake_result = {
        "question": "\u0645\u0627 \u0647\u064a \u0622\u0644\u064a\u0629 \u0627\u0644\u062a\u0639\u0627\u0645\u0644 \u0645\u0639 \u0627\u0644\u0634\u0643\u0648\u0649 \u0628\u0627\u0644\u0628\u0646\u0648\u0643\u061f",
        "answer": "\u0627\u0644\u0622\u0644\u064a\u0629 \u062a\u0623\u0645\u0646 \u0628\u0631\u0645\u0629 \u0648\u0627\u062b\u0631\u0629 \u0645\u0646 \u0627\u0644\u0634\u0643\u0648\u0649. [Citation: doc.pdf | Section: 1]",
        "context": [
            {
                "text": "some text",
                "citation": {
                    "document": "doc.pdf",
                    "section": "1",
                    "source": "Central Bank of Egypt",
                },
            },
        ],
    }

    monkeypatch.setattr(ask_banklaw, "ask", lambda question: fake_result)

    stream = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)

    # The CLI must not raise UnicodeEncodeError when printing Arabic text.
    try:
        ask_banklaw.main(["question"])
    except UnicodeEncodeError as exc:
        raise AssertionError(
            "CLI raised UnicodeEncodeError: " + str(exc)
        ) from exc

    output = stream.buffer.getvalue().decode("utf-8", errors="replace")

    # The answer's citation tag is preserved verbatim.
    assert "[Citation: doc.pdf | Section: 1]" in output

    # The Sources list is printed from result["context"].
    assert "Sources:" in output
    assert "doc.pdf -- Section: 1" in output


def test_cli_dedupes_duplicate_sources(monkeypatch):
    """Duplicate document + section pairs must not be printed twice."""
    fake_result = {
        "question": "\u0642\u0636\u0631\u064a",
        "answer": "\u0627\u0644\u0625\u062c\u0627\u0628\u0629. [Citation: doc.pdf | Section: 1]",
        "context": [
            {
                "text": "a",
                "citation": {
                    "document": "doc.pdf",
                    "section": "1",
                    "source": "Central Bank of Egypt",
                },
            },
            {
                "text": "b",
                "citation": {
                    "document": "doc.pdf",
                    "section": "1",
                    "source": "Central Bank of Egypt",
                },
            },
            {
                "text": "c",
                "citation": {
                    "document": "other.pdf",
                    "section": "2",
                    "source": "Central Bank of Egypt",
                },
            },
        ],
    }

    monkeypatch.setattr(ask_banklaw, "ask", lambda question: fake_result)
    stream = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)

    ask_banklaw.main(["question"])

    output = stream.buffer.getvalue().decode("utf-8", errors="replace")
    assert output.count("doc.pdf -- Section: 1") == 1, (
        "duplicate sources must be deduplicated in the CLI output"
    )
    assert output.count("other.pdf -- Section: 2") == 1