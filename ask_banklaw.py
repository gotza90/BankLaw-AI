"""
Minimal command-line entry point for banklaw-ai.

Usage:
    python ask_banklaw.py "what is the mechanism for handling complaints in banks?"

It calls banklaw_step1.query.ask(question), prints the answer verbatim
(including its [Citation: ...] tags), and prints a Sources list derived
from result["context"]. It does not modify the answer or citations.
"""

import io
import sys

from banklaw_step1.query import ask


def _configure_utf8():
    """
    Reconfigure stdout/stderr as UTF-8 so Arabic output prints safely.

    On Windows the console defaults to a code page (cp1252) that cannot
    encode Arabic, so plain print() raises UnicodeEncodeError. This
    re-wraps the streams as UTF-8 when they are not already UTF-8. It is
    defensive: if a stream cannot be reconfigured (for example when it is
    replaced by a non-text object during testing) it is left untouched.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not isinstance(stream, io.TextIOBase):
            continue
        encoding = (stream.encoding or "").lower()
        if encoding.startswith("utf"):
            continue
        try:
            wrapped = io.TextIOWrapper(
                stream.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=stream.line_buffering,
                write_through=True,
            )
            setattr(sys, stream_name, wrapped)
        except (AttributeError, io.UnsupportedOperation):
            # Stream has no underlying buffer (e.g. already wrapped); skip.
            continue


def main(argv=None):
    _configure_utf8()

    argv = list(sys.argv[1:] if argv is None else argv)

    if len(argv) < 1 or not argv[0].strip():
        print("Usage: python ask_banklaw.py <question>")
        return 2

    question = " ".join(argv).strip()

    result = ask(question)

    print("Question:")
    print(result["question"])
    print()
    print("Answer:")
    print(result["answer"])
    print()
    print("Sources:")
    seen = set()
    unique_items = []
    for item in result["context"]:
        citation = item.get("citation", {})
        key = (citation.get("document", "unknown"), citation.get("section", "unknown"))
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    for index, item in enumerate(unique_items, start=1):
        citation = item.get("citation", {})
        document = citation.get("document", "unknown")
        section = citation.get("section", "unknown")
        print(f"  {index}. {document} -- Section: {section}")

    return 0


if __name__ == "__main__":
    sys.exit(main())