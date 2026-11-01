import re


NUMBERED_SECTION_PATTERN = re.compile(
    r"^(?P<number>\d+(?:-\d+)*)(?:-)?\s+(?P<title>.+)$"
)

ARTICLE_PATTERN = re.compile(
    r"^مادة\s*\(\s*(?P<number>[^()]*)\s*\)\s*:?\s*$"
)

REVERSED_ARTICLE_PATTERN = re.compile(
    r"^مادة\s*\)\s*(?P<number>[^()]+?)\s*\(\s*:?\s*$"
)

PAGE_HEADER_PATTERN = re.compile(
    r"^(?:[\d٠-٩]+\s+)?الجريدة الرسمية\b"
)

BIDI_CONTROLS = {
    "\u061c",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}


def _strip_bidi_controls(value: str) -> str:
    """Remove Unicode bidirectional control characters."""
    return "".join(char for char in value if char not in BIDI_CONTROLS)


def _normalize_arabic_digits(value: str) -> str:
    """Convert Arabic-Indic digits to Western digits."""
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western_digits = "0123456789"

    translation = str.maketrans(arabic_digits, western_digits)
    return value.translate(translation)


def _match_article(line: str) -> re.Match[str] | None:
    """Match normal or BiDi-reversed legal article markers."""
    normalized_line = _strip_bidi_controls(line)

    match = ARTICLE_PATTERN.match(normalized_line)

    if match:
        return match

    return REVERSED_ARTICLE_PATTERN.match(normalized_line)


def _is_article_based(text: str) -> bool:
    """Detect whether a document is primarily organized by legal articles."""
    article_markers = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if _match_article(line):
            article_markers += 1

            if article_markers >= 2:
                return True

    return False


def extract_sections(text: str) -> list[dict[str, str]]:
    """
    Extract legal sections while preserving their full text.

    Supports:
    - numbered sections such as: 1 Title, 1-1 Title
    - legal articles such as: مادة (5)
    - BiDi-reversed article markers such as: مادة )١0(
    - OCR-noisy article markers such as: مادة (/ا), مادة (؟؟)
    - ignores official-gazette page headers
    - avoids numbered OCR noise in article-based documents
    """
    sections: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    preamble: list[str] = []

    article_based = _is_article_based(text)

    for raw_line in text.splitlines():
        original_line = raw_line.strip()

        if not original_line:
            continue

        normalized_line = _strip_bidi_controls(original_line)

        if PAGE_HEADER_PATTERN.match(normalized_line):
            continue

        article_match = _match_article(original_line)

        if article_match:
            if current is not None:
                current["text"] = current["text"].strip()
                sections.append(current)

            raw_number = article_match.group("number").strip()
            normalized_number = _normalize_arabic_digits(raw_number)

            current = {
                "number": normalized_number,
                "level": 1,
                "parent": None,
                "text": original_line,
            }

            continue

        if article_based:
            if current is not None:
                current["text"] += "\n" + original_line

            continue

        numbered_match = NUMBERED_SECTION_PATTERN.match(normalized_line)

        if numbered_match:
            if current is not None:
                current["text"] = current["text"].strip()
                sections.append(current)

            number = _normalize_arabic_digits(
                numbered_match.group("number")
            )
            parts = number.split("-")

            preamble_text = "\n".join(preamble).strip()

            section_text = original_line

            if preamble_text:
                section_text = f"{preamble_text}\n{section_text}"

            current = {
                "number": number,
                "level": len(parts),
                "parent": "-".join(parts[:-1]) if len(parts) > 1 else None,
                "text": section_text,
            }

            preamble = []

            continue

        if current is not None:
            current["text"] += "\n" + original_line
        else:
            preamble.append(original_line)

    if current is not None:
        current["text"] = current["text"].strip()
        sections.append(current)

    # Fallback: documents that contain no article markers and no numbered
    # section headers (e.g. product terms & conditions) would otherwise be
    # dropped entirely because all their text lands in the preamble. Emit a
    # single section holding the full text so the document is still indexed.
    if not sections and preamble:
        full_text = "\n".join(preamble).strip()
        if full_text:
            sections.append(
                {
                    "number": "1",
                    "level": 1,
                    "parent": None,
                    "text": full_text,
                }
            )

    return sections