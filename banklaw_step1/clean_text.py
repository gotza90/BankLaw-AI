import re

COMMON_ARABIC_MERGES = {
    "المصريبتطوير": "المصري بتطوير",
    "التأكيدعلى": "التأكيد على",
    "يتعينعلى": "يتعين على",
    "كمايحق": "كما يحق",
    "العملاءوفقا": "العملاء وفقا",
    "فيمايتعلق": "فيما يتعلق",
}

def clean_text(text: str) -> str:
    """
    Clean raw PDF text while preserving the legal structure.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Remove standalone page numbers
        if re.fullmatch(r"\d+", line):
            continue

        # Remove known CBE footnotes
        if re.match(
    r"^[12]\s+كتاب السيد\s*(?:/\s*)?محافظ البنك المركزي",
    line,
):
            continue

        # Remove a PDF footnote marker accidentally attached
        # to the end of a heading/title.
        line = re.sub(r"(?<=[\u0600-\u06FF])1$", "", line)

       # Normalize repeated whitespace
        line = re.sub(r"[ \t]+", " ", line)

        for merged, separated in COMMON_ARABIC_MERGES.items():
            line = line.replace(merged, separated)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)