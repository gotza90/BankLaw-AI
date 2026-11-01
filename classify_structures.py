from pathlib import Path
import re

from pypdf import PdfReader


ARTICLE_PATTERN = re.compile(r"^\s*مادة\s*\(")
NUMBERED_PATTERN = re.compile(r"^\s*\d+(?:-\d+)*\s+\S+")


data_dir = Path("data")

for pdf in sorted(data_dir.glob("*.pdf")):
    reader = PdfReader(str(pdf))

    article_count = 0
    numbered_count = 0

    for page in reader.pages:
        text = page.extract_text() or ""

        for line in text.splitlines():
            line = line.strip()

            if ARTICLE_PATTERN.match(line):
                article_count += 1

            if NUMBERED_PATTERN.match(line):
                numbered_count += 1

    print(
        f"{pdf.name}\n"
        f"  pages={len(reader.pages)} "
        f"article_markers={article_count} "
        f"numbered_markers={numbered_count}\n"
    )
