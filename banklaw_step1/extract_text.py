from pathlib import Path
import io
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pypdf import PdfReader


TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def _configure_tesseract() -> None:
    """Configure the Tesseract executable explicitly on Windows."""
    if not TESSERACT_PATH.is_file():
        raise FileNotFoundError(
            f"Tesseract executable was not found at: {TESSERACT_PATH}"
        )

    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)


def _ocr_page(document: fitz.Document, page_index: int) -> str:
    """OCR a single PDF page using Arabic Tesseract."""
    _configure_tesseract()

    page = document[page_index]
    pixmap = page.get_pixmap(dpi=200, alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))

    return pytesseract.image_to_string(
        image,
        lang="ara",
        config="--psm 3",
    ).strip()


def extract_pdf_text(pdf_path: str | Path) -> str:
    """
    Extract PDF text with pypdf as the primary source.

    OCR is used only for pages whose pypdf output is empty or
    fails Arabic text validation.
    """
    path = Path(pdf_path)
    reader = PdfReader(str(path))

    page_texts: list[str] = []

    with fitz.open(str(path)) as document:
        for page_index, page in enumerate(reader.pages):
            extracted_text = (page.extract_text() or "").strip()

            if is_valid_arabic_text(extracted_text):
                page_texts.append(extracted_text)
                continue

            ocr_text = _ocr_page(document, page_index)
            page_texts.append(ocr_text)

    return "\n\n".join(page_texts).strip()


def is_valid_arabic_text(text: str) -> bool:
    """Check whether extracted text contains meaningful, non-corrupted Arabic."""
    if not text.strip():
        return False

    arabic_chars = sum(
        1
        for char in text
        if "\u0600" <= char <= "\u06ff"
    )

    non_whitespace_chars = sum(
        1
        for char in text
        if not char.isspace()
    )

    if non_whitespace_chars == 0:
        return False

    arabic_ratio = arabic_chars / non_whitespace_chars

    if arabic_ratio < 0.20:
        return False

    if re.search(r"([\u0600-\u06ff])\1{3,}", text):
        return False

    return True