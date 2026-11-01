from pathlib import Path

from extract_text import extract_pdf_text

PDF = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "الفصل الأول - الباب الرابع عشر - حقوق العملاء.pdf"
)


def test_extracts_arabic_text_from_customer_rights_pdf():
    text = extract_pdf_text(PDF)
    assert len(text.strip()) > 500
    assert 'حماية حقوق عملاء البنوك' in text


def test_extracts_readable_arabic_from_cbe_law_194():
    pdf = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "CBE Law No. 194 of 2020.pdf"
    )

    text = extract_pdf_text(pdf)

    assert len(text) > 1000
    assert "الجهات الاجنبية المناظرة" in text or "الجهات الأجنبية المناظرة" in text
    assert "الشركات التابعة" in text
    assert "الشركات الشقيقة" in text


from banklaw_step1.extract_text import is_valid_arabic_text


def test_arabic_text_validator_rejects_garbled_pdf_text():
    valid_text = """
    الجهات الأجنبية المناظرة هي الجهات التي تمارس اختصاصات مماثلة.
    الشركات التابعة هي الشركات التي يمتلك البنك القدرة على التحكم في سياستها.
    """

    garbled_text = """
    UNOM/EOT ¥ sL/ESC/DC3«
    W‡M‡‡/US/DC3«
    Êu/SUB/US/DC3«Ë W/ESC/DC3U
    """

    assert is_valid_arabic_text(valid_text)
    assert not is_valid_arabic_text(garbled_text)


def test_extract_text_falls_back_to_ocr_for_garbled_pdf():
    pdf = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "CBE Law No. 194 of 2020.pdf"
    )

    text = extract_pdf_text(pdf)

    assert is_valid_arabic_text(text)
