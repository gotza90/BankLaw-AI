from banklaw_step1.extract_text import is_valid_arabic_text
import re


def test_corrupted_arabic_text_is_rejected():
    corrupted_text = """
    قرررمجلسرردمرة رلم ك ارررم كجرلرررىم كجسرررىم لسقرر بم كجا  رةلم
    فلام لرررررةتمقرررررةم ررررر م ةق ررررراملرررررر ممعسرررررىم ك اررررررم
    """

    assert not is_valid_arabic_text(corrupted_text)


def test_normal_arabic_text_is_accepted():
    normal_text = """
    تلتزم البنوك بتزويد مقدم الشكوى بالرقم المرجعي للشكوى.
    ويمكن تقديم الشكوى كتابياً أو إلكترونياً وفقاً للقنوات المتاحة.
    """

    assert is_valid_arabic_text(normal_text)


def test_arabic_text_with_normal_repeated_letters_is_accepted():
    normal_text = """
    يقرر مجلس الإدارة القواعد المنظمة للعمل.
    تتضمن التعليمات الإجراءات والتزامات البنوك تجاه العملاء.
    """

    assert is_valid_arabic_text(normal_text)

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

    # Long runs of the same Arabic character are a strong
    # indicator of corrupted PDF text extraction.
    if re.search(r"([\u0600-\u06ff])\1{3,}", text):
        return False

    return True
