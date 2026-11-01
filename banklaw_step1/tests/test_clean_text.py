from banklaw_step1.clean_text import clean_text


def test_clean_text_preserves_legal_numbers():
    raw = """
1

الفصل الأول
تعليمات حماية حقوق عملاء البنوك1

2

4-1 التعامل مع العملاء بمبدأ العدل والإنصاف

4-1-8-9 الإشارة إلى إمكانية تصعيد الشكوى إلى البنك المركزي المصري.
"""

    cleaned = clean_text(raw)

    assert "الفصل الأول" in cleaned
    assert "4-1" in cleaned
    assert "4-1-8-9" in cleaned
    assert "البنك المركزي المصري" in cleaned

    assert "\n1\n" not in cleaned
    assert "\n2\n" not in cleaned


def test_clean_text_removes_known_footnote():
    raw = """
4-2-13 تلتزم البنوك لدى اعداد عقود المنتجات والخدمات المصرفية.

2 كتاب السيد / محافظ البنك المركزي المصري بتاريخ24 ديسمبر2024
"""

    cleaned = clean_text(raw)

    assert "4-2-13" in cleaned
    assert "محافظ البنك المركزي المصري" not in cleaned


def test_clean_text_removes_real_cbe_footnotes():
    raw = """
1 كتاب السيد محافظ البنك المركزي الصادر بتاريخ19 فبراير2019

4-2-13 تلتزم البنوك لدى اعداد عقود المنتجات والخدمات المصرفية.

2 كتاب السيد / محافظ البنك المركزي المصري بتاريخ24 ديسمبر2024

4-2-13-1 أن توضح هذه النماذج بشكل مبسط جميع المصاريف والعمولات.
"""

    cleaned = clean_text(raw)

    assert "كتاب السيد محافظ البنك المركزي" not in cleaned
    assert "كتاب السيد / محافظ البنك المركزي المصري" not in cleaned

    assert "4-2-13" in cleaned
    assert "4-2-13-1" in cleaned


def test_clean_text_fixes_common_arabic_pdf_word_merges():
    raw = """
في إطار الاهتمام الذي يوليه البنك المركزي المصريبتطوير القطاع المصرفي.

يتعينعلى البنك الإفصاح عن البيانات.

كمايحق للعميل تقديم شكوى.
"""

    cleaned = clean_text(raw)

    assert "المصري بتطوير" in cleaned
    assert "يتعين على" in cleaned
    assert "كما يحق" in cleaned