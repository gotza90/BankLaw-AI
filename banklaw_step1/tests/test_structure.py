from banklaw_step1.structure import extract_sections
from pathlib import Path

from banklaw_step1.extract_text import extract_pdf_text

def test_extracts_numbered_legal_sections():
    text = """
1- مفهوم حماية حقوق عملاء البنوك وأهم التعريفات:
يُقصد بالمفهوم العام لحماية حقوق عملاء البنوك وفقاً لأفضل الممارسات الدولية وجود إطار تنظيمي واضح.

1-1 العملاء
يُقصد بهم الأشخاص الطبيعيين والاعتباريين الذين يتعاملون مع القطاع المصرفي.

2- نطاق التطبيق
تسري هذه التعليمات على كافة الخدمات والمنتجات المصرفية.

3- قواعد عامة
توجد مجموعة من القواعد العامة.

3-1 يتعين على مجلس إدارة البنك التأكد من الالتزام.
"""

    sections = extract_sections(text)

    assert len(sections) == 5

    assert sections[0]["number"] == "1"
    assert "مفهوم حماية حقوق عملاء البنوك" in sections[0]["text"]

    assert sections[1]["number"] == "1-1"
    assert "العملاء" in sections[1]["text"]

    assert sections[2]["number"] == "2"
    assert "نطاق التطبيق" in sections[2]["text"]

    assert sections[3]["number"] == "3"
    assert "قواعد عامة" in sections[3]["text"]

    assert sections[4]["number"] == "3-1"
    assert "مجلس إدارة البنك" in sections[4]["text"]


def test_sections_have_hierarchy():
    text = """
4- قواعد حماية حقوق عملاء البنوك
4-1 التعامل مع العملاء
4-1-8 العقود والنماذج
4-1-8-9 تصعيد الشكوى إلى البنك المركزي المصري.
"""

    sections = extract_sections(text)

    assert sections[0]["number"] == "4"
    assert sections[0]["level"] == 1
    assert sections[0]["parent"] is None

    assert sections[1]["number"] == "4-1"
    assert sections[1]["level"] == 2
    assert sections[1]["parent"] == "4"

    assert sections[2]["number"] == "4-1-8"
    assert sections[2]["level"] == 3
    assert sections[2]["parent"] == "4-1"

    assert sections[3]["number"] == "4-1-8-9"
    assert sections[3]["level"] == 4
    assert sections[3]["parent"] == "4-1-8"



def test_law_194_extracts_article_sections_without_page_header_numbers():
    pdf = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "CBE Law No. 194 of 2020.pdf"
    )

    text = extract_pdf_text(pdf)
    sections = extract_sections(text)

    article_5 = [
        section
        for section in sections
        if section["number"] == "5"
        and section["text"].lstrip().startswith("مادة (5)")
    ]

    assert article_5, "Law 194 Article 5 was not extracted as a legal section"

    assert not any(
        section["text"].lstrip().startswith("5 الجريدة الرسمية")
        for section in sections
    )


def test_law_194_keeps_articles_separate():
    from pathlib import Path

    from banklaw_step1.extract_text import extract_pdf_text
    from banklaw_step1.structure import extract_sections

    pdf = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "CBE Law No. 194 of 2020.pdf"
    )

    text = extract_pdf_text(pdf)
    sections = extract_sections(text)

    article_5 = [
        section
        for section in sections
        if section["number"] == "5"
        and section["text"].lstrip().startswith("مادة (5)")
    ]

    assert article_5, "Article 5 was not found"

    assert not any(
        "مادة (5):" in section["text"]
        and "مادة (/ا):" in section["text"]
        for section in sections
    )

def test_article_based_documents_ignore_numeric_ocr_noise():
    from banklaw_step1.structure import extract_sections

    text = """
    5171 سبتمبر سنة 2020 الجريدة الرسمية - العدد 11
    مادة (5):
    يهدف البنك المركزى إلى سلامة النظام النقدى والمصرفى.

    مادة (/ا):
    يباشر البنك المركزى جميع الاختصاصات اللازمة لتحقيق أهدافه.

    7171 سبتمبر سنة 2020 الجريدة الرسمية - العدد 12
    مادة (9):
    للبنك المركزى أن يقدم التمويل للجهات التى يساهم فيها.
    """

    sections = extract_sections(text)

    assert len(sections) == 3

    assert all(
        section["text"].lstrip().startswith("مادة")
        for section in sections
    )

    assert [section["number"] for section in sections] == [
        "5",
        "/ا",
        "9",
    ]


def test_article_parser_handles_bidi_reversed_parentheses():
    from banklaw_step1.structure import extract_sections

    text = """
    مادة (14):
    يختص البنك المركزى بجمع وتحليل البيانات والمعلومات.

    مادة \u200e)١0(\u200f
    للبنك المركزى أن يحتفظ ضمن الاحتياطيات الأجنبية بالأصول المقررة.

    مادة (19):
    يمثل المحافظ البنك المركزى أمام القضاء.
    """

    sections = extract_sections(text)

    assert len(sections) == 3

    assert sections[0]["number"] == "14"
    assert sections[1]["number"] == "10"
    assert sections[2]["number"] == "19"
