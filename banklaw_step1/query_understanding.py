"""Lightweight query-understanding for banklaw-ai.

Understands a banking-law question and rewrites it into a few semantic
retrieval queries. No retrieval is performed here: no ChromaDB, no reranker,
no vector store. This module only understands and reformulates a question.

Uses the project's existing Ollama model (qwen3:8b) through
``banklaw_step1.llm``. When the model is unavailable or returns invalid JSON
it falls back to a minimal understanding derived from the raw question, so it
always degrades safely and stays offline-testable.

Multilingual by relying on the LLM for meaning (not keyword matching):
Egyptian Arabic, MSA, English, mixed Arabic/English, and simple Arabizi,
including colloquial and indirect/natural phrasings.
"""
from __future__ import annotations

import json
import re
from typing import Any

from banklaw_step1.llm import _client, _invoke

__all__ = ["analyze_question", "rewrite_queries"]

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_WORD_RE = re.compile(r"\w*\d\w*")

_PROMPT = """أنت محلل أسئلة متخصص في القوانين والتعليمات المصرفية المصرية.
فهم معنى السؤال ولا تعتمد على مطابقة كلمات مفتاحية. افهمه بالكامل قبل أن تصنع JSON.

افهم الأسئلة بلهجة مصرية عامية، فصحى، إنجليزية، مختلطة عربية/إنجليزية، عربيزي بسيطة،
والأسئلة غير المباشرة أو الطبيعية (مثل: "عايز أعرف..." أو "إيه اللي بيحصل لو...").

حلل:
- النية (ما يريده المستخدم: معرفة رسوم، شروط، إجراءات، أهلية، حدود/حد أقصى، تعريف،
  استثناء، مدة/إقلاق، وثيقة/استمنا، مقارنة، حق أو واجب).
- الموضوع العام (topic).
- المنتج أو الخدمة المصرفية (حساب، بطاقة ائتمان، قرض، تحويل أموال، صراف آلي،
  صرف عملات، تأمين، استثمار، بنك ديجيتال/موبايل بنكينج، شيك، إيداع/وديعة،
  سحب أكثر من الرصيد/أجرة سحب...).
- نوع المعلومات المطلوبة (قيمة محددة/مبلغ، شرط أو متطلب، إجراءات/خطوات،
  أهلية/مؤهل، استثناء/معفاة، مقارنة، تعريف/معنى).
- القيود أو الشروط المهمة (فرد أو شركة، مقيم أو غير مقيم، عملة محددة، حد أدنى
  أو أقصى، فترة زمنية...).
- اللغة الأصلية (Egyptian Arabic, MSA, English, mixed Arabic/English, or Arabizi).

أرجع JSON فقط، دون أي نص إضافي، بهذا الشكل بالضبط:
{
  "intent": "...",
  "topic": "...",
  "product_or_service": "...",
  "requested_information_type": "...",
  "constraints": ["...", "..."],
  "language": "..."
}

السؤال:"""


def analyze_question(question: str, conversation_context: Any = None) -> dict:
    """Understand a question and return a structured analysis.

    Returns a dict with keys: intent, topic, product_or_service,
    requested_information_type, constraints, language, and `question`
    (the normalized original question, for downstream reformulation).

    Uses qwen3:8b via banklaw_step1.llm; falls back to a minimal analysis
    if the model is unavailable or returns invalid JSON.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    text = question.strip()
    prompt = f"{_PROMPT}\n{text}"
    if conversation_context:
        context_str = str(conversation_context).strip()
        if context_str:
            prompt += f"\n\nالسياق المحادثة:\n{context_str[:1000]}"

    analysis: dict | None = None
    try:
        raw = _invoke(_client, prompt)
        data = _parse_llm_json(raw)
        if isinstance(data, dict):
            analysis = {
                "intent": _as_str(data.get("intent")),
                "topic": _as_str(data.get("topic")),
                "product_or_service": _as_str(data.get("product_or_service")),
                "requested_information_type": _as_str(
                    data.get("requested_information_type")
                ),
                "constraints": _as_list(data.get("constraints")),
                "language": _as_str(data.get("language")),
            }
    except Exception:
        analysis = None

    if analysis is None:
        analysis = _fallback_analysis(text)

    analysis["question"] = text
    return analysis


_PROMPT_REWRITE = """أنت كاتب طلبات بحث متخصص في القوانين والتعليمات المصرفية المصرية.
مهمتك صياغة 2-3 صيغ موجزة (rewrites) للاستعلام البحثي الذي سيُرسل إلى
متخزم الناهلي (vector store).

القواعد:
- اعكس معنى السؤال الأصلي تمامًا ولا تخترع قيود أو معلومات غير موجودة فيه.
- حافظ على الصفات الهامة: اليومي/أقصى/أدنى/رسوم/أهلية/فرد أو شركة/عملة/مدة.
- قد تكون إحدى الصيغ هي السؤال الأصلي.
- صغ ومختصر، كأنه طلب بحث قانوني.
- لا تكرر، ولا تأخذ أكثر من 3 صيغ.
- أنتج JSON فقط: ["صيغة 1", "صيغة 2", "صيغة 3"].

التحليل:
{analysis}

السؤال الأصلي:
{question}

الصيغ:"""


_DIAC_RE = re.compile(r"[\u064B-\u0652]")

# Minimal concrete-entity lexicons used ONLY to guard against LLM rewrites that
# invent facts (currency, amounts, dates, a different product) not present in
# the original question. They are NOT a classifier and stay intentionally small.
_CURRENCY = ("جنيه", "دولار", "يورو", "فرنك", "ريال", "درهم", "دينار", "لير",
             "شيكل", "كرون", "EGP", "USD", "EUR", "GBP")
_MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو",
           "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")
_DURATIONS = ("يوم", "أسبوع", "شهر", "سنة", "سنوي", "فوري")
_PRODUCTS = ("بطاقة", "قرض", "حساب", "تحويل", "شيك", "تأمين", "تامين",
             "استثمار", "وديعة", "إيداع", "تمويل", "سلف", "مرابحة")


def _norm_ar(text: str) -> str:
    text = _DIAC_RE.sub("", str(text or ""))
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text.lower()


def _tokens(text: str, vocab) -> set[str]:
    norm = _norm_ar(text)
    return {token for token in vocab if token in norm}


def _query_supported(original: str, query: str) -> bool:
    """True when `query` adds no concrete entity absent from `original`.

    Guards against rewrites that invent currency, numeric amounts, dates or a
    different product. Paraphrases that only restate the original's meaning are
    accepted. The original question itself is always supported.
    """
    if original == query:
        return True
    o = _norm_ar(original)
    q = _norm_ar(query)
    orig_digits = set(re.findall(r"\d+", o))
    query_digits = set(re.findall(r"\d+", q))
    if query_digits - orig_digits:
        return False
    for vocab in (_CURRENCY, _MONTHS, _DURATIONS, _PRODUCTS):
        if _tokens(q, vocab) - _tokens(o, vocab):
            return False
    return True


def rewrite_queries(analysis: dict, use_llm: bool = True) -> list[str]:
    """Build 2-3 concise semantic retrieval queries from an analysis.

    The user's ORIGINAL question is always query #1 (it carries every real
    qualifier). LLM reformulations are validated: any rewrite that adds a
    concrete entity (currency, amount, date or product) not present in the
    original is discarded whole. If not enough valid rewrites survive, the
    deterministic constraint-free fallback fills the remaining slots.

    No retrieval happens here.
    """
    if not analysis:
        return []

    original = _clean(analysis.get("question") or "")
    if not original:
        return []

    reformulations = []
    if use_llm:
        reformulations = _llm_rewrite(analysis, original) or []

    out: list[str] = [original]
    seen = {original.lower()}
    for query in reformulations:
        if not _query_supported(original, query):
            continue
        cleaned = _clean(query)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)

    if len(out) < 3:
        for query in _fallback_rewrite(analysis, original):
            cleaned = _clean(query)
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                out.append(cleaned)
            if len(out) >= 3:
                break

    return out[:3]


def _llm_rewrite(analysis, original):
    prompt = _PROMPT_REWRITE.format(
        analysis=json.dumps(_analysis_for_prompt(analysis), ensure_ascii=False),
        question=original,
    )
    try:
        raw = _invoke(_client, prompt)
        items = _parse_list(raw)
    except Exception:
        return None
    if not items:
        return None
    cleaned = [_clean(x) for x in items if isinstance(x, str)]
    cleaned = [x for x in cleaned if x]
    return cleaned or None


def _fallback_rewrite(analysis, original):
    # Constraint-free combinations only. The original question (returned
    # separately by the caller) already carries the real qualifiers/constraints,
    # so we never emit invented ones like "عملة محددة".
    product = _clean(analysis.get("product_or_service") or "")
    info = _clean(analysis.get("requested_information_type") or "")
    topic = _clean(analysis.get("topic") or "")
    intent = _clean(analysis.get("intent") or "")

    candidates = []
    if product and info:
        candidates.append(f"{product} {info}")
    if intent and topic:
        candidates.append(f"{intent} {topic}")
    if product and intent:
        candidates.append(f"{product} {intent}")

    seen: set[str] = set()
    out: list[str] = []
    for query in candidates:
        if query and query.lower() not in seen:
            seen.add(query.lower())
            out.append(query)
    return out[:2]


def _analysis_for_prompt(analysis):
    return {
        "intent": analysis.get("intent", ""),
        "topic": analysis.get("topic", ""),
        "product_or_service": analysis.get("product_or_service", ""),
        "requested_information_type": analysis.get("requested_information_type", ""),
        "constraints": analysis.get("constraints") or [],
        "language": analysis.get("language", ""),
        "question": analysis.get("question", ""),
    }


def _parse_list(text):
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
    return data if isinstance(data, list) else None


def _fallback_analysis(question: str) -> dict:
    return {
        "intent": "general_inquiry",
        "topic": "",
        "product_or_service": "",
        "requested_information_type": "unspecified",
        "constraints": [],
        "language": _detect_language(question),
    }


def _detect_language(text: str) -> str:
    has_arabic = bool(_ARABIC_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    has_digit_word = bool(_DIGIT_WORD_RE.search(text))
    if has_arabic and has_latin:
        return "mixed Arabic/English"
    if has_arabic:
        return "Arabic"
    if has_latin and has_digit_word:
        return "Arabizi"
    if has_latin:
        return "English"
    return "unknown"


def _parse_llm_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except (ValueError, TypeError):
            return None


def _as_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())
