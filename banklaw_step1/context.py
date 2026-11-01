def _is_meaningful_section(section):
    """
    Return True when `section` carries a usable legal reference.

    The structure parser sometimes emits OCR-noise section numbers for
    chunks where it could not parse a legal article marker, for example
    "؟", "2:؟" or "1517/". Those values are useless as citations and, when
    they leak into the prompt, let the LLM attach fabricated claims to a
    bogus "[Citation: ... | Section: ؟]" label.

    A section is meaningful only if, after stripping bidi control marks,
    the Arabic tatweel and surrounding whitespace, it still contains at
    least one Arabic letter or decimal digit.
    """
    if section is None:
        return False

    cleaned = (
        section
        .replace("\u200e", "")   # LEFT-TO-RIGHT MARK
        .replace("\u200f", "")   # RIGHT-TO-LEFT MARK
        .replace("\u0640", "")   # ARABIC TATWEEL
        .strip()
    )

    if not cleaned:
        return False

    return any("\u0621" <= ch <= "\u064a" or ch.isdigit() for ch in cleaned)


def build_context(results):
    """
    Build citation-aware context objects from ranked retrieval results.

    Each result becomes a context item carrying its text plus a citation
    that identifies the document, section and authority. Section numbers
    that are pure OCR noise are replaced with an explicit "unavailable"
    marker so the LLM never receives a fabricated-looking but meaningless
    section reference.
    """
    context = []

    for result in results:
        metadata = result["metadata"]
        section = metadata.get("section_number")

        context.append(
            {
                "text": result["page_content"],
                "citation": {
                    "document": metadata.get("document"),
                    "section": section if _is_meaningful_section(section) else "غير متاح",
                    "source": metadata.get("source"),
                },
            }
        )

    return context