"""
Phase 6: Financial Intelligence Extractor
-------------------------------------------
Scans judgment text for monetary amounts and verifies them using:
1. Surgical Bouncer (Negative Lookbehind for Sections/Case IDs)
2. Semantic BERT Verifier (Cross-Encoder for Deep Context Analysis)
"""

import re
from typing import Dict, List

_AMOUNT_PATTERN = re.compile(
    r"(?:₹|Rs\.?|INR|Rupees)\s*(\d{1,2}(?:,\d{2})*,\d{3}|\d+)"
    r"(?:\s*/?\\-?)?"
    r"(?:\s*(?:lakhs?|crores?|thousands?))?",
    re.IGNORECASE,
)

_CATEGORY_CONTEXT = {
    "fine": re.compile(r"(?i)(fine|fined|imposed|penalty|default|amounting)\b"),
    "compensation": re.compile(r"(?i)(compensation|compensate|damages|solatium|ex\s*-?\s*gratia)\b"),
    "penalty": re.compile(r"(?i)(penalt(y|ies)|penalised|penalized|surcharge)\b"),
    "costs": re.compile(r"(?i)(costs?|court\s+fee|litigation\s+cost)\b"),
}

_FORBIDDEN_PREFIXES = {"section", "article", "case", "petition", "no.", "no", "writ", "act", "dated", "year"}
_CONTEXT_BACK = 120
_CONTEXT_FWD = 60
_SENTENCE_END = re.compile(r'[.!?]')

# Cross-Encoder (lazy-loaded). If unavailable, we fall back to heuristics.
_CROSS_ENCODER = None
_CROSS_ENCODER_FAILED = False


def _get_cross_encoder():
    global _CROSS_ENCODER, _CROSS_ENCODER_FAILED
    if _CROSS_ENCODER_FAILED:
        return None
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder

            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        except Exception as exc:
            print(f"[Financial Extractor] Cross-encoder unavailable, falling back to heuristics: {exc}")
            _CROSS_ENCODER_FAILED = True
            return None
    return _CROSS_ENCODER


def _trim_at_boundary(text: str, direction: str) -> str:
    if direction == 'back':
        m = list(_SENTENCE_END.finditer(text))
        return text[m[-1].end():] if m else text
    m = _SENTENCE_END.search(text)
    return text[:m.start()] if m else text


def _is_forbidden_context(text: str) -> bool:
    """Triggers if forbidden word is directly before amount (last 15 chars)."""
    prefix = text[-15:].lower()
    return any(f in prefix for f in _FORBIDDEN_PREFIXES)


def _is_semantic_match(category: str, context: str) -> bool:
    query = f"Is this a significant monetary amount, {category}, or legal value relevant to the case judgment?"
    model = _get_cross_encoder()
    if model is None:
        return True
    return model.predict([query, context]) > 0.01


def _classify_amount(full_text: str, start: int, end: int) -> str:
    raw_back = full_text[max(0, start - _CONTEXT_BACK):start]
    is_suspect = _is_forbidden_context(raw_back)

    back_ctx = _trim_at_boundary(raw_back, 'back')
    fwd_ctx = _trim_at_boundary(full_text[end:end + _CONTEXT_FWD], 'forward')

    best_cat, best_dist, found_kw = "fine", float("inf"), False

    for cat, pattern in _CATEGORY_CONTEXT.items():
        for m in pattern.finditer(back_ctx):
            found_kw = True
            dist = len(back_ctx) - m.end()
            if dist < best_dist:
                best_dist, best_cat = dist, cat
        for m in pattern.finditer(fwd_ctx):
            found_kw = True
            if m.start() < best_dist:
                best_dist, best_cat = m.start(), cat

    return "non_legal_mention" if (not found_kw and is_suspect) else best_cat


def extract_financials(text: str) -> Dict:
    result = {"fine": [], "compensation": [], "penalty": [], "costs": [], "raw_mentions": []}
    seen = set()

    for match in _AMOUNT_PATTERN.finditer(text):
        amount_str = f"₹{match.group(1)}"
        if amount_str in seen:
            continue
        seen.add(amount_str)

        category = _classify_amount(text, match.start(), match.end())

        ctx_start = max(0, match.start() - 100)
        ctx_end = min(len(text), match.end() + 100)
        snippet = text[ctx_start:ctx_end].strip()

        check_cat = category if category != "non_legal_mention" else "fine"
        verified = _is_semantic_match(check_cat, snippet)

        detail = {"amount": amount_str, "context": f"...{snippet}..."}

        if verified and category != "non_legal_mention":
            result[category].append(detail)
        elif verified:
            result["fine"].append(detail)
            category = "fine (verified)"

        result["raw_mentions"].append({"amount": amount_str, "category": category, "context": snippet})

    return result
