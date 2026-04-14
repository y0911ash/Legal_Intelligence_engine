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
    r"(?:₹|Rs\.?|INR|Rupees)\s*([\d, ]+(?:\s*(?:lakhs?|crores?|thousands?))?)"
    r"(?:\s*/?\\-?)?",
    re.IGNORECASE,
)

_CATEGORY_CONTEXT = {
    "fine": re.compile(r"(?i)(fine|fined|imposed|penalty|default|amounting)\b"),
    "compensation": re.compile(r"(?i)(compensation|compensate|damages|solatium|ex\s*-?\s*gratia)\b"),
    "penalty": re.compile(r"(?i)(penalt(y|ies)|penalised|penalized|surcharge)\b"),
    "costs": re.compile(r"(?i)(costs?|court\s+fee|litigation\s+cost)\b"),
}

_FORBIDDEN_PREFIXES = [
    r"\bsection\b", r"\barticle\b", r"\bcase\b", r"\bpetition\b", 
    r"\bno\.\b", r"\bno\b", r"\bwrit\b", r"\bact\b", r"\bdated\b", r"\byear\b"
]
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
    """A version that doesn't break on Rs. or No. but finds actual sentence breaks."""
    # We look for punctuation followed by space/newline, or end of string.
    # This prevents 'Rs.' from triggering a boundary.
    pattern = re.compile(r'[.!?](\s+|$|\n)')
    if direction == 'back':
        m = list(pattern.finditer(text))
        return text[m[-1].end():] if m else text
    m = pattern.search(text)
    return text[:m.start()] if m else text


def _is_forbidden_context(text: str) -> bool:
    """Triggers if forbidden word is directly before amount (last 20 chars)."""
    prefix = text[-20:].lower()
    return any(re.search(f, prefix) for f in _FORBIDDEN_PREFIXES)


def _is_semantic_match(category: str, context: str) -> bool:
    query = f"Is this a significant monetary amount, {category}, or legal value relevant to the case judgment?"
    model = _get_cross_encoder()
    if model is None:
        return True
    # CrossEncoder.predict usually expects a list of pairs [(q, c)]
    # We use an extremely permissive threshold because missing a financial
    # implication is worse than a false positive in a legal summary.
    scores = model.predict([(query, context)])
    return float(scores[0]) > -5.0


def _classify_amount(full_text: str, start: int, end: int) -> tuple:
    raw_back = full_text[max(0, start - _CONTEXT_BACK):start]
    is_suspect = _is_forbidden_context(raw_back)

    raw_fwd = full_text[end:end + _CONTEXT_FWD]

    best_cat, best_dist, found_kw = "fine", float("inf"), False

    for cat, pattern in _CATEGORY_CONTEXT.items():
        for m in pattern.finditer(raw_back):
            found_kw = True
            dist = len(raw_back) - m.end()
            if dist < best_dist:
                best_dist, best_cat = dist, cat
        for m in pattern.finditer(raw_fwd):
            found_kw = True
            if m.start() < best_dist:
                best_dist, best_cat = m.start(), cat

    # If no keyword found AND bouncer suspect, it's definitely non-legal
    if not found_kw and is_suspect:
        return "non_legal_mention", False
    
    return best_cat, found_kw


def extract_financials(text: str) -> Dict:
    result = {"fine": [], "compensation": [], "penalty": [], "costs": [], "raw_mentions": []}
    seen = set()

    for match in _AMOUNT_PATTERN.finditer(text):
        amount_str = f"₹{match.group(1)}".strip()
        if amount_str in seen:
            continue
        seen.add(amount_str)

        category, has_keyword = _classify_amount(text, match.start(), match.end())

        # Give AI a wide 200-char view for the final verdict
        ctx_start = max(0, match.start() - 100)
        ctx_end = min(len(text), match.end() + 100)
        snippet = text[ctx_start:ctx_end].strip()

        # If we have a strong keyword match or legal context, we trust it.
        # We only use semantic match as an extra 'rescue' for suspicious amounts.
        check_cat = category if category != "non_legal_mention" else "fine"
        verified = has_keyword or _is_semantic_match(check_cat, snippet)

        detail = {"amount": amount_str, "context": f"...{snippet}..."}

        if verified and category != "non_legal_mention":
            result[category].append(detail)
        elif verified:
            result["fine"].append(detail)
            category = "fine (verified)"

        result["raw_mentions"].append({"amount": amount_str, "category": category, "context": snippet})

    return result
