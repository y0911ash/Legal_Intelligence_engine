"""
Phase 6: Financial Intelligence Extractor
-------------------------------------------
Scans judgment text for monetary amounts and verifies them using:
1. Surgical Bouncer (Negative Lookbehind for Sections/Case IDs)
2. Semantic BERT Verifier (Cross-Encoder for Deep Context Analysis)

Fixes applied (2026-04-16):
- Corrected double-escaped backslash in _AMOUNT_PATTERN (\\\\- → \\-)
- Preserved original currency prefix in extracted amount label
- Replaced over-aggressive set-based dedup with (amount, position) keying
  so the same amount at two different positions is correctly captured
- _trim_at_boundary is now used to build cleaner context snippets
- Unverified / non-legal mentions are routed only to raw_mentions, not fine
"""

import re
from typing import Dict, List, Tuple

# ── Regex: matches Rs., Rs, ₹, INR, Rupees followed by a number (commas ok)
# with optional lakh/crore/thousand suffix and optional trailing /- marker.
# BUG FIX: was  r"(?:\s*/?\\\\-?)?"  (four backslashes → literal \\-)
#          now  r"(?:\s*/?\\-?)?"    (two backslashes → literal \-)
_AMOUNT_PATTERN = re.compile(
    r"((?:₹|Rs\.?|INR|Rupees)\s*[\d,]+(?:\s*(?:lakhs?|crores?|thousands?))?)"
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
    r"\bno\.\b", r"\bno\b", r"\bwrit\b", r"\bact\b", r"\bdated\b", r"\byear\b",
]
_CONTEXT_BACK = 120
_CONTEXT_FWD = 60

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
    """
    Trim a raw context window at a real sentence boundary.
    Avoids splitting on abbreviations like 'Rs.' or 'No.' by requiring
    the period to be followed by whitespace or end-of-string.
    """
    pattern = re.compile(r'[.!?](\s+|$|\n)')
    if direction == 'back':
        matches = list(pattern.finditer(text))
        return text[matches[-1].end():] if matches else text
    m = pattern.search(text)
    return text[:m.start()] if m else text


def _is_forbidden_context(text: str) -> bool:
    """Returns True if a forbidden (non-monetary) keyword appears immediately before the amount."""
    prefix = text[-20:].lower()
    return any(re.search(f, prefix) for f in _FORBIDDEN_PREFIXES)


def _is_semantic_match(category: str, context: str) -> bool:
    query = f"Is this a significant monetary amount, {category}, or legal value relevant to the case judgment?"
    model = _get_cross_encoder()
    if model is None:
        return True  # Heuristic fallback: trust the amount
    # Very permissive threshold — missing a financial mention is worse than a FP.
    scores = model.predict([(query, context)])
    return float(scores[0]) > -5.0


def _classify_amount(full_text: str, start: int, end: int) -> Tuple[str, bool]:
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

    # No keyword + bouncer flagged it → definitely a non-legal reference
    if not found_kw and is_suspect:
        return "non_legal_mention", False

    return best_cat, found_kw


def extract_financials(text: str) -> Dict:
    """
    Extract monetary amounts from legal judgment text and categorise them.

    Returns a dict with keys: fine, compensation, penalty, costs, raw_mentions.
    Each entry under a category is {"amount": str, "context": str}.
    raw_mentions also includes a "category" field for debugging.
    """
    result: Dict[str, List] = {
        "fine": [], "compensation": [], "penalty": [], "costs": [], "raw_mentions": []
    }

    # Deduplicate by (normalised_amount, span_start) so the same figure at two
    # different positions (e.g. an advance payment vs. a sentence fine) is kept,
    # but the exact same match isn't counted twice.
    seen: set = set()

    for match in _AMOUNT_PATTERN.finditer(text):
        # FIX: preserve original prefix (Rs., ₹, INR …) instead of always "₹"
        amount_str = match.group(1).strip()
        key = (amount_str, match.start())
        if key in seen:
            continue
        seen.add(key)

        category, has_keyword = _classify_amount(text, match.start(), match.end())

        # Build a clean 200-char context window using the boundary trimmer
        raw_back = text[max(0, match.start() - 100):match.start()]
        raw_fwd  = text[match.end():min(len(text), match.end() + 100)]
        clean_back = _trim_at_boundary(raw_back, 'back')
        clean_fwd  = _trim_at_boundary(raw_fwd, 'fwd')
        snippet = (clean_back + match.group(0) + clean_fwd).strip()

        # Semantic rescue: only call the cross-encoder for suspicious amounts
        if not has_keyword and not (category == "non_legal_mention"):
            verified = _is_semantic_match(category, snippet)
        else:
            verified = has_keyword

        detail = {"amount": amount_str, "context": f"...{snippet}..."}

        if verified and category != "non_legal_mention":
            result[category].append(detail)
        # FIX: do NOT push non-legal/unverified amounts into "fine" —
        #      let them stay in raw_mentions only so the UI stays clean.

        result["raw_mentions"].append({
            "amount": amount_str,
            "category": category if (verified or category == "non_legal_mention") else "unverified",
            "context": snippet,
        })

    return result
