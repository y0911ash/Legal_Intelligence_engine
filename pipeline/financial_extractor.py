"""
Phase 6: Financial Intelligence Extractor
-------------------------------------------
Scans judgment text for monetary amounts and verifies them using:
1. Surgical Bouncer (Negative Lookbehind for Sections/Case IDs)
2. Semantic BERT Verifier (Cross-Encoder for Deep Context Analysis)

Fixes applied (2026-04-16):
- CRITICAL: Added word-boundary assertion before currency prefixes.
  Without this, 'Rs' inside 'Ors', 'others', 'vers', 'orders' etc. matched
  as a currency prefix — causing hundreds of false positives on any PDF.
- Require a mandatory digit group of 3+ digits so a bare '₹' or 'Rs.'
  with no number is never captured.
- No newlines allowed between currency symbol and digits (PDF garbling
  causes ₹ to drift away from its actual number across line breaks).
- Minimum plausible amount: ≥ 100 (three digits). Legal monetary amounts
  in Indian courts are almost always ≥ Rs. 100; paragraph numbers, ages,
  section numbers, years etc. are typically < 100 or they don't follow
  a currency prefix in a tight span.
- Replaced over-aggressive set-based dedup with (amount, position) keying.
- _trim_at_boundary is now used to build cleaner context snippets.
- Unverified / non-legal mentions stay in raw_mentions only.
"""

import re
from typing import Dict, List, Tuple

# ── Regex ────────────────────────────────────────────────────────────────────
# KEY FIXES vs previous versions:
#   1. (?<!\w)  → word-boundary lookbehind so 'rs' inside 'Ors'/'others' is
#                 never treated as a currency prefix.
#   2. [^\S\n]* → only horizontal whitespace between prefix and digits, so
#                 a ₹ on one line doesn't grab a number on the next line.
#   3. [\d,]{3,}→ require at least 3 consecutive digit/comma chars → rules
#                 out paragraph numbers and section refs like '21', '139'.
#   4. (?!\w)   → no letter directly after digits (avoids '302IPC' etc.)
# ─────────────────────────────────────────────────────────────────────────────
_AMOUNT_PATTERN = re.compile(
    r"""
    (?<!\w)                          # must NOT be preceded by a word char
    (
        (?:₹|Rs\.?|INR|Rupees)       # currency prefix
        [^\S\n]*                     # optional horizontal space (no newlines)
        [\d,]{3,}                    # at least 3 digit/comma chars
        (?:\.\d{1,2})?               # optional decimal (e.g. 1,50,000.00)
        (?:
            [^\S\n]*                 # optional space
            (?:lakhs?|crores?|thousands?|lacs?)  # optional word suffix
        )?
    )
    (?!\w)                           # must NOT be followed by a word char
    (?:[^\S\n]*/?-?)?                # optional trailing /- marker
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Minimum numeric value to be considered a plausible legal amount.
# Filtering ≤ 99 removes ages, years, section numbers, paragraph numbers.
_MIN_AMOUNT = 100


def _parse_numeric(amount_str: str) -> float:
    """
    Extract the bare numeric value from an amount string like 'Rs. 5,00,000'
    or '₹1 lakh'. Returns 0.0 on parse failure.
    """
    # Strip currency prefix and suffix words
    s = re.sub(r"(?i)(₹|Rs\.?|INR|Rupees|lakhs?|crores?|thousands?|lacs?)", "", amount_str)
    s = re.sub(r"[,\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return 0.0


_CATEGORY_CONTEXT = {
    "fine": re.compile(r"(?i)\b(fine|fined|imposed|penalty|default|amounting)\b"),
    "compensation": re.compile(r"(?i)\b(compensation|compensate|damages|solatium|ex\s*-?\s*gratia)\b"),
    "penalty": re.compile(r"(?i)\b(penalt(?:y|ies)|penalised|penalized|surcharge)\b"),
    "costs": re.compile(r"(?i)\b(costs?|court\s+fee|litigation\s+cost)\b"),
}

_FORBIDDEN_PREFIXES = [
    r"\bsection\b", r"\barticle\b", r"\bcase\b", r"\bpetition\b",
    r"\bno\.\b", r"\bno\b", r"\bwrit\b", r"\bact\b", r"\bdated\b", r"\byear\b",
]
_CONTEXT_BACK = 120
_CONTEXT_FWD = 60

# Cross-Encoder (lazy-loaded).
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
    Requires period to be followed by whitespace/newline so 'Rs.' and 'No.'
    don't trigger a spurious trim.
    """
    pattern = re.compile(r"[.!?](\s+|$|\n)")
    if direction == "back":
        matches = list(pattern.finditer(text))
        return text[matches[-1].end():] if matches else text
    m = pattern.search(text)
    return text[: m.start()] if m else text


def _is_forbidden_context(text: str) -> bool:
    """True if a forbidden keyword appears in the 20 chars immediately before the amount."""
    prefix = text[-20:].lower()
    return any(re.search(f, prefix) for f in _FORBIDDEN_PREFIXES)


def _is_semantic_match(category: str, context: str) -> bool:
    query = (
        f"Is this a significant monetary amount, {category}, "
        "or legal value relevant to the case judgment?"
    )
    model = _get_cross_encoder()
    if model is None:
        return True  # Heuristic fallback: trust the amount
    scores = model.predict([(query, context)])
    return float(scores[0]) > -5.0


def _classify_amount(full_text: str, start: int, end: int) -> Tuple[str, bool]:
    raw_back = full_text[max(0, start - _CONTEXT_BACK) : start]
    is_suspect = _is_forbidden_context(raw_back)
    raw_fwd = full_text[end : end + _CONTEXT_FWD]

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

    if not found_kw and is_suspect:
        return "non_legal_mention", False

    return best_cat, found_kw


def extract_financials(text: str) -> Dict:
    """
    Extract monetary amounts from legal judgment text and categorise them.

    Returns dict with keys: fine, compensation, penalty, costs, raw_mentions.
    Each category entry: {"amount": str, "context": str}.
    raw_mentions includes a "category" field for debugging.
    """
    result: Dict[str, List] = {
        "fine": [], "compensation": [], "penalty": [], "costs": [], "raw_mentions": []
    }

    # Deduplicate by (amount_str, char_position) — same figure at two different
    # positions is kept; exact duplicate matches are dropped.
    seen: set = set()

    for match in _AMOUNT_PATTERN.finditer(text):
        amount_str = match.group(1).strip()

        # Gate 1: minimum numeric value filter
        numeric_val = _parse_numeric(amount_str)
        if numeric_val < _MIN_AMOUNT:
            result["raw_mentions"].append({
                "amount": amount_str,
                "category": "below_threshold",
                "context": text[max(0, match.start() - 40): match.end() + 40].strip(),
            })
            continue

        key = (amount_str, match.start())
        if key in seen:
            continue
        seen.add(key)

        category, has_keyword = _classify_amount(text, match.start(), match.end())

        # Build a clean context snippet using the sentence-boundary trimmer
        raw_back = text[max(0, match.start() - 100) : match.start()]
        raw_fwd = text[match.end() : min(len(text), match.end() + 100)]
        clean_back = _trim_at_boundary(raw_back, "back")
        clean_fwd = _trim_at_boundary(raw_fwd, "fwd")
        snippet = (clean_back + match.group(0) + clean_fwd).strip()

        # Semantic rescue: only call the (expensive) cross-encoder for
        # amounts that lack a clear legal keyword in their context window.
        if not has_keyword and category != "non_legal_mention":
            verified = _is_semantic_match(category, snippet)
        else:
            verified = has_keyword

        detail = {"amount": amount_str, "context": f"...{snippet}..."}

        if verified and category != "non_legal_mention":
            result[category].append(detail)
        # Non-legal / unverified amounts stay in raw_mentions only — they do
        # NOT get pushed into "fine" as a catch-all category.

        result["raw_mentions"].append({
            "amount": amount_str,
            "category": (
                category if (verified or category == "non_legal_mention") else "unverified"
            ),
            "context": snippet,
        })

    return result
