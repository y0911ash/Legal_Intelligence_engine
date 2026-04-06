"""
Phase 6: Financial Intelligence Extractor
------------------------------------------
Scans judgment text for monetary amounts and classifies them into
fine / compensation / penalty / cost buckets.

Output is a structured dict ready for UI display.

Example output:
{
  "fine":         ["₹50,000", "Rs. 10,000"],
  "compensation": ["₹1,00,000"],
  "penalty":      [],
  "costs":        ["₹5,000"],
  "raw_mentions": [
    {"amount": "₹50,000", "category": "fine", "context": "...imposed a fine of ₹50,000..."}
  ]
}
"""

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Amount pattern: matches ₹ or Rs./Rs followed by Indian number format
# Handles: Rs. 50,000 / Rs 1,00,000 / ₹50,000 / ₹1,00,000
# ---------------------------------------------------------------------------
_AMOUNT_PATTERN = re.compile(
    r"(?:₹|Rs\.?\s*)(\d{1,2}(?:,\d{2})*,\d{3}|\d+)"
    r"(?:\s*/?\-?)?"
    r"(?:\s*(?:lakhs?|crores?|thousands?))?",
    re.IGNORECASE
)

# Category context patterns — look at words BEFORE the amount
_CATEGORY_CONTEXT = {
    "fine": re.compile(
        r"(?i)(fine|fined|imposed\s+a\s+fine|penalty\s+fine|"
        r"paid\s+a\s+fine|default\s+fine)\b"
    ),
    "compensation": re.compile(
        r"(?i)(compensation|compensate|damages|solatium|"
        r"ex\s*-?\s*gratia|awarded\s+as\s+compensation|"
        r"awarded\s+(compensation|damages))\b"
    ),
    "penalty": re.compile(
        r"(?i)(penalt(y|ies)|penalised|penalized|surcharge|"
        r"interest\s+penalty|as\s+penalty)\b"
    ),
    "costs": re.compile(
        r"(?i)(costs?|court\s+fee|litigation\s+cost|"
        r"cost\s+of\s+the\s+petition|cost\s+of\s+the\s+appeal|"
        r"awarded\s+as\s+costs?)\b"
    ),
}

CONTEXT_WINDOW_BACK  = 120    # chars to look back from amount start
CONTEXT_WINDOW_FWRD  = 60     # chars to look forward from amount end

_SENTENCE_END = re.compile(r'[.!?]')


def _trim_at_boundary(text: str, direction: str) -> str:
    """
    Trim context at the nearest sentence boundary so we don't accidentally
    pick up keywords from adjacent sentences.
      direction='back'    → trim at the LAST sentence boundary (keep tail)
      direction='forward' → trim at the FIRST sentence boundary (keep head)
    """
    if direction == 'back':
        m = list(_SENTENCE_END.finditer(text))
        if m:
            return text[m[-1].end():]   # text after the last full stop
        return text
    else:  # forward
        m = _SENTENCE_END.search(text)
        if m:
            return text[:m.start()]     # text before the first full stop
        return text


# ---------------------------------------------------------------------------
# "Legal Guardrail" Heuristic Blacklist
# ---------------------------------------------------------------------------
_FORBIDDEN_PREFIXES = {
    "section", "article", "case", "petition", "no.", "no", "writ", "act",
    "dated", "year", "dated:", "paragraph", "para", "clause"
}

def _is_forbidden_context(text: str) -> bool:
    """Checks if the context around an amount indicates it's a non-fine (like a section)."""
    words = text.lower().split()
    # Check the last 3 words before the amount for forbidden prefixes
    for word in words[-3:]:
        clean_word = re.sub(r'[^a-z0-9.]', '', word)
        if clean_word in _FORBIDDEN_PREFIXES:
            return True
    return False


# ---------------------------------------------------------------------------
# Semantic BERT scoring for 'True Fine' detection
# ---------------------------------------------------------------------------
_CROSS_ENCODER = None

def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER

def _is_forbidden_context(text: str) -> bool:
    """Checks if context is a non-fine (like a section), but allows payment keyword overrides."""
    lower_text = text.lower()
    words = lower_text.split()
    
    # Check for 'Fine' or 'Penalty' in the same context - these OVERRIDE forbidden prefixes
    if any(keyword in lower_text for keyword in ["fine", "penalty", "compensation", "costs", "sum of"]):
        return False

    # Standard bouncer for case IDs and sections
    for word in words[-4:]:
        clean_word = re.sub(r'[^a-z0-9.]', '', word)
        if clean_word in _FORBIDDEN_PREFIXES:
            return True
    return False

# ...

def _is_semantic_match(category: str, context: str) -> bool:
    """Uses BERT to verify if context implies a legal penalty payment."""
    query = f"The court ordered a {category} or financial payment of an amount."
    model = _get_cross_encoder()
    score = model.predict([query, context])
    # Lowered threshold (0.1) for better recall on high-value fines
    return score > 0.1


def _classify_amount(full_text: str, match_start: int, match_end: int) -> str:
    """
    Return category using keywords and a heuristic bouncer.
    """
    raw_back = full_text[max(0, match_start - CONTEXT_WINDOW_BACK):match_start]
    
    # 🚨 GUARDRAIL 1: HEURISTIC BOUNCER
    if _is_forbidden_context(raw_back):
        return "non_legal_mention"

    raw_fwd  = full_text[match_end:match_end + CONTEXT_WINDOW_FWRD]
    back_context = _trim_at_boundary(raw_back, 'back')
    fwd_context  = _trim_at_boundary(raw_fwd,  'forward')

    best_category = "fine"
    best_distance = float("inf")

    for category, pattern in _CATEGORY_CONTEXT.items():
        for m in pattern.finditer(back_context):
            dist = len(back_context) - m.end()
            if dist < best_distance:
                best_distance = dist
                best_category = category

        for m in pattern.finditer(fwd_context):
            dist = m.start()
            if dist < best_distance:
                best_distance = dist
                best_category = category

    return best_category


def _format_amount(match_obj) -> str:
    """Return clean amount string with ₹ prefix using captured group."""
    number = match_obj.group(1)   # the digit portion only
    return f"₹{number}"


def extract_financials(text: str) -> Dict:
    result = {
        "fine": [],
        "compensation": [],
        "penalty": [],
        "costs": [],
        "raw_mentions": []
    }

    seen_amounts = set()      # avoid duplicates

    for match in _AMOUNT_PATTERN.finditer(text):
        amount_str = _format_amount(match)
        if amount_str in seen_amounts: continue
        seen_amounts.add(amount_str)

        category = _classify_amount(text, match.start(), match.end())

        # Capture ~80 chars of context for scoring and UI
        ctx_start = max(0, match.start() - 60)
        ctx_end = min(len(text), match.end() + 60)
        context_snippet = text[ctx_start:ctx_end].strip()

        # 🚨 GUARDRAIL 2: SEMANTIC VERIFIER (BERT)
        if category != "non_legal_mention":
            if not _is_semantic_match(category, context_snippet):
                category = "non_legal_mention"

        detail = {
            "amount": amount_str,
            "context": f"...{context_snippet}..."
        }

        # Only add to verified buckets if it passed the bouncers
        if category in ["fine", "compensation", "penalty", "costs"]:
            result[category].append(detail)

        result["raw_mentions"].append({
            "amount": amount_str,
            "category": category,
            "context": context_snippet
        })

    return result
