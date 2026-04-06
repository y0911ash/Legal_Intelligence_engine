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


def _classify_amount(full_text: str, match_start: int, match_end: int) -> str:
    """
    Return category using the closest keyword match in either direction,
    confined to the current sentence (stops at sentence boundaries).
    """
    raw_back = full_text[max(0, match_start - CONTEXT_WINDOW_BACK):match_start]
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
        if amount_str in seen_amounts:
            continue
        seen_amounts.add(amount_str)

        category = _classify_amount(text, match.start(), match.end())
        
        # Capture ~80 chars of context around the match for UI display
        ctx_start = max(0, match.start() - 50)
        ctx_end = min(len(text), match.end() + 50)
        context_snippet = text[ctx_start:ctx_end].strip()
        
        detail = {
            "amount": amount_str,
            "context": f"...{context_snippet}..."
        }
        
        result[category].append(detail)

        result["raw_mentions"].append({
            "amount": amount_str,
            "category": category,
            "context": context_snippet
        })

    return result
