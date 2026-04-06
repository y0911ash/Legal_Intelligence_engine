"""
Phase 6: Financial Intelligence Extractor (EXPERT EDITION)
-----------------------------------------------------------
Scans judgment text for monetary amounts and verifies them using:
1.  Surgical Bouncer (Negative Lookbehind for Sections/Case IDs)
2.  Semantic BERT Verifier (Cross-Encoder for Deep Context Analysis)
"""

import re
import torch
from typing import Dict, List

# ---------------------------------------------------------------------------
# Patterns & Windows
# ---------------------------------------------------------------------------
_AMOUNT_PATTERN = re.compile(
    r"(?:₹|Rs\.?|INR|Rupees)\s*(\d{1,2}(?:,\d{2})*,\d{3}|\d+)"
    r"(?:\s*/?\-?)?"
    r"(?:\s*(?:lakhs?|crores?|thousands?))?",
    re.IGNORECASE
)

_CATEGORY_CONTEXT = {
    "fine": re.compile(r"(?i)(fine|fined|imposed|penalty|default|amounting)\b"),
    "compensation": re.compile(r"(?i)(compensation|compensate|damages|solatium|ex\s*-?\s*gratia)\b"),
    "penalty": re.compile(r"(?i)(penalt(y|ies)|penalised|penalized|surcharge)\b"),
    "costs": re.compile(r"(?i)(costs?|court\s+fee|litigation\s+cost)\b"),
}

_FORBIDDEN_PREFIXES = {"section", "article", "case", "petition", "no.", "no", "writ", "act", "dated", "year"}

CONTEXT_WINDOW_BACK  = 120
CONTEXT_WINDOW_FWRD  = 60
_SENTENCE_END = re.compile(r'[.!?]')

# ---------------------------------------------------------------------------
# Stage 1: Surgical Bouncer & Cleanup
# ---------------------------------------------------------------------------
def _trim_at_boundary(text: str, direction: str) -> str:
    if direction == 'back':
        m = list(_SENTENCE_END.finditer(text))
        return text[m[-1].end():] if m else text
    else:
        m = _SENTENCE_END.search(text)
        return text[:m.start()] if m else text

def _is_forbidden_context(text: str) -> bool:
    """Only triggers if forbidden word is DIRECTLY before amount (last 15 chars)."""
    immediate_prefix = text[-15:].lower()
    return any(forbidden in immediate_prefix for forbidden in _FORBIDDEN_PREFIXES)

# ---------------------------------------------------------------------------
# Stage 2: BERT Cross-Encoder (The Final Judge)
# ---------------------------------------------------------------------------
_CROSS_ENCODER = None

def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER

def _is_semantic_match(category: str, context: str) -> bool:
    """Uses BERT with ultra-sensitivity to capture all significant legal values."""
    query = f"Is this a significant monetary amount, {category}, or legal value relevant to the case judgment?"
    model = _get_cross_encoder()
    score = model.predict([query, context])
    # Ultra-low threshold for high recall on case-central amounts
    return score > 0.01

# ---------------------------------------------------------------------------
# Classification & Extraction
# ---------------------------------------------------------------------------
def _classify_amount(full_text: str, match_start: int, match_end: int) -> str:
    raw_back = full_text[max(0, match_start - CONTEXT_WINDOW_BACK):match_start]
    
    # Check Bouncer - but we label it as 'suspect' for BERT to decide later
    is_suspect = _is_forbidden_context(raw_back)
    
    raw_fwd = full_text[match_end:match_end + CONTEXT_WINDOW_FWRD]
    back_context = _trim_at_boundary(raw_back, 'back')
    fwd_context  = _trim_at_boundary(raw_fwd,  'forward')

    best_category = "fine"
    best_distance = float("inf")
    found_keyword = False

    for category, pattern in _CATEGORY_CONTEXT.items():
        for m in pattern.finditer(back_context):
            found_keyword = True
            dist = len(back_context) - m.end()
            if dist < best_distance:
                best_distance, best_category = dist, category
        for m in pattern.finditer(fwd_context):
            found_keyword = True
            dist = m.start()
            if dist < best_distance:
                best_distance, best_category = dist, category

    # If no keyword found AND bouncer suspect, it's definitely non-legal
    if not found_keyword and is_suspect:
        return "non_legal_mention"
    
    return best_category

def extract_financials(text: str) -> Dict:
    result = {"fine":[], "compensation":[], "penalty":[], "costs":[], "raw_mentions":[]}
    seen_amounts = set()

    for match in _AMOUNT_PATTERN.finditer(text):
        amount_str = f"₹{match.group(1)}"
        if amount_str in seen_amounts: continue
        seen_amounts.add(amount_str)

        category = _classify_amount(text, match.start(), match.end())

        # Give AI a wide 200-char view for the final verdict
        ctx_start = max(0, match.start() - 100)
        ctx_end   = min(len(text), match.end() + 100)
        context_snippet = text[ctx_start:ctx_end].strip()

        # 🚨 THE SEMANTIC OVERRIDE
        # Even if it looks like a non_legal_mention, let BERT check if it's a hidden fine
        is_verified = _is_semantic_match(category if category != "non_legal_mention" else "fine", context_snippet)
        
        detail = {"amount": amount_str, "context": f"...{context_snippet}..."}

        if is_verified and category != "non_legal_mention":
            result[category].append(detail)
        elif is_verified:
            # BERT rescued it! Treat as a general fine
            result["fine"].append(detail)
            category = "fine (verified)"

        result["raw_mentions"].append({
            "amount": amount_str,
            "category": category,
            "context": context_snippet
        })

    return result
