"""
Scratch test for pipeline.financial_extractor.

Run from the project root:
    python scratch/test_financial_extractor.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.financial_extractor import extract_financials
from test_cases import SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT

# ── Inline micro-tests ──────────────────────────────────────────────────────
INLINE_CASES = [
    {
        "name": "Standard Fine (Sharma)",
        "text": "ORDERED: The respondent is convicted. A fine of Rs. 5,00,000 is imposed.",
        "expect_cat": "fine",
    },
    {
        "name": "Compensation with Lakhs suffix",
        "text": "The complainant is awarded compensation of Rs. 5 lakhs.",
        "expect_cat": "compensation",
    },
    {
        "name": "Penalty in Case No. context",
        "text": "In Case No. 123, the court ordered a penalty of Rs. 10,000/-.",
        "expect_cat": "penalty",
    },
    {
        "name": "Forbidden prefix — Section number (should NOT appear as fine)",
        "text": "The respondent was charged under Section 420 IPC.",
        "expect_cat": None,   # no financial hit
    },
    {
        "name": "INR prefix",
        "text": "A compensation of INR 2,00,000 was directed to the victim's family.",
        "expect_cat": "compensation",
    },
    {
        "name": "Costs awarded",
        "text": "Costs of Rs. 25,000 awarded to the respondent.",
        "expect_cat": "costs",
    },
    {
        "name": "Same amount, two contexts (dedup check)",
        "text": (
            "The complainant paid Rs. 20,00,000 as advance. "
            "The accused was fined Rs. 20,00,000 by the court."
        ),
        "expect_count": 2,   # both instances should be captured
    },
]

SEPARATOR = "-" * 60


def _check(result, case):
    cats = ["fine", "compensation", "penalty", "costs"]
    found_entries = [(c, e) for c in cats for e in result[c]]
    found = bool(found_entries)

    if "expect_count" in case:
        total = sum(len(result[c]) for c in cats)
        status = "✅ PASS" if total == case["expect_count"] else f"❌ FAIL (got {total}, expected {case['expect_count']})"
    elif case.get("expect_cat") is None:
        status = "✅ PASS" if not found else f"❌ FAIL (unexpected: {found_entries})"
    else:
        status = "✅ PASS" if any(c == case["expect_cat"] for c, _ in found_entries) else \
                 f"❌ FAIL (got {[c for c, _ in found_entries]}, expected {case['expect_cat']})"

    return status, found_entries


def print_result(result):
    for cat in ["fine", "compensation", "penalty", "costs"]:
        for entry in result[cat]:
            print(f"  [{cat.upper()}] {entry['amount']}")
            print(f"         {entry['context'][:120]}")
    if not any(result[c] for c in ["fine", "compensation", "penalty", "costs"]):
        print("  (no financial implication extracted)")


print("\n" + "=" * 60)
print("  FINANCIAL EXTRACTOR — INLINE MICRO-TESTS")
print("=" * 60)

for case in INLINE_CASES:
    print(f"\n{SEPARATOR}")
    print(f"TEST : {case['name']}")
    result = extract_financials(case["text"])
    status, found = _check(result, case)
    print(f"STATUS: {status}")
    print_result(result)

# ── Full-document tests using shared test_cases ─────────────────────────────
DOCUMENT_CASES = [
    ("SHORT_TEXT  (Supreme Court — 302 IPC)", SHORT_TEXT),
    ("MEDIUM_TEXT (High Court Delhi — 420 IPC)", MEDIUM_TEXT),
    ("LONG_TEXT   (Bombay HC — 302/392 IPC)", LONG_TEXT),
]

print("\n\n" + "=" * 60)
print("  FINANCIAL EXTRACTOR — FULL-DOCUMENT TESTS")
print("=" * 60)

for title, text in DOCUMENT_CASES:
    print(f"\n{SEPARATOR}")
    print(f"DOC: {title}")
    result = extract_financials(text)
    print_result(result)
    print(f"  raw_mentions total: {len(result['raw_mentions'])}")
