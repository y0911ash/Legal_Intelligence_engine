"""
Phase 1: Legal Document Segmenter
----------------------------------
Parses Indian court judgments into structured sections.

KEY FIX over naive version:
- Fallback mode: if no headers are detected (common in older judgments),
  the entire text is treated as a single 'judgment' block so the pipeline
  never fails silently.
- Confidence flag tells downstream modules how reliable the segmentation is.
"""

import re
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Section header patterns — ordered from weakest to strongest signal
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "facts": [
        r"(?i)\b(facts\s+of\s+the\s+(case|matter)|brief\s+facts|background(\s+facts)?|"
        r"factual\s+(background|matrix)|brief\s+background|stated\s+facts)\b",
        # Standalone headers: "FACTS:", "FACTS OF CASE:"
        r"(?i)^\s*facts\s*:?\s*$",
    ],
    "arguments": [
        r"(?i)\b(contentions?|submissions?|arguments?\s+(made|raised|advanced)|"
        r"learned\s+(counsel|advocate)(\s+for\s+\w+)?\s+(submitted?|argued?|contended?)|"
        r"on\s+behalf\s+of\s+(the\s+)?(appellant|respondent|petitioner))\b",
        # Standalone headers: "ARGUMENTS:", "SUBMISSIONS:", "CONTENTIONS:"
        r"(?i)^\s*(arguments?|submissions?|contentions?)\s*:?\s*$",
    ],
    "judgment": [
        r"(?i)\b(reasoning|analysis|point[s]?\s+for\s+determination|"
        r"consideration|discussion|our\s+view|we\s+find|we\s+are\s+of\s+the\s+opinion|"
        r"upon\s+perusal|having\s+heard|after\s+hearing)\b",
        # Standalone headers: "JUDGMENT:", "ANALYSIS:", "REASONING:"
        r"(?i)^\s*(judgment|analysis|reasoning|discussion|consideration)\s*:?\s*$",
    ],
    "final_order": [
        r"(?i)\b(in\s+the\s+result|for\s+(the\s+)?foregoing\s+reasons|"
        r"in\s+view\s+of\s+the\s+above|accordingly|order[s]?:|"
        r"it\s+is\s+(hereby\s+)?(ordered|directed|decreed)|"
        r"(the\s+)?appeal\s+is\s+(allowed|dismissed|disposed|partly\s+allowed)|"
        r"(the\s+)?petition\s+is\s+(allowed|dismissed|disposed))\b",
        # Standalone headers: "ORDER:", "ORDERED:", "CONCLUSION:", "RESULT:"
        r"(?i)^\s*(order|ordered|conclusion|result|dispositif|operative\s+order)\s*:?\s*$",
    ],
}


@dataclass
class SegmentedDocument:
    sections: Dict[str, str] = field(default_factory=dict)
    segmented: bool = True          # False = fallback mode was used
    detected_headers: list = field(default_factory=list)

    def summary(self) -> str:
        mode = "structured" if self.segmented else "FALLBACK (no headers detected)"
        lengths = {k: len(v.split()) for k, v in self.sections.items()}
        return f"Segmentation mode: {mode} | Section word counts: {lengths}"


def _compile_patterns():
    compiled = {}
    for section, patterns in SECTION_PATTERNS.items():
        # Keep as list — joining (?i) inline flags causes re.error when mixed with flags arg
        compiled[section] = [
            re.compile(p, re.MULTILINE) for p in patterns
        ]
    return compiled


def _match_section(line: str, compiled_map: dict):
    """Return section name if any pattern matches, else None."""
    for section, pattern_list in compiled_map.items():
        for pattern in pattern_list:
            if pattern.search(line):
                return section
    return None


_COMPILED = _compile_patterns()


def _split_into_units(text: str):
    """
    Split text into processable units.
    Strategy: try newline-based split first; if most lines are long
    (single-line PDFs, concatenated text), fall back to sentence splitting.
    """
    lines = text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    # If we get fewer than 3 non-empty lines, try sentence splitting
    if len(non_empty_lines) < 3:
        # Split on sentence boundaries that look like section transitions
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    return lines


def segment_legal_doc(text: str) -> SegmentedDocument:
    """
    Main entry point. Accepts raw judgment text, returns a SegmentedDocument.

    Strategy:
      1. Split text into lines (or sentences if line-split gives <3 units).
      2. Scan each unit for section header signals.
      3. If fewer than 2 distinct sections detected → activate fallback.
      4. Fallback: dump entire text into 'judgment' bucket so pipeline continues.
    """
    units = _split_into_units(text)
    sections: Dict[str, list] = {"facts": [], "arguments": [], "judgment": [], "final_order": []}
    current_section = "facts"
    detected_headers = []

    for unit in units:
        stripped = unit.strip()
        if not stripped:
            sections[current_section].append(unit)
            continue

        matched_section = _match_section(stripped, _COMPILED)

        if matched_section and matched_section != current_section:
            current_section = matched_section
            if matched_section not in detected_headers:
                detected_headers.append(matched_section)

        sections[current_section].append(unit)

    # Collapse lists to strings
    result = {k: " ".join(v).strip() for k, v in sections.items()}

    # ----- FALLBACK CHECK -----
    # If only one section has content (everything landed in 'facts' because
    # no headers were found), activate fallback.
    non_empty = [k for k, v in result.items() if v.split()]
    if len(non_empty) <= 1:
        fallback_sections = {
            "facts": "",
            "arguments": "",
            "judgment": text.strip(),   # entire text goes here
            "final_order": ""
        }
        return SegmentedDocument(
            sections=fallback_sections,
            segmented=False,
            detected_headers=[]
        )

    return SegmentedDocument(
        sections=result,
        segmented=True,
        detected_headers=detected_headers
    )
