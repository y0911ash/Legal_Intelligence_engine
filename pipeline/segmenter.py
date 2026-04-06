"""
Phase 1: Legal Document Segmenter
----------------------------------
Parses Indian court judgments into structured sections.
Falls back to treating the entire text as 'judgment' if no headers are detected.
"""

import re
from dataclasses import dataclass, field
from typing import Dict

# Section header patterns — ordered from weakest to strongest signal
SECTION_PATTERNS = {
    "facts": [
        r"(?i)\b(facts\s+of\s+the\s+(case|matter)|brief\s+facts|background(\s+facts)?|"
        r"factual\s+(background|matrix)|brief\s+background|stated\s+facts)\b",
        r"(?i)^\s*facts\s*:?\s*$",
    ],
    "arguments": [
        r"(?i)\b(contentions?|submissions?|arguments?\s+(made|raised|advanced)|"
        r"learned\s+(counsel|advocate)(\s+for\s+\w+)?\s+(submitted?|argued?|contended?)|"
        r"on\s+behalf\s+of\s+(the\s+)?(appellant|respondent|petitioner))\b",
        r"(?i)^\s*(arguments?|submissions?|contentions?)\s*:?\s*$",
    ],
    "judgment": [
        r"(?i)\b(reasoning|analysis|point[s]?\s+for\s+determination|"
        r"consideration|discussion|our\s+view|we\s+find|we\s+are\s+of\s+the\s+opinion|"
        r"upon\s+perusal|having\s+heard|after\s+hearing)\b",
        r"(?i)^\s*(judgment|analysis|reasoning|discussion|consideration)\s*:?\s*$",
    ],
    "final_order": [
        r"(?i)\b(in\s+the\s+result|for\s+(the\s+)?foregoing\s+reasons|"
        r"in\s+view\s+of\s+the\s+above|accordingly|order[s]?:|"
        r"it\s+is\s+(hereby\s+)?(ordered|directed|decreed)|"
        r"(the\s+)?appeal\s+is\s+(allowed|dismissed|disposed|partly\s+allowed)|"
        r"(the\s+)?petition\s+is\s+(allowed|dismissed|disposed))\b",
        r"(?i)^\s*(order|ordered|conclusion|result|dispositif|operative\s+order)\s*:?\s*$",
    ],
}


@dataclass
class SegmentedDocument:
    sections: Dict[str, str] = field(default_factory=dict)
    segmented: bool = True
    detected_headers: list = field(default_factory=list)

    def summary(self) -> str:
        mode = "structured" if self.segmented else "FALLBACK (no headers detected)"
        lengths = {k: len(v.split()) for k, v in self.sections.items()}
        return f"Segmentation mode: {mode} | Section word counts: {lengths}"


# Pre-compile all patterns at module load
_COMPILED = {
    section: [re.compile(p, re.MULTILINE) for p in patterns]
    for section, patterns in SECTION_PATTERNS.items()
}


def _match_section(line: str) -> str | None:
    """Return section name if any pattern matches, else None."""
    for section, pattern_list in _COMPILED.items():
        if any(p.search(line) for p in pattern_list):
            return section
    return None


def _split_into_units(text: str) -> list[str]:
    """Split text into processable units (lines or sentences for single-line PDFs)."""
    lines = text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) < 3:
        return re.split(r'(?<=[.!?])\s+', text)
    return lines


def segment_legal_doc(text: str) -> SegmentedDocument:
    """
    Main entry point. Accepts raw judgment text, returns a SegmentedDocument.
    If fewer than 2 distinct sections detected, falls back to dumping all text
    into the 'judgment' bucket.
    """
    units = _split_into_units(text)
    sections: Dict[str, list] = {k: [] for k in SECTION_PATTERNS}
    current_section = "facts"
    detected_headers = []

    for unit in units:
        stripped = unit.strip()
        if not stripped:
            sections[current_section].append(unit)
            continue

        matched = _match_section(stripped)
        if matched and matched != current_section:
            current_section = matched
            if matched not in detected_headers:
                detected_headers.append(matched)

        sections[current_section].append(unit)

    result = {k: " ".join(v).strip() for k, v in sections.items()}

    # Fallback: if only one section has content, treat entire text as judgment
    non_empty = [k for k, v in result.items() if v.split()]
    if len(non_empty) <= 1:
        return SegmentedDocument(
            sections={k: (text.strip() if k == "judgment" else "") for k in SECTION_PATTERNS},
            segmented=False,
            detected_headers=[],
        )

    return SegmentedDocument(sections=result, segmented=True, detected_headers=detected_headers)
