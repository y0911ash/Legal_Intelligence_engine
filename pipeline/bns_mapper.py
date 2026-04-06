"""
Phase 5: IPC → BNS Statute Mapper
-----------------------------------
Detects references to old IPC sections in any text and appends
the corresponding BNS (Bharatiya Nyaya Sanhita 2023) section.

Example:
  Input:  "...convicted under Section 302 IPC for murder..."
  Output: "...convicted under Section 302 IPC [→ BNS Section 101: Murder]..."

Design notes:
  - Regex handles common citation styles: "S. 302 IPC", "Sec 302 IPC",
    "Section 302, IPC", "u/s 302 IPC", "under Section 302 of the IPC"
  - Only appends mapping if NOT already mapped (idempotent)
  - Returns both the mapped text AND a list of detected statutes (for UI)
"""

import re
import json
from pathlib import Path
from typing import Tuple, List, Dict

_MAPPING_PATH = Path(__file__).parent.parent / "data" / "ipc_to_bns.json"
_MAPPING: Dict[str, dict] = {}


def _load_mapping():
    global _MAPPING
    if not _MAPPING:
        with open(_MAPPING_PATH, "r", encoding="utf-8") as f:
            _MAPPING = json.load(f)


# Matches all common Indian citation styles:
#   u/s 302 IPC  |  section 302 IPC  |  S. 302 IPC  |  under section 302 of the IPC
_IPC_PATTERN = re.compile(
    r"(?i)"
    r"(?:u[/\\]s\s*|(?:under\s+)?(?:section|sec\.?|s\.)\s*)"
    r"(\d{1,3}[A-Z]?)"
    r"(?:\s*,\s*|\s+)(?:of\s+the\s+)?IPC\b"
)

# Already-mapped marker — don't double-map
_ALREADY_MAPPED = re.compile(r"\[→\s*BNS Section")


def map_statutes(text: str) -> Tuple[str, List[Dict]]:
    """
    Returns:
      - mapped_text: original text with BNS annotations inserted
      - detected:    list of { ipc_section, bns_section, description }
    """
    _load_mapping()

    detected: List[Dict] = []
    offset = 0
    result = text

    for match in _IPC_PATTERN.finditer(text):
        # Check if already annotated right after this match
        end_pos = match.end() + offset
        after = result[end_pos:end_pos + 30]
        if _ALREADY_MAPPED.search(after):
            continue

        section_num = match.group(1).upper()

        if section_num in _MAPPING:
            entry = _MAPPING[section_num]
            annotation = f" [→ BNS Section {entry['bns']}: {entry['description']}]"
            insert_pos = match.end() + offset
            result = result[:insert_pos] + annotation + result[insert_pos:]
            offset += len(annotation)

            detected.append({
                "ipc_section": f"Section {section_num} IPC",
                "bns_section": f"Section {entry['bns']} BNS",
                "description": entry["description"]
            })

    return result, detected
