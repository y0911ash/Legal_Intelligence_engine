"""
Legal Intelligence Engine — Main Pipeline
==========================================
Orchestrates all phases: segmentation → ranking → summarization → BNS mapping → financials.

Usage:
  from main import run_pipeline
  result = run_pipeline(text)
"""

import re
from pipeline.segmenter import segment_legal_doc
from pipeline.ranker import rank_chunks
from pipeline.summarizer import summarize
from pipeline.bns_mapper import map_statutes
from pipeline.financial_extractor import extract_financials


def _clean_pdf_text(text: str) -> str:
    """
    Pre-clean PDF-extracted text before analysis.
    Removes common artefacts introduced by legal database PDF extractors
    (Manupatra, SCC Online, IndiaKanoon) that cause financial false positives.
    """
    # Remove page-stamp lines: "06-03-2026 (Page 12 of 108)"
    text = re.sub(r"\d{2}-\d{2}-\d{4}\s*\(Page\s+\d+\s+of\s+\d+\)", " ", text)
    # Remove Manupatra / SCC watermark lines
    text = re.sub(
        r"(?i)(www\.manupatra\.com|manupatra information solutions"
        r"|scc online|indiakanoon\.org|manu/sc/|manu/hc/)",
        " ", text
    )
    # Remove isolated institution names on their own line
    text = re.sub(r"(?m)^\s*[A-Z][A-Za-z ]+University\s*$", " ", text)
    # Collapse 3+ consecutive newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are purely digits (page numbers, paragraph counters)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    return text.strip()


def run_pipeline(raw_text: str) -> dict:
    """
    Full pipeline: raw judgment text → structured intelligence output.

    Returns dict with keys: summary, mapped_summary, statute_changes,
    financials, segmentation_mode, top_chunks.
    """
    print("\n[Pipeline] Step 0/5 -- Cleaning PDF text artefacts...")
    raw_text = _clean_pdf_text(raw_text)

    print("[Pipeline] Step 1/5 -- Segmenting document...")

    doc = segment_legal_doc(raw_text)
    print(f"  {doc.summary()}")

    print("[Pipeline] Step 2/5 -- Ranking chunks...")
    top_chunks = rank_chunks(doc)
    print(f"  Top {len(top_chunks)} chunks selected.")

    print("[Pipeline] Step 3/5 -- Generating summary...")
    summary = summarize(top_chunks)

    print("[Pipeline] Step 4/5 -- Mapping IPC -> BNS statutes...")
    mapped_summary, statute_changes = map_statutes(summary)
    _, full_text_changes = map_statutes(raw_text)
    # Deduplicate by IPC section
    all_changes = {d["ipc_section"]: d for d in statute_changes + full_text_changes}.values()

    print("[Pipeline] Step 5/5 -- Extracting financial data...")
    financials = extract_financials(raw_text)

    print("[Pipeline] Done.\n")
    return {
        "summary": summary,
        "mapped_summary": mapped_summary,
        "statute_changes": list(all_changes),
        "financials": financials,
        "segmentation_mode": "structured" if doc.segmented else "fallback",
        "top_chunks": [(c, round(s, 4), sec) for c, s, sec in top_chunks],
    }


if __name__ == "__main__":
    from test_cases import SHORT_TEXT

    result = run_pipeline(SHORT_TEXT)
    print("=" * 60)
    print("SUMMARY:", result["summary"])
    print("\nMAPPED SUMMARY (IPC -> BNS):", result["mapped_summary"])
    print("\nSTATUTE CHANGES:")
    for s in result["statute_changes"]:
        print(f"  {s['ipc_section']} -> {s['bns_section']} ({s['description']})")
    print("\nFINANCIALS:")
    print(f"  Fines: {result['financials']['fine']}")
    print(f"  Compensation: {result['financials']['compensation']}")
    print(f"\nSegmentation mode: {result['segmentation_mode']}")
