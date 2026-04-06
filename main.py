"""
Legal Intelligence Engine — Main Pipeline
==========================================
Orchestrates all phases: segmentation → ranking → summarization → BNS mapping → financials.

Usage:
  from main import run_pipeline
  result = run_pipeline(text)
"""

from pipeline.segmenter import segment_legal_doc
from pipeline.ranker import rank_chunks
from pipeline.summarizer import summarize
from pipeline.bns_mapper import map_statutes
from pipeline.financial_extractor import extract_financials


def run_pipeline(raw_text: str) -> dict:
    """
    Full pipeline: raw judgment text → structured intelligence output.

    Returns dict with keys: summary, mapped_summary, statute_changes,
    financials, segmentation_mode, top_chunks.
    """
    print("\n[Pipeline] Step 1/5 -- Segmenting document...")
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
