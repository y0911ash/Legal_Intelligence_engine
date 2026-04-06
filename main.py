"""
Legal Intelligence Engine — Main Pipeline
==========================================
Orchestrates all phases in sequence.

Usage:
  from main import run_pipeline

  with open("judgment.txt") as f:
      text = f.read()

  result = run_pipeline(text)
  print(result["summary"])
  print(result["financials"])
  print(result["statute_changes"])
"""

from pipeline.segmenter import segment_legal_doc
from pipeline.ranker import rank_chunks
from pipeline.summarizer import summarize
from pipeline.bns_mapper import map_statutes
from pipeline.financial_extractor import extract_financials


def run_pipeline(raw_text: str) -> dict:
    """
    Full pipeline: raw judgment text → structured intelligence output.

    Returns:
    {
      "summary":          str,       # clean abstractive summary
      "mapped_summary":   str,       # summary with BNS annotations
      "statute_changes":  list,      # IPC → BNS mappings detected
      "financials":       dict,      # structured financial data
      "segmentation_mode": str,      # "structured" or "fallback"
      "top_chunks":       list,      # (text, score, section) tuples
    }
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
    # Also map the original text so the UI can highlight changes
    _, full_text_changes = map_statutes(raw_text)
    all_changes = {
        d["ipc_section"]: d for d in (statute_changes + full_text_changes)
    }.values()

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


# ---------------------------------------------------------------------------
# Quick smoke test — run with: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE = """
    IN THE SUPREME COURT OF INDIA
    Civil Appeal No. 1234 of 2024

    FACTS OF THE CASE:
    The appellant was charged under Section 302 IPC for the alleged murder
    of the deceased on 15th March 2023. The Sessions Court convicted the
    appellant and sentenced him to life imprisonment along with a fine of
    Rs. 50,000.

    ARGUMENTS:
    Learned counsel for the appellant submitted that the prosecution had
    failed to prove beyond reasonable doubt that the appellant committed the
    offence. The learned counsel for the respondent contended that the
    evidence on record was sufficient to sustain the conviction.

    JUDGMENT:
    We have heard learned counsel for both sides and perused the records.
    The prosecution has established the guilt of the accused beyond reasonable
    doubt. The circumstantial evidence clearly points to the appellant.

    ORDERED:
    In view of the foregoing, the appeal is dismissed. The conviction under
    Section 302 IPC is upheld. The accused shall pay compensation of
    Rs. 1,00,000 to the family of the deceased.
    """

    result = run_pipeline(SAMPLE)

    print("=" * 60)
    print("SUMMARY:")
    print(result["summary"])
    print("\nMAPPED SUMMARY (IPC -> BNS):")
    print(result["mapped_summary"])
    print("\nSTATUTE CHANGES:")
    for s in result["statute_changes"]:
        print(f"  {s['ipc_section']} -> {s['bns_section']} ({s['description']})")
    print("\nFINANCIALS:")
    f = result["financials"]
    print(f"  Fines: {f['fine']}")
    print(f"  Compensation: {f['compensation']}")
    print(f"\nSegmentation mode: {result['segmentation_mode']}")
