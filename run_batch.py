"""
Batch Evaluation on ILDC (Indian Legal Documents Corpus)
Loads real case data from HuggingFace, processes through our pipeline,
and saves generated summaries to CSV.
"""
import os, sys, time, warnings
import pandas as pd
from datasets import load_dataset

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

from pipeline.segmenter import segment_legal_doc
from pipeline.ranker import rank_chunks
from pipeline.summarizer import summarize


def process_batch(num_cases=5):
    print("=" * 64)
    print("  BATCH PROCESSING: ILDC DATASET (Real Indian Cases)")
    print("=" * 64)

    print("[1] Downloading ILDC dataset from HuggingFace...")
    try:
        dataset = load_dataset('anuragiiser/ILDC_expert', split='train')
        print(f"    Loaded {len(dataset)} real cases from ILDC.")
    except Exception as e:
        print(f"Error loading ILDC: {e}")
        return

    cases = dataset.select(range(min(num_cases, len(dataset))))
    results = []
    print("\n[2] Connecting to Pipeline (flan-t5-large on GPU)...")
    start_time = time.time()

    for i, row in enumerate(cases):
        case_id = row.get("Case ID", f"Unknown-{i}")
        text = row.get("Case Description", "")
        words = len(text.split())
        print(f"\nProcessing [{i+1}/{num_cases}]: Case {case_id} ({words} words)")

        try:
            doc = segment_legal_doc(text)
            ranked = rank_chunks(doc)
            t0 = time.time()
            summary = summarize(ranked)
            gen_time = time.time() - t0
            print(f"  -> Generated {len(summary.split())}-word summary in {gen_time:.1f}s")
            results.append({
                "case_id": case_id,
                "input_length_words": words,
                "generation_time_sec": round(gen_time, 2),
                "generated_summary": summary,
            })
        except Exception as e:
            print(f"  -> Error processing case {case_id}: {e}")

    total_time = time.time() - start_time
    print(f"\n[3] Completed processing {len(results)} cases in {total_time:.1f}s")

    output_file = "ildc_batch_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"[✓] Results saved to {output_file}")
    print("\nExample Summary Generated:")
    print(df.iloc[0]["generated_summary"])


if __name__ == "__main__":
    process_batch(num_cases=5)
