"""
Batch Evaluation on ILDC (Indian Legal Documents Corpus)
Loads real case data from HuggingFace (anuragiiser/ILDC_expert),
processes them through our Chunk-Ranker Pipeline, and saves the 
generated summaries to a CSV file.
"""
import os
import sys
import time
import pandas as pd
from datasets import load_dataset
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Import our pipeline
from pipeline.segmenter import segment_legal_doc
from pipeline.ranker import rank_chunks
from pipeline.summarizer import summarize

def process_batch(num_cases=5):
    print("=" * 64)
    print("  BATCH PROCESSING: ILDC DATASET (Real Indian Cases)")
    print("=" * 64)
    
    # Load authentic ILDC subset from huggingface
    print("[1] Downloading ILDC dataset from HuggingFace...")
    try:
        dataset = load_dataset('anuragiiser/ILDC_expert', split='train')
        print(f"    Loaded {len(dataset)} real cases from ILDC.")
    except Exception as e:
        print(f"Error loading ILDC: {e}")
        return

    # Prepare batch
    cases_to_process = dataset.select(range(min(num_cases, len(dataset))))
    results = []

    print("\n[2] Connecting to Pipeline (flan-t5-large on GPU)...")
    
    start_time = time.time()
    
    for i, row in enumerate(cases_to_process):
        case_id = row.get("Case ID", f"Unknown-{i}")
        text = row.get("Case Description", "")
        
        # Word count of input
        words = len(text.split())
        print(f"\nProcessing [{i+1}/{num_cases}]: Case {case_id} ({words} words)")
        
        try:
            # 1. Segment
            doc = segment_legal_doc(text)
            
            # 2. Rank
            ranked_chunks = rank_chunks(doc)
            
            # 3. Summarize
            t_sum_start = time.time()
            summary = summarize(ranked_chunks)
            t_sum_end = time.time()
            
            gen_time = t_sum_end - t_sum_start
            
            print(f"  -> Generated {len(summary.split())}-word summary in {gen_time:.1f}s")
            
            results.append({
                "case_id": case_id,
                "input_length_words": words,
                "generation_time_sec": round(gen_time, 2),
                "generated_summary": summary
            })
            
        except Exception as e:
            print(f"  -> Error processing case {case_id}: {e}")
            
    total_time = time.time() - start_time
    print(f"\n[3] Completed processing {len(results)} cases in {total_time:.1f}s")
    
    # Save to CSV
    output_file = "ildc_batch_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"[✓] Results properly saved to {output_file}")
    
    print("\nExample Summary Generated:")
    print(df.iloc[0]["generated_summary"])


if __name__ == "__main__":
    # You can process the entire dataset by passing len(dataset), but we'll do 5 for a quick test
    process_batch(num_cases=5)
