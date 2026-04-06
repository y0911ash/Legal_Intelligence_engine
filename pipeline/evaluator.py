"""
Evaluation: ROUGE Scorer
-------------------------
Computes ROUGE-1, ROUGE-2, and ROUGE-L between generated summaries
and reference summaries (from ILDC gold labels or human-written).

Usage:
  from pipeline.evaluator import evaluate_summary, batch_evaluate

  score = evaluate_summary(
      generated="The court dismissed the appeal...",
      reference="Appeal dismissed. Conviction upheld..."
  )
  print(score)
  # {'rouge1': 0.42, 'rouge2': 0.18, 'rougeL': 0.38}

For your viva:
  - Show ROUGE-L >= 0.25 as baseline (pretrained model, no fine-tuning)
  - Compare: with ranker vs. without ranker (just first 512 tokens)
  - This comparison IS your core experiment result
"""

from rouge_score import rouge_scorer
from typing import Dict, List


_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)


def evaluate_summary(generated: str, reference: str) -> Dict[str, float]:
    """
    Returns dict of ROUGE scores (F1 values only — cleaner for reporting).
    """
    if not generated.strip() or not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scores = _SCORER.score(reference, generated)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def batch_evaluate(
    generated_list: List[str],
    reference_list: List[str]
) -> Dict[str, float]:
    """
    Average ROUGE across a list of (generated, reference) pairs.
    Use with your 10-15 ILDC blind test cases.
    """
    assert len(generated_list) == len(reference_list), \
        "Generated and reference lists must have the same length."

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(generated_list)

    for gen, ref in zip(generated_list, reference_list):
        scores = evaluate_summary(gen, ref)
        for k in totals:
            totals[k] += scores[k]

    return {k: round(v / n, 4) for k, v in totals.items()}


def baseline_comparison(
    full_text: str,
    reference: str,
    ranked_summary: str,
    truncate_tokens: int = 512
) -> Dict:
    """
    Fair comparison:
      A) Naive baseline: first N words -> summarize_raw() (same model/params)
      B) Ranked system: segmented -> ranked -> summarize() (same model/params)

    Both use the EXACT same model, tokenizer, beam search, and decoding.
    Only difference: input text selection strategy.
    """
    from pipeline.summarizer import summarize_raw

    # Naive baseline: take the first N words (simulating no ranking)
    words = full_text.split()
    naive_input = " ".join(words[:truncate_tokens])

    # Use summarize_raw() — same model, same prefix, same generate() params
    naive_summary = summarize_raw(naive_input)

    naive_scores = evaluate_summary(naive_summary, reference)
    ranked_scores = evaluate_summary(ranked_summary, reference)

    delta = {
        k: round(ranked_scores[k] - naive_scores[k], 4)
        for k in naive_scores
    }

    return {
        "naive_truncation": naive_scores,
        "retrieval_ranked": ranked_scores,
        "improvement_delta": delta,
        "naive_summary": naive_summary,
        "ranked_summary": ranked_summary,
    }

