"""
Evaluation: ROUGE Scorer
-------------------------
Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores between generated
and reference summaries.
"""

from rouge_score import rouge_scorer
from typing import Dict, List

_METRICS = ["rouge1", "rouge2", "rougeL"]
_SCORER = rouge_scorer.RougeScorer(_METRICS, use_stemmer=True)


def evaluate_summary(generated: str, reference: str) -> Dict[str, float]:
    """Returns dict of ROUGE F1 scores."""
    if not generated.strip() or not reference.strip():
        return {m: 0.0 for m in _METRICS}
    scores = _SCORER.score(reference, generated)
    return {m: round(scores[m].fmeasure, 4) for m in _METRICS}


def batch_evaluate(generated_list: List[str], reference_list: List[str]) -> Dict[str, float]:
    """Average ROUGE across a list of (generated, reference) pairs."""
    assert len(generated_list) == len(reference_list), \
        "Generated and reference lists must have the same length."
    n = len(generated_list)
    totals = {m: 0.0 for m in _METRICS}
    for gen, ref in zip(generated_list, reference_list):
        for k, v in evaluate_summary(gen, ref).items():
            totals[k] += v
    return {k: round(v / n, 4) for k, v in totals.items()}


def baseline_comparison(full_text: str, reference: str, ranked_summary: str,
                        truncate_tokens: int = 512) -> Dict:
    """
    Fair comparison: naive truncation vs ranked pipeline.
    Both use the exact same model, tokenizer, and decoding params.
    """
    from pipeline.summarizer import summarize_raw
    naive_input = " ".join(full_text.split()[:truncate_tokens])
    naive_summary = summarize_raw(naive_input)
    naive_scores = evaluate_summary(naive_summary, reference)
    ranked_scores = evaluate_summary(ranked_summary, reference)
    return {
        "naive_truncation": naive_scores,
        "retrieval_ranked": ranked_scores,
        "improvement_delta": {k: round(ranked_scores[k] - naive_scores[k], 4) for k in _METRICS},
        "naive_summary": naive_summary,
        "ranked_summary": ranked_summary,
    }
