"""
Phase 4: Summarization
-----------------------
Takes top-ranked chunks and generates a focused multi-pass abstractive summary.

Model hierarchy (auto-selects best available):
  - CUDA GPU        -> facebook/bart-large-cnn
  - CPU + 'large'   -> google/flan-t5-large  (float16, ~1.5GB RAM)
  - CPU + 'base'    -> google/flan-t5-base   (default, ~1GB RAM)

Override via env: SUMMARIZER_MODEL=large|base|bart
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple

MAX_SUMMARY_TOKENS = 256

_MODELS = {
    "base":  "google/flan-t5-base",
    "large": "google/flan-t5-large",
    "bart":  "facebook/bart-large-cnn",
}

_TOKENIZER = None
_MODEL = None
_MODEL_NAME = None


def _select_model() -> str:
    """Select best model based on environment and hardware."""
    override = os.environ.get("SUMMARIZER_MODEL", "").lower().strip()
    if override in _MODELS:
        name = _MODELS[override]
        print(f"[Summarizer] Using {name} (SUMMARIZER_MODEL={override})")
        return name
    if torch.cuda.is_available():
        print("[Summarizer] GPU detected -> using bart-large-cnn")
        return _MODELS["bart"]
    print("[Summarizer] CPU mode -> using flan-t5-base (set SUMMARIZER_MODEL=large for upgrade)")
    return _MODELS["base"]


def _load_model():
    global _TOKENIZER, _MODEL, _MODEL_NAME
    if _MODEL is not None:
        return _TOKENIZER, _MODEL

    _MODEL_NAME = _select_model()
    _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)

    use_half = "large" in _MODEL_NAME and not torch.cuda.is_available()
    if use_half:
        print("[Summarizer] Loading in float16 to save memory...")
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME, torch_dtype=torch.float16)
    else:
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL.to(device)
    _MODEL.eval()

    param_count = sum(p.numel() for p in _MODEL.parameters()) / 1e6
    print(f"[Summarizer] Loaded {_MODEL_NAME} ({param_count:.0f}M params)")
    return _TOKENIZER, _MODEL


def summarize_raw(text: str, instruction: str = "summarize") -> str:
    """Summarize raw text directly with a focused single instruction."""
    if not text.strip():
        return ""

    tokenizer, model = _load_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        f"{instruction}: {text}",
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    if input_len == 0:
        return "Summarization failed: input text was empty after truncation."

    adaptive_min = min(40, max(10, input_len // 4))

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=MAX_SUMMARY_TOKENS,
            min_length=adaptive_min,
            do_sample=False,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _clean_summary(text: str, heading: str) -> str:
    """Removes redundant headings or 'Summary:' prefixes from model output."""
    clean = text.strip()
    # Remove common model echoes
    noise = [
        "summary:", "facts of the case:", "legal issues:", 
        "final verdict:", "judgment:", "case facts:"
    ]
    for n in noise:
        if clean.lower().startswith(n):
            clean = clean[len(n):].strip().lstrip(":").strip()
    
    # Capitalize first letter if it's now lowercase
    if clean:
        clean = clean[0].upper() + clean[1:]
    return clean


def summarize(ranked_chunks: List[Tuple[str, float, str]]) -> str:
    if not ranked_chunks:
        return "Summarization failed: no chunks provided."

    section_map = {
        "facts":     ("### Case Facts", "Summary of current case facts"),
        "arguments": ("### Legal Issues & Arguments", "Main legal contentions and issues"),
    }
    verdict_sections = {"judgment", "final_order"}

    brief_parts = []

    for sec_key, (heading, instruction) in section_map.items():
        text = " ".join(c for c, _, s in ranked_chunks if s == sec_key)
        if text:
            raw_sum = summarize_raw(text, instruction)
            brief_parts.append(heading)
            brief_parts.append(_clean_summary(raw_sum, heading))

    verdict_text = " ".join(c for c, _, s in ranked_chunks if s in verdict_sections)
    if verdict_text:
        raw_verdict = summarize_raw(verdict_text, "The final court decision and reasoning")
        brief_parts.append("### Final Verdict")
        brief_parts.append(_clean_summary(raw_verdict, "Final Verdict"))

    if not brief_parts:
        return summarize_raw(" ".join(c for c, _, _ in ranked_chunks), "General Case Summary")

    return "\n\n".join(brief_parts)


def reset_model():
    """Reset loaded model so next call loads fresh (used for A/B testing)."""
    global _TOKENIZER, _MODEL, _MODEL_NAME
    if _MODEL is not None:
        del _MODEL
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _TOKENIZER = None
    _MODEL = None
    _MODEL_NAME = None
