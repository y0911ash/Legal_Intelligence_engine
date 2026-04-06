"""
Phase 4: Summarization
-----------------------
Takes top-ranked chunks and generates an abstractive summary.

Model hierarchy (auto-selects best available):
  - CUDA GPU        -> facebook/bart-large-cnn
  - CPU + 'large'   -> google/flan-t5-large  (float16, ~1.5GB RAM)
  - CPU + 'base'    -> google/flan-t5-base   (default, ~1GB RAM)

Set environment variable SUMMARIZER_MODEL to override:
  SUMMARIZER_MODEL=large   -> force flan-t5-large
  SUMMARIZER_MODEL=base    -> force flan-t5-base

Note: Uses AutoModelForSeq2SeqLM + generate() directly because
      transformers >=5.x removed the "summarization" pipeline alias.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple

MAX_INPUT_TOKENS = 2048     # expanded for complex cases
MAX_SUMMARY_TOKENS = 512    # allow multi-paragraph output
MIN_SUMMARY_TOKENS = 120    # prevent overly vague summaries

# Model registry
_MODELS = {
    "base":  "google/flan-t5-base",
    "large": "google/flan-t5-large",
    "bart":  "facebook/bart-large-cnn",
}


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
    else:
        print("[Summarizer] CPU mode -> using flan-t5-base (set SUMMARIZER_MODEL=large for upgrade)")
        return _MODELS["base"]


_TOKENIZER = None
_MODEL = None
_MODEL_NAME = None

def _load_model():
    global _TOKENIZER, _MODEL, _MODEL_NAME
    if _MODEL is None:
        _MODEL_NAME = _select_model()
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)

        # Use float16 for large models on CPU to save memory
        use_half = "large" in _MODEL_NAME and not torch.cuda.is_available()

        if use_half:
            print("[Summarizer] Loading in float16 to save memory...")
            _MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                _MODEL_NAME,
                torch_dtype=torch.float16,
            )
        else:
            _MODEL = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.to(device)
        _MODEL.eval()

        param_count = sum(p.numel() for p in _MODEL.parameters()) / 1e6
        print(f"[Summarizer] Loaded {_MODEL_NAME} ({param_count:.0f}M params)")

    return _TOKENIZER, _MODEL


def _prepare_input(ranked_chunks: List[Tuple[str, float, str]]) -> str:
    """
    Concatenate top chunks into a single input string.
    Chunks are sorted so final_order and judgment sections come first
    (highest section weight = most important).
    """
    priority = {"final_order": 0, "judgment": 1, "arguments": 2, "facts": 3}
    sorted_chunks = sorted(
        ranked_chunks,
        key=lambda x: priority.get(x[2], 99)
    )
    combined = " ".join(chunk for chunk, _, _ in sorted_chunks)
    return combined


def summarize(ranked_chunks: List[Tuple[str, float, str]]) -> str:
    """
    Section-by-Section Multi-Pass Summarizer
    -----------------------------------------
    Generates a full-page structured legal brief.
    Runs 3 separate LLM passes to provide deep detail.
    """
    if not ranked_chunks:
        return "Summarization failed: no chunks provided."

    # Group chunks by section
    facts_text = " ".join(c for c, _, s in ranked_chunks if s == "facts")
    args_text  = " ".join(c for c, _, s in ranked_chunks if s == "arguments")
    # Judgment + Final Order are high priority
    verdict_text = " ".join(c for c, _, s in ranked_chunks if s in ["judgment", "final_order"])
    
    # Fallback if specific sections weren't found: just use the ranker's order
    if not any([facts_text, args_text, verdict_text]):
        input_text = " ".join(c for c, _, _ in ranked_chunks)
        return summarize_raw(input_text, "Full Legal Brief")

    brief_parts = []
    
    if facts_text:
        brief_parts.append("### 📜 FACTUAL BACKGROUND")
        brief_parts.append(summarize_raw(facts_text, "Case Facts"))
    
    if args_text:
        brief_parts.append("### ⚖️ LEGAL ARGUMENTS")
        brief_parts.append(summarize_raw(args_text, "Court Arguments"))
        
    if verdict_text:
        brief_parts.append("### 🏛️ JUDICIAL REASONING & VERDICT")
        brief_parts.append(summarize_raw(verdict_text, "Rulings"))

    return "\n\n".join(brief_parts)


def summarize_raw(text: str, context_label: str = "Summary") -> str:
    """
    Summarize raw text directly. 
    Refined with 'context_label' to harden the prompt and focus the model 
    exclusively on legal outcome rather than document noise.
    """
    if not text.strip():
        return f"{context_label} skipped: input text was empty."

    tokenizer, model = _load_model()
    device = next(model.parameters()).device

    input_text = text
    # Refined legal-focused instruction for FLAN models
    if "flan" in (_MODEL_NAME or ""):
        input_text = f"Summarize the {context_label} of this court document in detail. Ignore all document metadata and noise: " + input_text
    else: 
        # BART needs the context in the beginning
        input_text = f"LEGAL ANALYSIS ({context_label}): " + input_text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True
    ).to(device)

    # Cast input to model dtype (needed for float16)
    input_token_count = inputs["input_ids"].shape[1]

    if input_token_count == 0:
        return "Summarization failed: input text was empty after truncation."

    # Adaptive min_length: at most 1/3 of input tokens, capped at MIN_SUMMARY_TOKENS
    # Prevents degenerate output when input is short
    adaptive_min = min(MIN_SUMMARY_TOKENS, max(10, input_token_count // 3))

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=MAX_SUMMARY_TOKENS,
            min_length=adaptive_min,
            do_sample=False,
            num_beams=2,
            length_penalty=1.5,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()


def reset_model():
    """Reset loaded model so next call loads fresh (used for A/B testing)."""
    global _TOKENIZER, _MODEL, _MODEL_NAME
    if _MODEL is not None:
        del _MODEL
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    _TOKENIZER = None
    _MODEL = None
    _MODEL_NAME = None
