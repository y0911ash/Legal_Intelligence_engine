"""
Phase 2: Chunking + Hybrid Ranking Engine
------------------------------------------
Simulates Longformer-style global attention via a retrieval-based
scoring function.

Scoring formula:
  S(C_i) = w1 * CosineSim(E_Ci, E_query)
          + w2 * KeywordDensity(C_i)
          + w3 * SectionWeight(C_i)

Weights: w1=0.6, w2=0.2, w3=0.2
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
from pipeline.segmenter import SegmentedDocument
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512        # words per chunk (not tokens — avoids tokenizer dep)
TOP_K = 5               # number of chunks returned to summarizer

RETRIEVAL_QUERY = (
    "final judgment, legal decision, court ruling, ordered, dismissed, held"
)

IMPORTANT_KEYWORDS = {
    "held", "ordered", "dismissed", "allowed", "directed",
    "section", "court", "judgment", "ruling", "liable",
    "convicted", "acquitted", "sentenced", "compensation",
    "penalty", "fine", "appeal", "petition", "bench"
}

SECTION_WEIGHTS = {
    "facts": 0.4,
    "arguments": 0.7,
    "judgment": 1.5,
    "final_order": 2.0,
    "unknown": 1.0       # fallback section label
}

SCORING_WEIGHTS = {
    "semantic": 0.6,
    "keyword": 0.2,
    "section": 0.2
}


# ---------------------------------------------------------------------------
# Model (loaded once, module-level)
# ---------------------------------------------------------------------------
_MODEL = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("[Ranker] Loading sentence-transformer model...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size // 2          # 50% overlap for context continuity
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 30:    # discard tiny trailing chunks
            chunks.append(chunk)
    return chunks


def keyword_density(chunk: str) -> float:
    """Fraction of words that are legal signal keywords."""
    words = chunk.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.rstrip(".,;:") in IMPORTANT_KEYWORDS)
    return hits / len(words)


def score_chunk(
    semantic_sim: float,
    kw_density: float,
    section_name: str
) -> float:
    section_w = SECTION_WEIGHTS.get(section_name, SECTION_WEIGHTS["unknown"])
    # Normalise section weight to [0, 1] for fair weighting
    normalised_section = section_w / max(SECTION_WEIGHTS.values())
    return (
        SCORING_WEIGHTS["semantic"]  * semantic_sim +
        SCORING_WEIGHTS["keyword"]   * kw_density +
        SCORING_WEIGHTS["section"]   * normalised_section
    )


def rank_chunks(
    doc: SegmentedDocument,
    query: str = RETRIEVAL_QUERY
) -> List[Tuple[str, float, str]]:
    """
    Returns a Balanced Selection of chunks to provide Case Diversity.
    Picks at least 2 facts, 2 arguments, and 3-4 judgment/order chunks.
    This ensures the 'Beginning/Middle/End' of 108 pages aren't missed.
    """
    model = _get_model()
    
    # Track segments by their section
    sectional_scored: Dict[str, List[Tuple[float, str]]] = {
        "facts": [], "arguments": [], "judgment": [], "final_order": []
    }

    # First, chunk and score everything
    query_emb = model.encode(query, convert_to_tensor=True)
    
    for section_name, text in doc.sections.items():
        if not text.strip(): continue
        
        chunks = chunk_text(text)
        if not chunks: continue
        
        chunk_embs = model.encode(chunks, convert_to_tensor=True)
        similarities = util.cos_sim(query_emb, chunk_embs)[0].cpu().numpy()
        
        # Determine the logical section category (merge judgment/final_order weights if needed)
        cat = "judgment" if section_name in ["judgment", "final_order"] else section_name
        if cat not in sectional_scored: cat = "judgment" # fallback

        for i, chunk in enumerate(chunks):
            s = score_chunk(
                semantic_sim=float(similarities[i]),
                kw_density=keyword_density(chunk),
                section_name=section_name
            )
            sectional_scored[cat].append((s, chunk, section_name))

    # Balanced Collection:
    # 2 Facts, 2 Args, 4 Judgment
    final_selection = []
    
    for cat, items in sectional_scored.items():
        items.sort(key=lambda x: x[0], reverse=True)
        # Dynamic pull based on section importance
        limit = 4 if cat == "judgment" else 2
        for score, chunk, orig_sec in items[:limit]:
            final_selection.append((chunk, score, orig_sec))

    return final_selection
