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
    query: str = RETRIEVAL_QUERY,
    top_k: int = TOP_K
) -> List[Tuple[str, float, str]]:
    """
    Returns top_k chunks as list of (chunk_text, score, section_name).

    Handles edge case: if a section is empty, it is skipped without error.
    """
    model = _get_model()
    all_chunks: List[str] = []
    metadata: List[str] = []           # section name per chunk

    for section_name, text in doc.sections.items():
        if not text.strip():
            continue
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            metadata.append(section_name)

    if not all_chunks:
        return []

    # Encode everything
    query_emb = model.encode(query, convert_to_tensor=True)
    chunk_embs = model.encode(all_chunks, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, chunk_embs)[0].cpu().numpy()

    scored: List[Tuple[float, int]] = []
    for i, chunk in enumerate(all_chunks):
        s = score_chunk(
            semantic_sim=float(similarities[i]),
            kw_density=keyword_density(chunk),
            section_name=metadata[i]
        )
        scored.append((s, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [(all_chunks[i], s, metadata[i]) for s, i in top]
