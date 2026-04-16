"""
Phase 2: Chunking + Hybrid Ranking Engine
------------------------------------------
Scoring formula:
  S(C_i) = 0.6 * CosineSim(E_Ci, E_query)
          + 0.2 * KeywordDensity(C_i)
          + 0.2 * SectionWeight(C_i)
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
from pipeline.segmenter import SegmentedDocument
from typing import Dict, List, Tuple

CHUNK_SIZE = 512
RETRIEVAL_QUERY = "final judgment, legal decision, court ruling, ordered, dismissed, held"

IMPORTANT_KEYWORDS = {
    "held", "ordered", "dismissed", "allowed", "directed",
    "section", "court", "judgment", "ruling", "liable",
    "convicted", "acquitted", "sentenced", "compensation",
    "penalty", "fine", "appeal", "petition", "bench",
}

SECTION_WEIGHTS = {
    "facts": 0.4, "arguments": 0.7,
    "judgment": 1.5, "final_order": 2.0, "unknown": 1.0,
}

W_SEMANTIC, W_KEYWORD, W_SECTION = 0.6, 0.2, 0.2
_MAX_SECTION_W = max(SECTION_WEIGHTS.values())

# Model (loaded once)
_MODEL = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("[Ranker] Loading sentence-transformer model...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into overlapping word-level chunks (50% overlap)."""
    words = text.split()
    if not words:
        return []
    step = chunk_size // 2
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), step)
        if len(words[i:i + chunk_size]) >= 30
    ]


def keyword_density(chunk: str) -> float:
    """Fraction of words that are legal signal keywords."""
    words = chunk.lower().split()
    if not words:
        return 0.0
    return sum(1 for w in words if w.rstrip(".,;:") in IMPORTANT_KEYWORDS) / len(words)


def score_chunk(semantic_sim: float, kw_density: float, section_name: str) -> float:
    norm_section = SECTION_WEIGHTS.get(section_name, SECTION_WEIGHTS["unknown"]) / _MAX_SECTION_W
    return W_SEMANTIC * semantic_sim + W_KEYWORD * kw_density + W_SECTION * norm_section


def rank_chunks(doc: SegmentedDocument, query: str = RETRIEVAL_QUERY) -> List[Tuple[str, float, str]]:
    """
    Balanced selection: 2 facts, 2 arguments, 4 judgment/order chunks.
    Ensures beginning/middle/end of long documents aren't missed.
    """
    model = _get_model()
    query_emb = model.encode(query, convert_to_tensor=True)

    sectional_scored: Dict[str, List[Tuple[float, str, str]]] = {
        "facts": [], "arguments": [], "judgment": [],
    }

    for section_name, text in doc.sections.items():
        if not text.strip():
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue

        chunk_embs = model.encode(chunks, convert_to_tensor=True)
        sims = util.cos_sim(query_emb, chunk_embs)[0].cpu().numpy()

        cat = "judgment" if section_name in ("judgment", "final_order") else section_name
        if cat not in sectional_scored:
            cat = "judgment"

        for i, chunk in enumerate(chunks):
            s = score_chunk(float(sims[i]), keyword_density(chunk), section_name)
            sectional_scored[cat].append((s, chunk, section_name))

    # Scale chunk budget with document size so large PDFs get better coverage.
    # Short (<10K words): facts=2, args=2, judgment=4
    # Medium (10K-20K):   facts=2, args=3, judgment=6
    # Large (>20K words): facts=3, args=4, judgment=8
    total_words = sum(len(v.split()) for v in doc.sections.values())
    if total_words > 20_000:
        limits = {"facts": 3, "arguments": 4, "judgment": 8}
    elif total_words > 10_000:
        limits = {"facts": 2, "arguments": 3, "judgment": 6}
    else:
        limits = {"facts": 2, "arguments": 2, "judgment": 4}

    final = []
    for cat, items in sectional_scored.items():
        items.sort(key=lambda x: x[0], reverse=True)
        limit = limits.get(cat, 2)
        for score, chunk, orig_sec in items[:limit]:
            final.append((chunk, score, orig_sec))

    return final
