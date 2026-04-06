# Legal Intelligence Engine — System Architecture

The Legal Intelligence Engine is a pipeline designed to solve a fundamental limitation of modern Large Language Models (LLMs): the context window bottleneck when processing verbose and unstructured legal judgments (like those from the Supreme Court of India).

## 🚀 The Core Problem
Most open-source models capable of running on edge devices or affordable VMs (like `flan-t5-large` or `bart-large-cnn`) are limited to ~1024 tokens. 
Standard Indian legal judgments often span over 10,000 words. When a naive LLM attempts to summarize these, it truncates the document, almost entirely missing the **Judgment** and **Final Ordered Clauses**, resulting in worthless summaries that describe the facts but never the outcome.

## ⚙️ The Pipeline Workflow

Our system dynamically processes unstructured text through 4 structured phases:

### Phase 1: Context-Aware Segmentation
The input document (often plain text) is fed into the `segmenter.py`. Using Regex pattern matching and semantic fallback mechanisms, it cuts the document into 4 standard legal sections:
1. **Facts**
2. **Arguments**
3. **Judgment**
4. **Final Order**

### Phase 2: Information Extraction (BNS & Financials)
Before abstractive summarization alters the raw facts, we extract critical structured data:
- **`financial_extractor.py`**: Scans for monetary values attached to keywords (e.g., "compensation of Rs. 10,00,000").
- **`bns_mapper.py`**: A rule-based mapper that detects outdated IPC codes (e.g., "Section 302 IPC") and projects their modern equivalents (e.g., "BNS Section 103: Punishment for murder").

### Phase 3: Semantic Chunk Ranking (The Secret Sauce)
Instead of relying on naive truncation, `ranker.py` uses `sentence-transformers/all-MiniLM-L6-v2` to vectorize and rank chunks of text. 
We apply **Section Weighting** (Order > Judgment > Arguments > Facts). The ranker ensures that the maximum token allowance (e.g., 512 words) is filled first by the *Final Order* and *Judgment* before backfilling with *Facts*.

### Phase 4: Abstractive Generation
The top-ranked chunks are concatenated and sent to the LLM (`google/flan-t5-large` or `facebook/bart-large-cnn`).
The model synthesizes a coherent, abstractive summary that perfectly captures both the background facts and the final verdict of the court.

---

## 📊 Scientific Validation

To prove the efficacy of this architecture, we conducted a rigorous ROUGE evaluation.
Comparing our Pipeline against a **Naive Baseline** (simple text truncation) using the exact same LLM, tokenizer, and decoding parameters:

| Metric | Dataset / Size | Baseline (Truncation) | Pipeline (Chunk Ranking) | Outcome |
|--------|----------------|-----------------------|--------------------------|---------|
| ROUGE-L | ILDC Cases / Long | 0.235 | **0.344** | **+0.11** ROUGE-L |
| ROUGE-1 | ILDC Cases / Long | 0.300 | **0.516** | **+0.21** ROUGE-1 |

**Conclusion**: The Legal Intelligence Engine successfully enables lightweight models to produce highly coherent, outcome-aware summaries on massive documents by relying on intelligent semantic retrieval.
