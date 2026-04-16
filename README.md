# Legal Intelligence Engine — System Architecture

The Legal Intelligence Engine is a pipeline designed to address a fundamental limitation of modern Large Language Models (LLMs): the context window bottleneck when processing verbose and unstructured legal judgments, such as those from the Supreme Court of India.

## The Core Problem

Most open-source models capable of running on edge devices or affordable virtual machines, such as FLAN-T5 or BART, are limited to approximately 1024 tokens.

Standard Indian legal judgments often exceed 10,000 words. When a naive LLM attempts to summarize such documents, it truncates large portions of the input. As a result, critical sections such as the *Judgment* and *Final Ordered Clauses* are often omitted, producing summaries that describe the facts but fail to capture the outcome.

## The Pipeline Workflow

The system processes unstructured text through six structured phases:

### Phase 0: PDF Pre-Cleaning

Before analysis, `main.py` strips PDF artefacts such as Manupatra page stamps, SCC watermarks, isolated page numbers, and institutional footers that corrupt extracted text.

### Phase 1: Context-Aware Segmentation

The input document, typically in plain text format, is processed by `segmenter.py`. Using regular expression pattern matching and semantic fallback mechanisms, the document is segmented into four standard legal sections:

1. Facts  
2. Arguments  
3. Judgment  
4. Final Order  

### Phase 2: Information Extraction (BNS and Financial Data)

Before abstractive summarization alters the source text, key structured information is extracted:

- **`financial_extractor.py`** uses a 3-layer verification system (word-boundary regex + heuristic bouncer + BERT cross-encoder) to extract and categorise monetary amounts.
- **`bns_mapper.py`** maps outdated IPC sections (for example, Section 302 IPC) to their modern equivalents under the Bharatiya Nyaya Sanhita.

### Phase 3: Semantic Chunk Ranking

Instead of naive truncation, `ranker.py` uses Sentence Transformers (`all-MiniLM-L6-v2`) to vectorize and rank text chunks.

A section-weighting strategy is applied:

Final Order > Judgment > Arguments > Facts

The ranker uses an adaptive chunk budget that scales with document size:
- Short (<10K words): 2 facts + 2 arguments + 4 judgment chunks
- Medium (10K–20K): 2 facts + 3 arguments + 6 judgment chunks
- Large (>20K words): 3 facts + 4 arguments + 8 judgment chunks

### Phase 4: Abstractive Generation

The highest-ranked chunks are concatenated and passed to the LLM. Models such as FLAN-T5 or BART generate a coherent multi-pass abstractive summary covering Case Facts, Legal Issues & Arguments, and Final Verdict.

---

## Scientific Validation

To evaluate the effectiveness of this architecture, a ROUGE-based comparison was conducted against a naive truncation baseline using identical models, tokenizers, and decoding parameters.

| Metric   | Baseline (Truncation) | Pipeline (Chunk Ranking) | Delta   | Winner   |
|----------|----------------------|--------------------------|---------|----------|
| ROUGE-1  | 0.5127               | 0.4829                   | -0.03   | Baseline |
| ROUGE-2  | 0.2255               | **0.2720**               | **+0.047** | **Pipeline** |
| ROUGE-L  | 0.3493               | 0.3315                   | -0.02   | Baseline |

### Key Finding
- On short texts (<512 words), the baseline sees 100% of the document — no ranking advantage.
- On medium texts, the pipeline captures more relevant detail (ROUGE-1: +0.07).
- ROUGE-2 (bigram overlap) consistently favours the pipeline, indicating better phrase-level accuracy.
- The pipeline's true advantage is qualitative: it always captures the verdict and final order, which naive truncation misses on long documents.

**Conclusion:**  
The Legal Intelligence Engine enables lightweight models to generate coherent, outcome-aware summaries of large legal documents by combining structured segmentation with semantic retrieval and prioritization.

