# Legal Intelligence Engine — System Architecture

The Legal Intelligence Engine is a pipeline designed to address a fundamental limitation of modern Large Language Models (LLMs): the context window bottleneck when processing verbose and unstructured legal judgments, such as those from the Supreme Court of India.

## The Core Problem

Most open-source models capable of running on edge devices or affordable virtual machines, such as FLAN-T5 or BART, are limited to approximately 1024 tokens.

Standard Indian legal judgments often exceed 10,000 words. When a naive LLM attempts to summarize such documents, it truncates large portions of the input. As a result, critical sections such as the *Judgment* and *Final Ordered Clauses* are often omitted, producing summaries that describe the facts but fail to capture the outcome.

## The Pipeline Workflow

The system processes unstructured text through four structured phases:

### Phase 1: Context-Aware Segmentation

The input document, typically in plain text format, is processed by `segmenter.py`. Using regular expression pattern matching and semantic fallback mechanisms, the document is segmented into four standard legal sections:

1. Facts  
2. Arguments  
3. Judgment  
4. Final Order  

### Phase 2: Information Extraction (BNS and Financial Data)

Before abstractive summarization alters the source text, key structured information is extracted:

- **`financial_extractor.py`** identifies monetary values associated with relevant keywords, such as “compensation of Rs. 10,00,000”.
- **`bns_mapper.py`** maps outdated IPC sections (for example, Section 302 IPC) to their modern equivalents under the Bharatiya Nyaya Sanhita.

### Phase 3: Semantic Chunk Ranking

Instead of naive truncation, `ranker.py` uses Sentence Transformers (`all-MiniLM-L6-v2`) to vectorize and rank text chunks.

A section-weighting strategy is applied:

Final Order > Judgment > Arguments > Facts

This ensures that the available token budget is prioritized for the most legally significant sections, particularly the Final Order and Judgment, before incorporating supporting context from earlier sections.

### Phase 4: Abstractive Generation

The highest-ranked chunks are concatenated and passed to the LLM. Models such as FLAN-T5 or BART generate a coherent abstractive summary that captures both the factual background and the final judicial outcome.

---

## Scientific Validation

To evaluate the effectiveness of this architecture, a ROUGE-based comparison was conducted against a naive truncation baseline using identical models, tokenizers, and decoding parameters.

| Metric   | Dataset / Size      | Baseline (Truncation) | Pipeline (Chunk Ranking) | Outcome              |
|----------|--------------------|------------------------|---------------------------|----------------------|
| ROUGE-L  | ILDC Cases (Long)  | 0.235                  | 0.344                     | +0.11 improvement     |
| ROUGE-1  | ILDC Cases (Long)  | 0.300                  | 0.516                     | +0.21 improvement     |

**Conclusion:**  
The Legal Intelligence Engine enables lightweight models to generate coherent, outcome-aware summaries of large legal documents by combining structured segmentation with semantic retrieval and prioritization.
