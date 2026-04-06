# ANTIGRAVITY AGENT BRIEF
# Legal Intelligence Engine — Build & Verification Instructions

## MISSION
Build, verify, and run the Legal Intelligence Engine for Indian court
judgment summarization. All source code is already written — your job is
to set up the environment, fix any import/runtime errors, and confirm
each phase works end-to-end.

---

## PROJECT STRUCTURE (already exists)

```
legal_intelligence_engine/
├── main.py                         ← Master orchestrator (run this last)
├── requirements.txt
├── data/
│   └── ipc_to_bns.json             ← IPC→BNS lookup table (25 sections)
└── pipeline/
    ├── segmenter.py                 ← Phase 1: Section detection
    ├── ranker.py                    ← Phase 2: Hybrid chunk scoring
    ├── summarizer.py                ← Phase 4: Abstractive summarization
    ├── bns_mapper.py                ← Phase 5: IPC→BNS annotation
    ├── financial_extractor.py       ← Phase 6: Fine/penalty extraction
    └── evaluator.py                 ← ROUGE evaluation
```

---

## STEP-BY-STEP INSTRUCTIONS

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```
If torch is slow to install, use the CPU-only wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers rouge-score numpy
```

### Step 2 — Smoke test individual modules

Run each module's sanity check:

```bash
# Test segmenter
python -c "
from pipeline.segmenter import segment_legal_doc
result = segment_legal_doc('Facts of the case: accused was charged. Ordered: appeal dismissed.')
print(result.summary())
print('Sections:', list(result.sections.keys()))
"

# Test financial extractor (no model needed, instant)
python -c "
from pipeline.financial_extractor import extract_financials
r = extract_financials('The court imposed a fine of Rs. 50,000 and compensation of ₹1,00,000.')
print(r)
"

# Test BNS mapper (no model needed, instant)
python -c "
from pipeline.bns_mapper import map_statutes
text, changes = map_statutes('Convicted under Section 302 IPC and Section 420 IPC.')
print(text)
print(changes)
"
```

### Step 3 — Test ranker (downloads ~80MB model first time)
```bash
python -c "
from pipeline.segmenter import segment_legal_doc
from pipeline.ranker import rank_chunks
doc = segment_legal_doc(open('test_judgment.txt').read() if __import__('os').path.exists('test_judgment.txt') else 'Facts: accused. Arguments: counsel. Judgment: held guilty. Ordered: dismissed.')
chunks = rank_chunks(doc)
for c, s, sec in chunks:
    print(f'[{sec}] score={s:.3f} | {c[:80]}...')
"
```

### Step 4 — Full pipeline smoke test
```bash
python main.py
```
Expected output:
- 5 pipeline steps logged
- Summary paragraph (~2-3 sentences)
- Mapped summary with "[→ BNS Section 101: Murder]" annotation
- Statute changes list
- Financials dict with fine and compensation entries

### Step 5 — ROUGE baseline comparison (optional but recommended)
```bash
python -c "
from pipeline.evaluator import baseline_comparison

sample_text = open('sample_judgment.txt').read()  # use any ILDC case
reference = 'The appeal was dismissed. Conviction under Section 302 IPC upheld.'
ranked_summary = 'The court dismissed the appeal upholding the conviction for murder.'

result = baseline_comparison(sample_text, reference, ranked_summary)
print('Naive truncation ROUGE-L:', result['naive_truncation']['rougeL'])
print('Ranked system ROUGE-L:', result['retrieval_ranked']['rougeL'])
print('Delta:', result['improvement_delta'])
"
```

---

## KNOWN ISSUES TO WATCH FOR

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: pipeline.segmenter` | Running from wrong dir | `cd legal_intelligence_engine` first |
| `OSError: data/ipc_to_bns.json not found` | Path resolution | Ensure cwd is project root |
| Summarizer very slow on CPU | `bart-large-cnn` is 400MB | It auto-switches to `flan-t5-base` on CPU — wait ~2min first time |
| Empty financials output | Judgment has no ₹ symbols | Check for "Rs." format — extractor handles both |
| All text in 'facts' section | Old judgment, no headers | Fallback mode activates automatically — pipeline still works |

---

## VALIDATION CHECKLIST

Before marking complete, confirm:
- [ ] `python main.py` runs without errors
- [ ] Summary output is 2+ sentences and coherent
- [ ] BNS annotation appears in mapped_summary for Section 302 IPC sample
- [ ] Financials dict has at least one non-empty category for the sample
- [ ] ROUGE-L score is computed and > 0

---

## WHAT NOT TO CHANGE

- Do NOT modify the scoring weights in `ranker.py` (w1=0.6, w2=0.2, w3=0.2)
  without re-running ROUGE comparison — these are tuned values.
- Do NOT add fine-tuning code — out of scope for this phase.
- The `ipc_to_bns.json` file is final. Do not add sections without
  verifying against the official BNS 2023 gazette.
