# CSE/DSC 234 Project 1

## Overview

Question-answering system over the RapidFire AI documentation corpus.
The final pipeline is `main.py` (deterministic TF-IDF RAG).
`rapidfire_experiments_golden.py` and `rapidfire_experiments_validation.py`
are the multi-config experiment drivers used to compare configurations.

---

## Data

| File | Description |
|---|---|
| `data/validation-set-golden-qa-pairs.json` | Released validation/dev set — 45 questions (used for evaluation and submission) |
| `data/golden_qa.json` | Manually curated golden QA set — 25 questions (used during experiment exploration) |
| `data/validation-set-question-type-distribution.json` | Question-type breakdown for the validation set |
| `sourcedocs/*.rst` | RapidFire AI documentation corpus (auto-extracted from `sourcedocs.zip` on first run) |

### Schema — input QA file

```json
[
  {
    "question_id": 1,
    "question": "...",
    "reference_answer": "...",
    "source_evidence": [{"file": "configs.rst", "lines": [9, 13]}]
  }
]
```

### Schema — output answers file

```json
[
  {
    "question_id": 1,
    "answer": "...",
    "sources": [{"file": "configs.rst", "lines": [1, 77]}],
    "retrieved_context": "..."
  }
]
```

---

## Final Pipeline — `main.py`

### What it does

Deterministic TF-IDF RAG using config **TC02**:

| Parameter | Value |
|---|---|
| `chunk_tokens` | 420 |
| `chunk_overlap` | 80 |
| `top_k` | 1 |
| Title boost | enabled |
| Related sources | enabled |
| `source_expand_lines` | 3 |
| `context_expand_lines` | 3 |
| Prompt token cap | 1900 real `api-gpt-oss-120b` prompt tokens |

Steps: load `.rst` corpus → chunk → build TF-IDF index → retrieve top-k
with title/symbol boost → select related sources → assemble token-capped
context using the generation model's `usage.prompt_tokens` → generate answer
via LLM (extractive fallback if API unavailable).

### Dependencies

```
numpy
scikit-learn
openai
```

### Run

```bash
# Generate validation answers (submission file)
python3 main.py \
  --input data/validation-set-golden-qa-pairs.json \
  --output outputs/validation_answers.json \
  --corpus-dir sourcedocs \
  --apikey-txt ~/api-key.txt \
  --generation-model api-gpt-oss-120b

# Generate answers for the curated golden set
python3 main.py \
  --input data/golden_qa.json \
  --output outputs/golden_answers.json \
  --corpus-dir sourcedocs \
  --apikey-txt ~/api-key.txt \
  --generation-model api-gpt-oss-120b
```

### Output location

`outputs/validation_answers.json` — one answer object per question,
fields: `question_id`, `answer`, `sources`, `retrieved_context`.
The final saved submission is available at
`submission/validation_answers.json`.

---

## Experiments

### What it does

Runs multiple RAG configurations through RapidFire AI's `Experiment.run_evals`
framework, computing retrieval (F1/Precision/Recall@5) and generation
(Correctness, Faithfulness, Completeness via LLM judge) metrics for each.

Two config families:

**Dense (D01–D12)** — LangChain + FAISS + CrossEncoder reranker:
- 2 chunk sizes: 320/64, 512/96
- 3 search strategies: similarity k=5, similarity k=8, MMR k=8
- 2 reranker depths: top_n=3, top_n=5
- → 12 configs

**TF-IDF (T01–T13)** — custom deterministic retriever:
- Ablations over chunk size, title boost, related-source expansion,
  source_expand_lines, top_k
- → 13 configs

Each config writes its own folder under `--experiments-path` containing:
- `run_evals_return.json` — per-question retrieval spans, generated answers, and summary metrics
- `validation_answers.json` — answers in submission format

### Dependencies

Install in a Python 3.12 environment:

```bash
pip install -r requirements.txt
```

### API key

The scripts read the gateway key from `--apikey-txt` and configure the
OpenAI-compatible gateway internally. They do not require manually exporting
`OPENAI_API_KEY` before running.

### Commands

```bash
# Golden-set experiments write to rapidfire_experiments_golden/ by default.
python3 rapidfire_experiments_golden.py --mode list-configs

# Run a single dense config
python3 rapidfire_experiments_golden.py --mode dense-single --dense-config-id D07 \
    --input data/golden_qa.json \
    --experiments-path rapidfire_experiments_golden/dense_D07 \
    --apikey-txt ~/api-key.txt

# Run all 12 dense configs
python3 rapidfire_experiments_golden.py --mode dense-all \
    --input data/golden_qa.json \
    --experiments-path rapidfire_experiments_golden/dense_all \
    --apikey-txt ~/api-key.txt

# Run a single TF-IDF config
python3 rapidfire_experiments_golden.py --mode tfidf-single --tfidf-config-id T13 \
    --input data/golden_qa.json \
    --experiments-path rapidfire_experiments_golden/tfidf_T13 \
    --apikey-txt ~/api-key.txt

# Run all 13 TF-IDF configs
python3 rapidfire_experiments_golden.py --mode tfidf-all \
    --input data/golden_qa.json \
    --experiments-path rapidfire_experiments_golden/tfidf_all \
    --apikey-txt ~/api-key.txt

# Released validation/dev-set experiments write to rapidfire_experiments_validation/ by default.
python3 rapidfire_experiments_validation.py --mode tfidf-all \
    --input data/validation-set-golden-qa-pairs.json \
    --experiments-path rapidfire_experiments_validation/tfidf_all \
    --apikey-txt ~/api-key.txt
```

### Re-running judge scoring on existing runs

If `Judge Failures` is 25 in a completed run (judge API not configured
during the original run), patch all results in-place:

```bash
python3 rejudge_dense.py \
    --experiments-path rapidfire_experiments_golden/dense_all
```

This overwrites each `run_evals_return.json` with correct generation
metrics and writes a summary to `rapidfire_experiments_golden/dense_all/judge_results.csv`.

### Expand-line ablation

The released validation set was used to compare the same TF-IDF retrieval
setting while varying the shared `--expand-lines` value:

| Setting | Expand | Retrieval | Generation | Overall mean |
|---|---:|---:|---:|---:|
| E01 | 1 | 0.9421 | 0.9378 | 0.9400 |
| E02 | 2 | 0.9421 | 0.9244 | 0.9333 |
| E03 | 3 | 0.9421 | 0.9422 | 0.9422 |
| E04 | 4 | 0.9421 | 0.9230 | 0.9326 |

The table above preserves the recorded summary. Generated per-run files are
excluded from the repository.

### Token-budget context ablation

Additional validation experiments measured real prompt tokens from the
generation model API (`response.usage.prompt_tokens`) instead of relying on
local estimates. The goal was to keep the full LLM prompt under the 2,000-token
context budget.

| Setting | Chunk | Related | Context packing | Cap | Retrieval | Generation | Overall mean | Max prompt tokens |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| TC01 | 420/80 | 5 | sequential | 1950 | 0.9389 | 0.9422 | 0.9406 | 1909 |
| TC02 | 420/80 | 5 | sequential | 1900 | 0.9389 | 0.9496 | 0.9443 | 1858 |
| TC03 | 384/64 | 5 | sequential | 1900 | 0.8199 | 0.8681 | 0.8440 | 1895 |
| TC04 | 320/64 | 5 | sequential | 1900 | 0.8310 | 0.8711 | 0.8511 | 1891 |
| TC05 | 256/48 | 5 | sequential | 1900 | 0.7755 | 0.8148 | 0.7951 | 1880 |
| TC06 | 420/80 | 4 | sequential | 1900 | 0.9389 | 0.9304 | 0.9346 | 1858 |
| TC07 | 420/80 | 3 | sequential | 1900 | 0.9351 | 0.9481 | 0.9416 | 1744 |
| TC08 | 320/64 | 5 | balanced | 1900 | 0.8310 | 0.8726 | 0.8518 | 1891 |
| TC09 | 320/64 | 5 | question-aware | 1900 | 0.8310 | 0.8326 | 0.8318 | 1900 |
| TC10 | 256/48 | 5 | staged | 1900 | 0.7755 | 0.8074 | 0.7914 | 1892 |

Best validation overall in this set was **TC02**. Detailed generated answer
files, token traces, and debug outputs are excluded from the repository.

### Experiment artifacts

Experiment scripts are retained, while reproducible run directories and logs
are ignored. The important aggregate results remain documented above.

---

## Final Configuration Selection

Experiments in `rapidfire_experiments_golden/`, follow-up validation alignment
tests, and token-budget ablations were used to select the final single
configuration. **TC02** (`chunk420_overlap80_rel5_sequential_cap1900`) was
selected for `main.py` because it had the best validation overall score in the
token-budget ablation while keeping every measured prompt under 2,000 real
generation-model prompt tokens.

Key design choices:

- **TF-IDF over dense embeddings** — deterministic, no GPU required,
  competitive retrieval F1 on this documentation corpus
- **Title + symbol boost** — improves ranking for API-name and class-name queries
- **top_k=1 with related sources** — starts from the best-matching chunk and
  adds targeted related documentation for tutorial, vector-store, glossary, and
  comparison-style questions
- **source_expand_lines=3 and context_expand_lines=3** — uses the same
  expanded evidence window for both reported source spans and generation
  context, keeping retrieval reporting aligned with what the model sees
- **prompt_token_cap=1900** — applies a real prompt-token preflight with the
  generation model and trims the final context block at complete text units
  when needed
