# CSE/DSC 234 Project 1

## Overview

Question-answering system over the RapidFire AI documentation corpus.
The final pipeline is `main.py` (deterministic TF-IDF RAG).
`rapidfire_experiments.py` is the multi-config experiment driver used to
explore and compare configurations before selecting the final one.

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

Deterministic TF-IDF RAG using config **T05**:

| Parameter | Value |
|---|---|
| `chunk_tokens` | 420 |
| `chunk_overlap` | 80 |
| `top_k` | 1 |
| Title boost | enabled |
| Related sources | disabled |
| `source_expand_lines` | 0 |
| `context_expand_lines` | 20 |
| Max context tokens | 1850 |

Steps: load `.rst` corpus → chunk → build TF-IDF index → retrieve top-k
with title/symbol boost → select related sources → assemble context →
generate answer via LLM (extractive fallback if API unavailable).

### Dependencies

```
numpy
scikit-learn
openai
```

### Run

```bash
export OPENAI_API_KEY="$(cat ~/api-key.txt)"
export OPENAI_BASE_URL="https://tritonai-api.ucsd.edu/v1"
export RAG_MODEL="api-gpt-oss-120b"

# Generate validation answers (submission file)
python3 main.py \
  --input data/validation-set-golden-qa-pairs.json \
  --output outputs/validation_answers.json

# Generate answers for the curated golden set
python3 main.py \
  --input data/golden_qa.json \
  --output outputs/golden_answers.json
```

### Output location

`outputs/validation_answers.json` — one answer object per question,
fields: `question_id`, `answer`, `sources`, `retrieved_context`.

---

## Experiments — `rapidfire_experiments.py`

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

### Environment variables

```bash
export OPENAI_API_KEY="$(cat ~/api-key.txt)"
export OPENAI_BASE_URL="https://tritonai-api.ucsd.edu/v1"
export JUDGE_BASE_URL="https://tritonai-api.ucsd.edu/v1"
export JUDGE_MODEL="claude-sonnet-4-6"
export RAG_MODEL="api-gpt-oss-120b"
```

### Commands

```bash
# List all available configs
python3 rapidfire_experiments.py --mode list-configs

# Run a single dense config
python3 rapidfire_experiments.py --mode dense-single --dense-config-id D07 \
    --input data/validation-set-golden-qa-pairs.json \
    --experiments-path rapidfire_experiments/dense_D07

# Run all 12 dense configs
python3 rapidfire_experiments.py --mode dense-all \
    --input data/validation-set-golden-qa-pairs.json \
    --experiments-path rapidfire_experiments/dense_all

# Run a single TF-IDF config
python3 rapidfire_experiments.py --mode tfidf-single --tfidf-config-id T13 \
    --input data/validation-set-golden-qa-pairs.json \
    --experiments-path rapidfire_experiments/tfidf_T05

# Run all 13 TF-IDF configs
python3 rapidfire_experiments.py --mode tfidf-all \
    --input data/validation-set-golden-qa-pairs.json \
    --experiments-path rapidfire_experiments/tfidf_all
```

### Re-running judge scoring on existing runs

If `Judge Failures` is 25 in a completed run (judge API not configured
during the original run), patch all results in-place:

```bash
python3 rejudge_dense.py \
    --experiments-path rapidfire_experiments/dense_all
```

This overwrites each `run_evals_return.json` with correct generation
metrics and writes a summary to `rapidfire_experiments/dense_all/judge_results.csv`.

### Experiment artifacts

```
rapidfire_experiments/
  dense_all/
    D01_.../
      run_evals_return.json   # per-question results + summary metrics
      validation_answers.json # submission-format answers
    ...
    judge_results.csv         # summary table across all dense configs
  tfidf_all/
    T01_.../
      run_evals_return.json
      validation_answers.json
    ...
```

---

## Final Configuration Selection

Experiments were run on `data/validation-set-golden-qa-pairs.json`.
**T05** (`tfidf_420_80_k1_title_no_related_expand0`) was selected for
`main.py` based on retrieval score and generation quality across configs.

Key design choices:

- **TF-IDF over dense embeddings** — deterministic, no GPU required,
  competitive retrieval F1 on this documentation corpus
- **Title + symbol boost** — improves ranking for API-name and class-name queries
- **top_k=1, no related sources** — focuses on the single best-matching chunk,
  keeping sources precise and context concise
- **source_expand_lines=0** — reports exact chunk spans, no padding
- **context_expand_lines=20** — expands context window by 20 lines around the
  chunk for richer generation input without inflating reported sources
