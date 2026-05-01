# CSE/DSC 234 Project 1 Curated Report

## Context

The first round of experiments was exploratory and used the provided validation set to tune retrieval and generation settings. After that, I created a separate manually curated golden QA set from the RapidFire documentation. I then reused several stronger or methodologically different settings from the validation-stage archive and evaluated them on the curated set.

The curated-set results below should be read as a second evaluation pass on my own manually curated data, not as hidden-test results.

## Objective

The project implements a single-turn RAG question answering system over the RapidFire AI documentation corpus. For the curated-data run, the concrete command was:

```bash
python3 main.py --input data/golden_qa.json --output outputs/curated_golden_answers.json
```

Each output item contains:

- `question_id`
- `answer`
- `sources`
- `retrieved_context`

The `sources` field uses file names and line ranges, for example:

```json
{"file": "configs.rst", "lines": [9, 13]}
```

## Current Data Structure

- `sourcedocs.zip`: original RapidFire documentation corpus.
- `sourcedocs/`: extracted `.rst` documentation files used by the retriever.
- `validation-set-golden-qa-pairs.json`: provided validation/dev set used only for exploratory tuning and validation-stage metrics.
- `data/golden_qa.json`: current curated golden QA deliverable with 25 manually curated Q&A pairs.
- `data/golden_qa_source_audit.md`: one-to-one source audit for the 25 curated Q&A pairs.
- `archive/validation_dev_tuning_artifacts/`: archived validation-stage experiments, outputs, metrics, and logs.
- `archive/data/golden_qa_40_before_diversity_filter.json`: earlier 40-item curated draft before the diversity filter.
- `outputs/curated_golden_*`: final/c09 run artifacts on the 25 curated Q&A pairs.
- `outputs/curated_setting_runs/`: four additional archived settings rerun on the 25 curated Q&A pairs.
- `logs/curated_golden_run.log` and `logs/additional_curated_settings.log`: run summaries and observations.

## Implementation

The main pipeline is implemented in `main.py`.

Main steps:

1. Load RapidFire documentation from `sourcedocs/`.
2. Split `.rst` files into line-aware chunks.
3. Build a TF-IDF retrieval index.
4. Retrieve candidate chunks with cosine similarity or MMR, depending on the setting.
5. Apply optional lexical/title/API-symbol reranking.
6. Build a source-aware context under the token budget.
7. Generate an answer using the configured chat backend, with an extractive fallback if generation is unavailable.

## Curated Data

The current curated set contains 25 Q&A pairs. I reduced it from an earlier 40-item draft to improve diversity and reduce overlap. The final set covers 10 documentation files:

- `configs.rst`
- `experiment.rst`
- `ragspecs.rst`
- `promptothers.rst`
- `evalsfunctions.rst`
- `fitfunctions.rst`
- `models.rst`
- `trainers.rst`
- `generators.rst`
- `onlineagg.rst`

The questions cover factual, conceptual, procedural, schema/API behavior, and comparative types. Each answer is grounded in raw `.rst` source line ranges listed in `data/golden_qa.json` and checked in `data/golden_qa_source_audit.md`.

## Commands Used

I first converted the curated QA file into the evaluator-compatible schema:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path(".")
curated = json.loads((root / "data/golden_qa.json").read_text())
reference = [
    {
        "question_id": item["question_id"],
        "question": item["question"],
        "reference_answer": item["gold_answer"],
        "source_evidence": item["sources"],
    }
    for item in curated
]
(root / "outputs/curated_golden_eval_reference.json").write_text(
    json.dumps(reference, indent=2) + "\n"
)
PY
```

The main curated run used the archived best validation-stage setting, copied from:

```text
archive/validation_dev_tuning_artifacts/outputs/final_config.json
```

Answer generation:

```bash
python3 main.py --input data/golden_qa.json --output outputs/curated_golden_answers.json
```

Retrieval evaluation:

```bash
python3 Metrics/evaluate_retrieval.py --output outputs/curated_golden_answers.json --validation outputs/curated_golden_eval_reference.json --k 5 > outputs/curated_golden_retrieval_metrics.json
```

Generation evaluation:

```bash
python3 Metrics/run_judge.py --output outputs/curated_golden_answers.json --validation outputs/curated_golden_eval_reference.json --base-url https://tritonai-api.ucsd.edu/v1 --model claude-sonnet-4-6 > outputs/curated_golden_generation_metrics.json
```

I also reran four additional archived settings. The answer-generation commands were:

```bash
python3 main.py --config outputs/curated_setting_runs/configs/c01_similarity_180_30_k3.json --input data/golden_qa.json --output outputs/curated_setting_runs/answers/c01_similarity_180_30_k3_answers.json
python3 main.py --config outputs/curated_setting_runs/configs/c04_similarity_320_70_k5_lexical.json --input data/golden_qa.json --output outputs/curated_setting_runs/answers/c04_similarity_320_70_k5_lexical_answers.json
python3 main.py --config outputs/curated_setting_runs/configs/c07_mmr_260_55_fetch20_lexical.json --input data/golden_qa.json --output outputs/curated_setting_runs/answers/c07_mmr_260_55_fetch20_lexical_answers.json
python3 main.py --config outputs/curated_setting_runs/configs/c08_similarity_char_260_55_k5.json --input data/golden_qa.json --output outputs/curated_setting_runs/answers/c08_similarity_char_260_55_k5_answers.json
```

The retrieval evaluation commands were:

```bash
python3 Metrics/evaluate_retrieval.py --output outputs/curated_setting_runs/answers/c01_similarity_180_30_k3_answers.json --validation outputs/curated_golden_eval_reference.json --k 5 > outputs/curated_setting_runs/metrics/c01_similarity_180_30_k3_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/curated_setting_runs/answers/c04_similarity_320_70_k5_lexical_answers.json --validation outputs/curated_golden_eval_reference.json --k 5 > outputs/curated_setting_runs/metrics/c04_similarity_320_70_k5_lexical_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/curated_setting_runs/answers/c07_mmr_260_55_fetch20_lexical_answers.json --validation outputs/curated_golden_eval_reference.json --k 5 > outputs/curated_setting_runs/metrics/c07_mmr_260_55_fetch20_lexical_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/curated_setting_runs/answers/c08_similarity_char_260_55_k5_answers.json --validation outputs/curated_golden_eval_reference.json --k 5 > outputs/curated_setting_runs/metrics/c08_similarity_char_260_55_k5_retrieval_metrics.json
```

The generation judge commands were:

```bash
python3 Metrics/run_judge.py --output outputs/curated_setting_runs/answers/c01_similarity_180_30_k3_answers.json --validation outputs/curated_golden_eval_reference.json --base-url https://tritonai-api.ucsd.edu/v1 --model claude-sonnet-4-6 > outputs/curated_setting_runs/metrics/c01_similarity_180_30_k3_generation_metrics.json
python3 Metrics/run_judge.py --output outputs/curated_setting_runs/answers/c04_similarity_320_70_k5_lexical_answers.json --validation outputs/curated_golden_eval_reference.json --base-url https://tritonai-api.ucsd.edu/v1 --model claude-sonnet-4-6 > outputs/curated_setting_runs/metrics/c04_similarity_320_70_k5_lexical_generation_metrics.json
python3 Metrics/run_judge.py --output outputs/curated_setting_runs/answers/c07_mmr_260_55_fetch20_lexical_answers.json --validation outputs/curated_golden_eval_reference.json --base-url https://tritonai-api.ucsd.edu/v1 --model claude-sonnet-4-6 > outputs/curated_setting_runs/metrics/c07_mmr_260_55_fetch20_lexical_generation_metrics.json
python3 Metrics/run_judge.py --output outputs/curated_setting_runs/answers/c08_similarity_char_260_55_k5_answers.json --validation outputs/curated_golden_eval_reference.json --base-url https://tritonai-api.ucsd.edu/v1 --model claude-sonnet-4-6 > outputs/curated_setting_runs/metrics/c08_similarity_char_260_55_k5_generation_metrics.json
```

## Methods Tried

The first curated run reused `c09_title_related_420_80_k1`, which had been the best validation-stage setting. It uses word TF-IDF retrieval, similarity search, `top_k=1`, title/API-symbol lexical reranking, related-source expansion, and 160-line source expansion around the selected chunk.

I then ran four additional settings from the archive to compare different retrieval styles:

| Setting | Method |
|---|---|
| `c01_similarity_180_30_k3` | Small chunks, word TF-IDF similarity, top-3, no rerank |
| `c04_similarity_320_70_k5_lexical` | Larger chunks, word TF-IDF similarity, top-5, lexical rerank |
| `c07_mmr_260_55_fetch20_lexical` | MMR retrieval, wider fetch, lexical rerank |
| `c08_similarity_char_260_55_k5` | Character n-gram TF-IDF with lexical rerank |

## Curated Results

Overall is computed as:

```text
Overall = (Retrieval Score + Partial Generation Score) / 2
```

| Setting | Retrieval Score | F1@5 | P@5 | R@5 | Correctness | Faithfulness | Completeness | Partial Gen Score | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c09_title_related_420_80_k1` | 0.8400 | 0.8400 | 0.8400 | 0.8400 | 0.8000 | 1.0000 | 3.8000 | 0.8533 | 0.8467 |
| `c01_similarity_180_30_k3` | 0.6360 | 0.5480 | 0.4000 | 0.9600 | 0.9200 | 0.9600 | 4.4400 | 0.9227 | 0.7793 |
| `c04_similarity_320_70_k5_lexical` | 0.5404 | 0.4213 | 0.2800 | 0.9200 | 0.8800 | 0.9600 | 4.4000 | 0.9067 | 0.7236 |
| `c07_mmr_260_55_fetch20_lexical` | 0.5127 | 0.3360 | 0.2020 | 1.0000 | 0.9600 | 0.9600 | 4.5200 | 0.9413 | 0.7270 |
| `c08_similarity_char_260_55_k5` | 0.6031 | 0.4814 | 0.3280 | 1.0000 | 1.0000 | 1.0000 | 4.8000 | 0.9867 | 0.7949 |

## Analysis

The best overall setting on the curated set was still `c09_title_related_420_80_k1`, with an overall score of 0.8467. Its retrieval score was much higher than the four additional settings because it was designed to return a focused core chunk and expand nearby source lines.

The strongest generation-only result came from `c08_similarity_char_260_55_k5`, with a Partial Generation Score of 0.9867. This setting had lower retrieval precision but very high recall, and the generated answers were judged highly correct and complete.

The curated set is different from the original validation set. It contains more narrow API-definition questions, which exposed several retrieval misses for `c09` on questions about `configs.rst`, `promptothers.rst`, and `fitfunctions.rst`. The run log records these cases in `logs/curated_golden_run.log`.

## Artifacts

- `data/golden_qa.json`
- `data/golden_qa_source_audit.md`
- `outputs/curated_golden_answers.json`
- `outputs/curated_golden_retrieval_metrics.json`
- `outputs/curated_golden_generation_metrics.json`
- `outputs/curated_setting_runs/summary_metrics.json`
- `logs/curated_golden_run.log`
- `logs/additional_curated_settings.log`

## Notes

The provided validation set was used for exploratory tuning and archived separately. The curated results in this report are based on the 25 manually curated Q&A pairs in `data/golden_qa.json`.

## AI Usage Disclaimer

GPT was used to help organize the commands, summarize the metric outputs, polish the writing flow, and provide a concise overview of the project structure. The reported commands, data files, outputs, and metrics come from the project files listed in this report.
