# CSE/DSC 234 Project 1 Report

## Objective

The goal of this project is to build a single-turn RAG question answering system for the RapidFire AI documentation corpus. For the validation run reported below, I used the provided validation file as the input questions file:

```bash
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/final_validation_answers.json
```

Each output item contains:

- `question_id`
- `answer`
- `sources`
- `retrieved_context`

The `sources` field uses file names and line ranges such as:

```json
{"file": "configs.rst", "lines": [23, 48]}
```

## Current Data Structure

- `sourcedocs.zip`: original RapidFire documentation corpus.
- `sourcedocs/`: extracted `.rst` documentation files used by the retriever.
- `validation-set-golden-qa-pairs.json`: provided validation/dev set. The reported scores below use this file directly as the gold QA/evidence file for testing.
- `validation-set-question-type-distribution.json`: provided validation question type metadata.
- `data/golden_qa.json`: 40 manually curated documentation Q&A pairs for the golden-data deliverable.
- `archive/validation_dev_tuning_artifacts/`: archived validation/dev tuning outputs, including earlier experiment outputs, metrics, logs, and the old schema-converted validation file.
- `outputs/`, `experiments/`, `logs/`: kept clean for new runs; historical runs are in the archive folder.

## Implementation

The final RAG pipeline is implemented in `main.py`.

Main steps:

1. Load RapidFire documentation from `sourcedocs/`.
2. Split `.rst` files into line-aware chunks.
3. Build a TF-IDF retrieval index.
4. Retrieve candidate chunks with cosine similarity.
5. Apply title/API-symbol lexical reranking.
6. Add related source spans for structured documentation questions.
7. Build a context under the token budget.
8. Generate an answer using the configured chat backend, with an extractive fallback.

Final selected settings:

- `chunk_tokens`: 420
- `chunk_overlap`: 80
- `top_k`: 1
- `fetch_k`: 8
- `rerank`: `title_lexical`
- `source_expand_lines`: 160
- `max_context_tokens`: 1850

## Commands Used

The main answer-generation command was:

```bash
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/final_validation_answers.json
```

The retrieval evaluation command was:

```bash
python3 Metrics/evaluate_retrieval.py --output outputs/final_validation_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/final_retrieval_metrics.json
```

The generation judge command was:

```bash
python3 Metrics/run_judge.py --output outputs/final_validation_answers.json --validation validation-set-golden-qa-pairs.json --base-url https://tritonai-api.ucsd.edu/v1 --model claude-sonnet-4-6 > outputs/final_generation_metrics.json
```

After project cleanup, these output files were moved into `archive/validation_dev_tuning_artifacts/outputs/`.

I also ran several intermediate validation passes while changing the retrieval and generation settings. The concrete answer-generation commands were:

```bash
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v1_final_gptoss_answers.json
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v2_c04_gptoss_answers.json
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v3_prompt_gptoss_answers.json
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v4_faithful_prompt_gptoss_answers.json
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v5_title_related_answers.json
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v6_expanded_context_answers.json
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v7_expanded160_answers.json
python3 main.py --input validation-set-golden-qa-pairs.json --output outputs/v8_core_first_answers.json
```

The concrete intermediate retrieval evaluation commands were:

```bash
python3 Metrics/evaluate_retrieval.py --output outputs/v1_final_gptoss_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/v1_final_gptoss_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/v2_c04_gptoss_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/v2_c04_gptoss_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/v3_prompt_gptoss_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/v3_prompt_gptoss_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/v5_title_related_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/v5_title_related_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/v6_expanded_context_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/v6_expanded_context_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/v7_expanded160_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/v7_expanded160_retrieval_metrics.json
python3 Metrics/evaluate_retrieval.py --output outputs/v8_core_first_answers.json --validation validation-set-golden-qa-pairs.json --k 5 > outputs/v8_core_first_retrieval_metrics.json
```

For the intermediate generation checks that produced judge metrics, I used:

```bash
python3 Metrics/run_judge.py --output outputs/v1_final_gptoss_answers.json --validation validation-set-golden-qa-pairs.json --base-url https://tritonai-api.ucsd.edu/v1 --model api-gpt-oss-120b > outputs/v1_final_gptoss_generation_metrics.json
python3 Metrics/run_judge.py --output outputs/v2_c04_gptoss_answers.json --validation validation-set-golden-qa-pairs.json --base-url https://tritonai-api.ucsd.edu/v1 --model api-gpt-oss-120b > outputs/v2_c04_gptoss_generation_metrics.json
python3 Metrics/run_judge.py --output outputs/v3_prompt_gptoss_answers.json --validation validation-set-golden-qa-pairs.json --base-url https://tritonai-api.ucsd.edu/v1 --model api-gpt-oss-120b > outputs/v3_prompt_gptoss_generation_metrics.json
```

## Methods Tried

I tested several retrieval settings by changing chunk size, overlap, retriever type, top-k, and reranking. The experiments were run with:

```bash
python3 run_experiments.py
```

The table below summarizes the validation retrieval results.

| Config | Chunk | Overlap | Retriever | Top-k | Rerank | F1@5 | P@5 | R@5 | Retrieval Score |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| `c01_similarity_180_30_k3` | 180 | 30 | similarity | 3 | none | 0.5584 | 0.4370 | 0.8722 | 0.6226 |
| `c02_similarity_220_45_k5` | 220 | 45 | similarity | 5 | none | 0.4765 | 0.3333 | 0.9389 | 0.5829 |
| `c03_similarity_260_55_k5_lexical` | 260 | 55 | similarity | 5 | lexical | 0.4796 | 0.3400 | 0.9167 | 0.5788 |
| `c04_similarity_320_70_k5_lexical` | 320 | 70 | similarity | 5 | lexical | 0.5041 | 0.3611 | 0.9500 | 0.6051 |
| `c05_similarity_420_90_k5_lexical` | 420 | 90 | similarity | 5 | lexical | 0.4990 | 0.3663 | 0.8944 | 0.5866 |
| `c06_mmr_260_55_k5` | 260 | 55 | MMR | 5 | none | 0.2991 | 0.1867 | 0.8167 | 0.4342 |
| `c07_mmr_260_55_fetch20_lexical` | 260 | 55 | MMR | 8->5 | lexical | 0.3295 | 0.2089 | 0.8667 | 0.4683 |
| `c08_similarity_char_260_55_k5` | 260 | 55 | char TF-IDF | 5 | lexical | 0.4995 | 0.3533 | 0.9444 | 0.5991 |
| `c09_title_related_420_80_k1` | 420 | 80 | similarity | 1 | title/related | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The best setting was `c09_title_related_420_80_k1`. Compared with the earlier configs, it used larger chunks, smaller overlap than `c05`, only one final top source, title/API-symbol lexical reranking, and related-source expansion for documentation pages that need connected evidence.

I also kept the intermediate `v1` to `v8` validation runs in the archive. These were local validation/dev checks while changing retrieval, prompt wording, source expansion, and context ordering.

| Run | Main Change | Retrieval Score | Partial Generation Score |
|---|---|---:|---:|
| `v1_final_gptoss` | First full validation answer run from the earlier best retrieval setup | 0.6226 | 0.8474 |
| `v2_c04_gptoss` | Used the higher-recall `c04` retrieval setting | 0.6043 | 0.8637 |
| `v3_prompt_gptoss` | Prompt wording change for more complete grounded answers | 0.6226 | 0.8785 |
| `v4_faithful_prompt_gptoss` | Faithfulness-focused prompt trial | not recorded | not recorded |
| `v5_title_related` | Title/API-symbol reranking plus related-source retrieval | 1.0000 | not run |
| `v6_expanded_context` | Expanded context around retrieved source spans | 1.0000 | not run |
| `v7_expanded160` | Increased source expansion to 160 lines | 1.0000 | not run |
| `v8_core_first` | Reordered context so the core retrieved chunk appears first | 1.0000 | not run |
| `final_validation` | Final selected setup with generation judge run | 1.0000 | 0.9793 |

## Validation Results

These results were obtained by using `validation-set-golden-qa-pairs.json` directly as the gold QA/evidence file for testing. They are validation/dev results, not hidden-test results.

Final retrieval metrics on the 45-question validation set:

- F1@5: 1.0000
- Precision@5: 1.0000
- Recall@5: 1.0000
- Retrieval Score: 1.0000
- Missing outputs: 0
- Malformed sources: 0

Final generation judge metrics:

- Correctness pass rate: 1.0000
- Faithfulness pass rate: 1.0000
- Completeness: 4.6889 / 5
- Partial Generation Score: 0.9793
- Judge failures: 0

The final validation artifacts are archived at:

- `archive/validation_dev_tuning_artifacts/outputs/final_validation_answers.json`
- `archive/validation_dev_tuning_artifacts/outputs/final_retrieval_metrics.json`
- `archive/validation_dev_tuning_artifacts/outputs/final_generation_metrics.json`
- `archive/validation_dev_tuning_artifacts/outputs/final_config.json`

## Notes

The manually curated `data/golden_qa.json` is separate from the provided validation set. The high validation scores above come from testing against the provided validation gold file, not from hidden evaluation data.

## AI Usage Disclaimer

GPT was used to help organize the commands, summarize the metric outputs, polish the writing flow, and provide a concise overview of the project structure. The reported commands, data files, outputs, and metrics come from the project files listed in this report.
