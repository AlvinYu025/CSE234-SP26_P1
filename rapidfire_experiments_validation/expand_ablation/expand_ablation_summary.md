# Expand-Line Ablation on Validation Set

This ablation uses the same TF-IDF retrieval setting in `main.py` and varies only the shared source/context expansion window via `--expand-lines`.

| Setting | Expand | Retrieval | Generation | Overall mean | Correctness | Faithfulness | Completeness | Max prompt est. words*1.3 | Max prompt est. chars/4 | chars/4 > 2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 | 1 | 0.9421 | 0.9378 | 0.9400 | 0.9556 | 1.0000 | 4.29/5 | 1709 | 2465 | 1 |
| E02 | 2 | 0.9421 | 0.9244 | 0.9333 | 0.9556 | 0.9778 | 4.20/5 | 1709 | 2465 | 1 |
| E03 | 3 | 0.9421 | 0.9422 | 0.9422 | 1.0000 | 0.9778 | 4.24/5 | 1709 | 2465 | 1 |
| E04 | 4 | 0.9421 | 0.9230 | 0.9326 | 0.9333 | 0.9778 | 4.29/5 | 1709 | 2465 | 1 |

Commands used:

- `python3 main.py --input data/validation-set-golden-qa-pairs.json --output rapidfire_experiments_validation/expand_ablation/E01/validation_answers.json --corpus-dir sourcedocs --apikey-txt ~/api-key.txt --generation-model api-gpt-oss-120b --expand-lines 1`
- `python3 main.py --input data/validation-set-golden-qa-pairs.json --output rapidfire_experiments_validation/expand_ablation/E02/validation_answers.json --corpus-dir sourcedocs --apikey-txt ~/api-key.txt --generation-model api-gpt-oss-120b --expand-lines 2`
- `python3 main.py --input data/validation-set-golden-qa-pairs.json --output rapidfire_experiments_validation/expand_ablation/E03/validation_answers.json --corpus-dir sourcedocs --apikey-txt ~/api-key.txt --generation-model api-gpt-oss-120b --expand-lines 3`
- `python3 main.py --input data/validation-set-golden-qa-pairs.json --output rapidfire_experiments_validation/expand_ablation/E04/validation_answers.json --corpus-dir sourcedocs --apikey-txt ~/api-key.txt --generation-model api-gpt-oss-120b --expand-lines 4`

Notes:

- Retrieval is unchanged across these runs because `expand_lines` is applied after ranking, when constructing `sources` and `retrieved_context`.
- Generation scores come from `Metrics/run_judge.py` on the already generated answer files.
- Prompt estimates include the system prompt, the user prompt wrapper, the validation question, and `retrieved_context`.
