# Validation/Dev Tuning Artifact Archive

This folder contains historical artifacts produced while the provided validation set was used as dev/evaluation data during optimization. These files are retained for auditability only and are not the submitted manually curated golden-data deliverable.

Contents:

- `data/validation_converted_qa_DO_NOT_SUBMIT.json`: legacy schema conversion of the provided validation set. Do not submit or describe this as hand-curated golden data.
- `experiments/`: deterministic retrieval experiment configs, outputs, and metrics evaluated against the provided validation set.
- `outputs/`: validation answer files, retrieval metrics, generation judge metrics, and intermediate optimization outputs.
- `logs/rapidfire.log`: historical optimization log for validation/dev tuning runs.

The submitted golden-data deliverable remains `data/golden_qa.json` in the project root. The provided `validation-set-golden-qa-pairs.json` also remains in the project root and is treated only as the validation/dev set.
