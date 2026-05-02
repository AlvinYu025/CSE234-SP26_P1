# Data Files

- `golden_qa.json`: submitted golden-data deliverable. It contains 25 manually curated RapidFire documentation Q&A pairs with source file names and line ranges. It now follows the same schema as the released validation set: `question_id`, `question`, `reference_answer`, and `source_evidence`.
- `golden_qa_source_audit.md`: one-to-one audit notes for the 25 curated Q&A pairs, including the raw `.rst` source spans used to verify each answer.

The provided `validation-set-golden-qa-pairs.json` in the project root is treated only as a validation/dev set for tuning and evaluation.

Historical validation/dev tuning artifacts, including the legacy converted validation file, were moved to `../archive/validation_dev_tuning_artifacts/`. The previous 40-item golden QA draft was moved to `../archive/data/golden_qa_40_before_diversity_filter.json`.
