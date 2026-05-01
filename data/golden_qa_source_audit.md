# Golden QA Source Audit

This file records the source audit for the current `data/golden_qa.json`.

The current deliverable contains 25 Q&A pairs selected from the original 40-item draft after a diversity pass. I removed highly similar API-detail questions and kept a broader mix across document areas and question types. The previous 40-item draft is archived at `archive/data/golden_qa_40_before_diversity_filter.json`.

The source line numbers refer to the raw `.rst` files under `sourcedocs/`, not rendered documentation pages. To inspect a source span manually, run:

```bash
nl -ba sourcedocs/<file>.rst | sed -n '<start>,<end>p'
```

Example:

```bash
nl -ba sourcedocs/configs.rst | sed -n '9,13p'
```

| ID | Type | Source Span(s) | Audit Result |
|---:|---|---|---|
| 1 | Conceptual | `sourcedocs/configs.rst:9-13` | Pass: answer lists the knob types named in the source. |
| 2 | Factual/API | `sourcedocs/configs.rst:33-47` | Pass: answer states the `Range()` parameters and uniform sampling note. |
| 3 | Procedural | `sourcedocs/experiment.rst:56-63` | Pass: answer states why to end experiments and how reconnect works. |
| 4 | Procedural/API | `sourcedocs/experiment.rst:116-120` | Pass: answer lists the supported `param_config` forms. |
| 5 | Comparative | `sourcedocs/experiment.rst:186-189` | Pass: answer maps `run_evals()` `num_shards` to `run_fit()` `num_chunks`. |
| 6 | Factual | `sourcedocs/ragspecs.rst:4-6` | Pass: answer lists the RAG stages in the source. |
| 7 | Comparative/Conceptual | `sourcedocs/ragspecs.rst:21-25` | Pass: answer contrasts document preprocessing and query processing. |
| 8 | Procedural | `sourcedocs/ragspecs.rst:61-65` | Pass: answer describes create/read/update vector store modes. |
| 9 | Comparative | `sourcedocs/ragspecs.rst:69-88` | Pass: answer compares FAISS and Pinecone behavior. |
| 10 | API Behavior | `sourcedocs/ragspecs.rst:161-175` | Pass: answer describes `get_context()` options and return behavior. |
| 11 | Conceptual | `sourcedocs/promptothers.rst:7-12` | Pass: answer describes RFPromptManager and few-shot selection. |
| 12 | Procedural | `sourcedocs/evalsfunctions.rst:38-77` | Pass: answer follows the FiQA preprocessing example. |
| 13 | Procedural/Multimodal | `sourcedocs/evalsfunctions.rst:116-157` | Pass: answer follows the multimodal preprocessing example. |
| 14 | Schema | `sourcedocs/evalsfunctions.rst:220-234` | Pass: answer states the expected metric output schema. |
| 15 | Procedural | `sourcedocs/fitfunctions.rst:34-59` | Pass: answer follows the sample model creation branches. |
| 16 | Schema | `sourcedocs/fitfunctions.rst:113-153` | Pass: answer states the formatting function return shape. |
| 17 | Factual | `sourcedocs/fitfunctions.rst:232-234` | Pass: answer states allowed reward-function returns. |
| 18 | Conceptual/API | `sourcedocs/models.rst:13-20` | Pass: answer describes RFLoraConfig's List/Range extension. |
| 19 | Conceptual | `sourcedocs/models.rst:74-83` | Pass: answer describes RFModelConfig responsibilities. |
| 20 | Factual/API | `sourcedocs/trainers.rst:42-55` | Pass: answer names the two List-valued RFSFTConfig arguments. |
| 21 | Comparative | `sourcedocs/trainers.rst:65-97` | Pass: answer compares FSDP `offload_params` values in the examples. |
| 22 | Factual/API | `sourcedocs/trainers.rst:121-143` | Pass: answer identifies the List-valued DPO `beta` parameter. |
| 23 | Factual/API | `sourcedocs/generators.rst:23-28` | Pass: answer lists recommended RFvLLMModelConfig keys. |
| 24 | Conceptual | `sourcedocs/onlineagg.rst:51-54` | Pass: answer states how confidence intervals narrow. |
| 25 | Comparative | `sourcedocs/onlineagg.rst:203-224` | Pass: answer states the batch vs RapidFire runtime comparison. |
