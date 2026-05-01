# Golden QA Source Audit

This file records the source audit for `data/golden_qa.json`.

The source line numbers refer to the raw `.rst` files under `sourcedocs/`, not rendered documentation pages. To inspect a source span manually, run:

```bash
nl -ba sourcedocs/<file>.rst | sed -n '<start>,<end>p'
```

Example:

```bash
nl -ba sourcedocs/configs.rst | sed -n '9,13p'
```

| ID | Source Span(s) | Audit Result |
|---:|---|---|
| 1 | `sourcedocs/configs.rst:9-13` | Pass: answer lists the knob types named in the source. |
| 2 | `sourcedocs/configs.rst:33-47` | Pass: answer states the Range parameters and uniform sampling note. |
| 3 | `sourcedocs/configs.rst:50-51`, `sourcedocs/configs.rst:101-102` | Pass: answer uses both internal-return-type notes. |
| 4 | `sourcedocs/experiment.rst:7-13` | Pass: answer states unique naming and suffix behavior. |
| 5 | `sourcedocs/experiment.rst:56-63` | Pass: answer states why to end experiments and how reconnect works. |
| 6 | `sourcedocs/experiment.rst:116-120` | Pass: answer lists the supported `param_config` forms. |
| 7 | `sourcedocs/experiment.rst:127-129` | Pass: answer describes `num_chunks` and the recommendation. |
| 8 | `sourcedocs/experiment.rst:186-189` | Pass: answer maps `num_shards` to `num_chunks`. |
| 9 | `sourcedocs/experiment.rst:247-253` | Pass: answer describes `get_results()` return contents. |
| 10 | `sourcedocs/ragspecs.rst:4-6` | Pass: answer lists the RAG stages in the source. |
| 11 | `sourcedocs/ragspecs.rst:21-25` | Pass: answer contrasts document preprocessing and query processing. |
| 12 | `sourcedocs/ragspecs.rst:53-54` | Pass: answer states the `embedding_cfg` dictionary requirement. |
| 13 | `sourcedocs/ragspecs.rst:61-65` | Pass: answer describes create/read/update modes. |
| 14 | `sourcedocs/ragspecs.rst:69-88` | Pass: answer compares FAISS and Pinecone behavior. |
| 15 | `sourcedocs/ragspecs.rst:78-84` | Pass: answer states the Pinecone `text_key` purpose and default. |
| 16 | `sourcedocs/ragspecs.rst:90-95` | Pass: answer states why `pre_delete_collection` needs caution. |
| 17 | `sourcedocs/ragspecs.rst:161-175` | Pass: answer describes `get_context()` options and return behavior. |
| 18 | `sourcedocs/ragspecs.rst:366-370` | Pass: answer describes retriever vs vector-store control flow. |
| 19 | `sourcedocs/promptothers.rst:7-12` | Pass: answer describes RFPromptManager and few-shot selection. |
| 20 | `sourcedocs/promptothers.rst:95-100` | Pass: answer lists required `online_strategy_kwargs` keys. |
| 21 | `sourcedocs/evalsfunctions.rst:38-77` | Pass: answer follows the FiQA preprocessing example. |
| 22 | `sourcedocs/evalsfunctions.rst:116-157` | Pass: answer follows the multimodal preprocessing example. |
| 23 | `sourcedocs/evalsfunctions.rst:186-213` | Pass: answer identifies the two postprocess examples. |
| 24 | `sourcedocs/evalsfunctions.rst:220-234` | Pass: answer states the expected metric output schema. |
| 25 | `sourcedocs/evalsfunctions.rst:298-317` | Pass: answer lists optional metric metadata keys. |
| 26 | `sourcedocs/fitfunctions.rst:34-59` | Pass: answer follows the sample model creation branches. |
| 27 | `sourcedocs/fitfunctions.rst:66-75` | Pass: answer states invocation cadence and List variants. |
| 28 | `sourcedocs/fitfunctions.rst:113-153` | Pass: answer states the formatting function return shape. |
| 29 | `sourcedocs/fitfunctions.rst:232-234` | Pass: answer states allowed reward-function returns. |
| 30 | `sourcedocs/models.rst:13-20` | Pass: answer describes RFLoraConfig's List/Range extension. |
| 31 | `sourcedocs/models.rst:74-83` | Pass: answer describes RFModelConfig responsibilities. |
| 32 | `sourcedocs/models.rst:171-174` | Pass: answer states the one-base-model limitation and allowed Lists. |
| 33 | `sourcedocs/trainers.rst:42-55` | Pass: answer names the two List-valued RFSFTConfig arguments. |
| 34 | `sourcedocs/trainers.rst:65-97` | Pass: answer compares FSDP `offload_params` values in the examples. |
| 35 | `sourcedocs/trainers.rst:121-143` | Pass: answer identifies the List-valued DPO `beta` parameter. |
| 36 | `sourcedocs/trainers.rst:190-208` | Pass: answer lists RFGRPOConfig settings shown in the example. |
| 37 | `sourcedocs/generators.rst:23-28` | Pass: answer lists recommended RFvLLMModelConfig keys. |
| 38 | `sourcedocs/generators.rst:101-107` | Pass: answer lists RFOpenAIAPIModelConfig model-config knobs. |
| 39 | `sourcedocs/onlineagg.rst:51-54` | Pass: answer states how confidence intervals narrow. |
| 40 | `sourcedocs/onlineagg.rst:203-224` | Pass: answer states the batch vs RapidFire runtime comparison. |
