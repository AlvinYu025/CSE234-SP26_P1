# CSE/DSC 234 Project 1

## Final Pipeline

The final runnable pipeline is `main.py`:

```bash
python3 main.py --input <input_questions_json> --output <output_answers_json>
```

It loads the RapidFire documentation from `sourcedocs/`, builds an in-memory TF-IDF retrieval index, assembles source-aware context under the 2,000-token budget, and writes one answer object per question with:

- `question_id`
- `answer`
- `sources`
- `retrieved_context`

The final pipeline is deterministic and does not depend on replaying the RapidFire experiment sequence.

## RapidFire Experiments

`rapidfire_experiments.py` is the RapidFire AI experimentation layer. It follows the course tutorial pattern:

- `RFLangChainRagSpec` for document loading, chunking, OpenAI-compatible embeddings, FAISS vector store retrieval, search strategy, and reranking.
- `RFGridSearch` for multi-config expansion.
- `Experiment(..., mode="evals")` and `run_evals(...)` for systematic comparison.

The script defines 12 configurations from:

- 2 chunking strategies: 320/64 and 512/96
- 3 search strategies: similarity k=5, similarity k=8, MMR k=8/fetch_k=20
- 2 reranker depths: top_n=3 and top_n=5

Run it after installing the RapidFire dependencies in a Python 3.12 environment:

```bash
export OPENAI_API_KEY=<your_key>
export OPENAI_BASE_URL=https://tritonai-api.ucsd.edu/v1
export RAG_MODEL=api-gpt-oss-120b
python3 rapidfire_experiments.py --input data/golden_qa.json --experiments-path rapidfire_experiments --experiment-name project1-rapidfire-rag-mlflow --num-shards 4 --num-actors 4
```

The script starts a local MLflow server on RapidFire's default port if one is not already running, then shuts down the server it started when the experiment ends. RapidFire writes experiment artifacts under `rapidfire_experiments/`; the final run log is copied to `logs/rapidfire.log`. The final `main.py` can then be run separately on the released validation set or hidden test set.

## Data

- `data/golden_qa.json`: manually curated golden QA data in the same schema as the released validation set.
- `validation-set-golden-qa-pairs.json`: released validation/dev set.
- `sourcedocs/`: RapidFire documentation corpus.

## Dependencies

The final `main.py` uses:

- `numpy`
- `scikit-learn`
- `openai`

The RapidFire experiment driver additionally uses the packages listed in `requirements.txt`, including `rapidfireai`, LangChain packages, `datasets`, and `faiss-cpu`.
