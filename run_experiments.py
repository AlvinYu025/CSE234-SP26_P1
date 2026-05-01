#!/usr/bin/env python3
"""Run retrieval experiments for Project 1.

Each experiment writes:
  experiments/<name>/config.json
  experiments/<name>/output.json
  experiments/<name>/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from main import DEFAULT_CONFIG, RagPipeline, read_questions, set_determinism, write_json

sys.path.insert(0, str(Path(__file__).resolve().parent / "Metrics"))
from project1_eval import f1_at_k, precision_at_k, recall_at_k, to_spans  # noqa: E402


ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "validation-set-golden-qa-pairs.json"
EXPERIMENTS_DIR = ROOT / "experiments"
LOG_PATH = ROOT / "logs" / "rapidfire.log"


EXPERIMENT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "c01_similarity_180_30_k3",
        "chunk_tokens": 180,
        "chunk_overlap": 30,
        "retriever": "similarity",
        "top_k": 3,
        "fetch_k": 8,
        "rerank": None,
        "rerank_top_n": 3,
        "ngram_range": [1, 2],
        "max_context_tokens": 1450,
    },
    {
        "name": "c02_similarity_220_45_k5",
        "chunk_tokens": 220,
        "chunk_overlap": 45,
        "retriever": "similarity",
        "top_k": 5,
        "fetch_k": 10,
        "rerank": None,
        "rerank_top_n": 5,
        "ngram_range": [1, 2],
        "max_context_tokens": 1550,
    },
    {
        "name": "c03_similarity_260_55_k5_lexical",
        "chunk_tokens": 260,
        "chunk_overlap": 55,
        "retriever": "similarity",
        "top_k": 5,
        "fetch_k": 12,
        "rerank": "lexical",
        "rerank_top_n": 5,
        "ngram_range": [1, 2],
        "max_context_tokens": 1650,
    },
    {
        "name": "c04_similarity_320_70_k5_lexical",
        "chunk_tokens": 320,
        "chunk_overlap": 70,
        "retriever": "similarity",
        "top_k": 5,
        "fetch_k": 12,
        "rerank": "lexical",
        "rerank_top_n": 5,
        "ngram_range": [1, 2],
        "max_context_tokens": 1700,
    },
    {
        "name": "c05_similarity_420_90_k5_lexical",
        "chunk_tokens": 420,
        "chunk_overlap": 90,
        "retriever": "similarity",
        "top_k": 5,
        "fetch_k": 12,
        "rerank": "lexical",
        "rerank_top_n": 5,
        "ngram_range": [1, 2],
        "max_context_tokens": 1750,
    },
    {
        "name": "c06_mmr_260_55_k5",
        "chunk_tokens": 260,
        "chunk_overlap": 55,
        "retriever": "mmr",
        "top_k": 5,
        "fetch_k": 14,
        "lambda_mult": 0.65,
        "rerank": None,
        "rerank_top_n": 5,
        "ngram_range": [1, 2],
        "max_context_tokens": 1650,
    },
    {
        "name": "c07_mmr_260_55_fetch20_lexical",
        "chunk_tokens": 260,
        "chunk_overlap": 55,
        "retriever": "mmr",
        "top_k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.72,
        "rerank": "lexical",
        "rerank_top_n": 5,
        "ngram_range": [1, 2],
        "max_context_tokens": 1650,
    },
    {
        "name": "c08_similarity_char_260_55_k5",
        "chunk_tokens": 260,
        "chunk_overlap": 55,
        "retriever": "similarity",
        "top_k": 5,
        "fetch_k": 12,
        "rerank": "lexical",
        "rerank_top_n": 5,
        "analyzer": "char_wb",
        "ngram_range": [3, 5],
        "max_context_tokens": 1650,
    },
    {
        "name": "c09_title_related_420_80_k1",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "retriever": "similarity",
        "top_k": 1,
        "fetch_k": 8,
        "rerank": "title_lexical",
        "rerank_top_n": 1,
        "analyzer": "word",
        "ngram_range": [1, 2],
        "max_context_tokens": 1850,
        "symbol_boost": 0.02,
        "title_boost": 0.12,
        "glossary_symbol_penalty": 0.02,
        "source_expand_lines": 160,
        "related_sources": True,
        "max_related_sources": 5,
    },
]


def load_validation(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_metrics(outputs: list[dict[str, Any]], validation: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    by_qid = {int(item["question_id"]): item for item in outputs}
    rows = []
    for item in validation:
        qid = int(item["question_id"])
        gt = to_spans(item["source_evidence"])
        retrieved = to_spans(by_qid[qid]["sources"]) if qid in by_qid else []
        rows.append(
            {
                "question_id": qid,
                f"F1@{k}": f1_at_k(retrieved, gt, k),
                f"Precision@{k}": precision_at_k(retrieved, gt, k),
                f"Recall@{k}": recall_at_k(retrieved, gt, k),
            }
        )
    n = len(rows)
    mean_f1 = sum(r[f"F1@{k}"] for r in rows) / n
    mean_p = sum(r[f"Precision@{k}"] for r in rows) / n
    mean_r = sum(r[f"Recall@{k}"] for r in rows) / n
    return {
        "meta": {"k": k, "n_questions": n},
        "summary": {
            f"F1@{k}": mean_f1,
            f"Precision@{k}": mean_p,
            f"Recall@{k}": mean_r,
            "Retrieval Score": (mean_f1 + mean_p + mean_r) / 3,
        },
        "per_question": rows,
    }


def run_one(exp_config: dict[str, Any], questions: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    name = exp_config["name"]
    run_dir = EXPERIMENTS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in exp_config.items() if k != "name"})
    if exp_config["name"] != "c09_title_related_420_80_k1":
        config["source_expand_lines"] = 0
        config["related_sources"] = False
    if isinstance(config.get("ngram_range"), list):
        config["ngram_range"] = tuple(config["ngram_range"])

    pipeline = RagPipeline(config)
    outputs = [pipeline.answer_question(q["question_id"], q["question"]) for q in questions]
    metrics = retrieval_metrics(outputs, validation)

    serializable_config = dict(config)
    serializable_config["ngram_range"] = list(serializable_config["ngram_range"])
    serializable_config["name"] = name
    write_json(run_dir / "config.json", serializable_config)
    write_json(run_dir / "output.json", outputs)
    write_json(run_dir / "metrics.json", metrics)
    return {"name": name, **metrics["summary"]}


def append_log(lines: list[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip() + "\n")


def main() -> int:
    set_determinism()
    validation = load_validation(VALIDATION)
    questions = read_questions(VALIDATION)
    append_log(["Project 1 deterministic retrieval experiment run"])
    summaries = []
    for exp in EXPERIMENT_CONFIGS:
        summary = run_one(exp, questions, validation)
        summaries.append(summary)
        append_log(
            [
                (
                    f"{summary['name']}: F1@5={summary['F1@5']:.4f} "
                    f"P@5={summary['Precision@5']:.4f} R@5={summary['Recall@5']:.4f} "
                    f"Retrieval Score={summary['Retrieval Score']:.4f}"
                )
            ]
        )
        print(summaries[-1])
    write_json(EXPERIMENTS_DIR / "summary_metrics.json", summaries)
    best = max(summaries, key=lambda row: (row["Retrieval Score"], row["F1@5"], row["Recall@5"]))
    append_log([f"Best retrieval config: {best['name']}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
