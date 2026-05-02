#!/usr/bin/env python3
"""RapidFire AI experiment driver for CSE/DSC 234 Project 1.

This file is the experiment entry point. It runs both:

1. Dense RapidFire/LangChain RAG configs
   - RFLangChainRagSpec
   - OpenAI-compatible embeddings
   - FAISS vector store
   - similarity / MMR search
   - CrossEncoderReranker

2. TF-IDF extension configs
   - executed through RapidFire Experiment.run_evals
   - custom deterministic TF-IDF retriever defined in this file
   - title/API-symbol boost, related-source expansion, and conservative
     source-span expansion ablations

Metrics:
   This script uses the course Metrics/project1_eval.py primitives so retrieval
   and judge metrics match the provided evaluation CLIs:
   - F1_at_5
   - Precision_at_5
   - Recall_at_5
   - Retrieval Score
   - Correctness_pass_rate
   - Faithfulness_pass_rate
   - Completeness_normalized
   - Generation_Score_3_released
   - Partial_Overall_Score

Typical usage:

  export OPENAI_API_KEY="$(cat ~/api-key.txt)"
  export OPENAI_BASE_URL="https://tritonai-api.ucsd.edu/v1"
  export JUDGE_BASE_URL="https://tritonai-api.ucsd.edu/v1"
  export JUDGE_MODEL="claude-sonnet-4-6"
  export RAG_MODEL="api-gpt-oss-120b"

  # List configs
  python3 rapidfire_experiments.py --mode list-configs

  # Run one dense config
  python3 rapidfire_experiments.py --mode dense-single --dense-config-id D07 \
      --input data/golden_qa.json \
      --experiments-path rapidfire_experiments/dense_D07

  # Run all dense configs, each in its own folder
  python3 rapidfire_experiments.py --mode dense-all \
      --input data/golden_qa.json \
      --experiments-path rapidfire_experiments/dense_all

  # Run one TF-IDF config
  python3 rapidfire_experiments.py --mode tfidf-single --tfidf-config-id T11 \
      --input data/golden_qa.json \
      --experiments-path rapidfire_experiments/tfidf_T11

  # Run all TF-IDF configs
  python3 rapidfire_experiments.py --mode tfidf-all \
      --input data/golden_qa.json \
      --experiments-path rapidfire_experiments/tfidf_all

  # Run every config
  python3 rapidfire_experiments.py --mode all \
      --input data/golden_qa.json \
      --experiments-path rapidfire_experiments/all

Notes:
- Dense configs and TF-IDF configs are both executed with RapidFire's
  Experiment.run_evals.
- Dense configs use RFLangChainRagSpec directly.
- TF-IDF configs use a custom preprocessor to build retrieved context and spans,
  then still use RapidFire for generation, metrics, logging, and artifacts.
- Large source expansion is intentionally not used as a default. Source expansion
  is an ablation knob: 0, 5, 10, 20, and 50 stress-test lines.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sourcedocs"
SOURCE_ZIP = ROOT / "sourcedocs.zip"
METRICS_DIR = ROOT / "Metrics"
DEFAULT_INPUT = ROOT / "data" / "validation-set-golden-qa-pairs.json"
DEFAULT_EXPERIMENTS_PATH = ROOT / "rapidfire_experiments"

# Add Metrics/ to both sys.path (driver) and PYTHONPATH (Ray workers).
# sys.path covers the current process; PYTHONPATH is inherited by every
# subprocess Ray spawns, so `import project1_eval` works on workers too.
_metrics_str = str(METRICS_DIR)
if _metrics_str not in sys.path:
    sys.path.insert(0, _metrics_str)
os.environ["PYTHONPATH"] = (
    _metrics_str + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
)

from project1_eval import (  # noqa: E402
    call_judge,
    f1_at_k,
    precision_at_k,
    recall_at_k,
    to_spans,
)
from rapidfire_integration_example import (  # noqa: E402
    sample_accumulate_metrics_fn,
    sample_compute_metrics_fn,
)


# Ensure judge uses TritonAI endpoint (same gateway as RAG_MODEL).
# Can be overridden by exporting these vars before running.
if not os.environ.get("JUDGE_BASE_URL"):
    os.environ["JUDGE_BASE_URL"] = "https://tritonai-api.ucsd.edu/v1"
if not os.environ.get("JUDGE_MODEL"):
    os.environ["JUDGE_MODEL"] = "claude-sonnet-4-6"

INSTRUCTIONS = (
    "You answer questions about the RapidFire AI documentation. Use only the "
    "provided context. If the context does not support the answer, say that the "
    "documentation does not provide enough information. Keep the answer concise "
    "but include important API names, defaults, constraints, and contrasts when present."
)

QUERY_STOPWORDS = {
    "a", "ai", "an", "and", "api", "are", "argument", "arguments", "class",
    "config", "configs", "different", "difference", "do", "does", "for",
    "function", "how", "in", "is", "parameter", "parameters", "provided",
    "rapidfire", "rapidfireai", "support", "supported", "terms", "the", "they",
    "to", "tutorial", "tutorials", "type", "types", "use", "uses", "using",
    "what", "with", "you",
}


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

DENSE_CONFIGS: dict[str, dict[str, Any]] = {
    "D01": {
        "name": "dense_320_64_similarity_k5_rerank3",
        "chunk_size": 320,
        "chunk_overlap": 64,
        "search_cfg": {"type": "similarity", "k": 5},
        "reranker_top_n": 3,
    },
    "D02": {
        "name": "dense_320_64_similarity_k5_rerank5",
        "chunk_size": 320,
        "chunk_overlap": 64,
        "search_cfg": {"type": "similarity", "k": 5},
        "reranker_top_n": 5,
    },
    "D03": {
        "name": "dense_320_64_similarity_k8_rerank3",
        "chunk_size": 320,
        "chunk_overlap": 64,
        "search_cfg": {"type": "similarity", "k": 8},
        "reranker_top_n": 3,
    },
    "D04": {
        "name": "dense_320_64_similarity_k8_rerank5",
        "chunk_size": 320,
        "chunk_overlap": 64,
        "search_cfg": {"type": "similarity", "k": 8},
        "reranker_top_n": 5,
    },
    "D05": {
        "name": "dense_320_64_mmr_k8_fetch20_lam07_rerank3",
        "chunk_size": 320,
        "chunk_overlap": 64,
        "search_cfg": {"type": "mmr", "k": 8, "fetch_k": 20, "lambda_mult": 0.7},
        "reranker_top_n": 3,
    },
    "D06": {
        "name": "dense_320_64_mmr_k8_fetch20_lam07_rerank5",
        "chunk_size": 320,
        "chunk_overlap": 64,
        "search_cfg": {"type": "mmr", "k": 8, "fetch_k": 20, "lambda_mult": 0.7},
        "reranker_top_n": 5,
    },
    "D07": {
        "name": "dense_512_96_similarity_k5_rerank3",
        "chunk_size": 512,
        "chunk_overlap": 96,
        "search_cfg": {"type": "similarity", "k": 5},
        "reranker_top_n": 3,
    },
    "D08": {
        "name": "dense_512_96_similarity_k5_rerank5",
        "chunk_size": 512,
        "chunk_overlap": 96,
        "search_cfg": {"type": "similarity", "k": 5},
        "reranker_top_n": 5,
    },
    "D09": {
        "name": "dense_512_96_similarity_k8_rerank3",
        "chunk_size": 512,
        "chunk_overlap": 96,
        "search_cfg": {"type": "similarity", "k": 8},
        "reranker_top_n": 3,
    },
    "D10": {
        "name": "dense_512_96_similarity_k8_rerank5",
        "chunk_size": 512,
        "chunk_overlap": 96,
        "search_cfg": {"type": "similarity", "k": 8},
        "reranker_top_n": 5,
    },
    "D11": {
        "name": "dense_512_96_mmr_k8_fetch20_lam07_rerank3",
        "chunk_size": 512,
        "chunk_overlap": 96,
        "search_cfg": {"type": "mmr", "k": 8, "fetch_k": 20, "lambda_mult": 0.7},
        "reranker_top_n": 3,
    },
    "D12": {
        "name": "dense_512_96_mmr_k8_fetch20_lam07_rerank5",
        "chunk_size": 512,
        "chunk_overlap": 96,
        "search_cfg": {"type": "mmr", "k": 8, "fetch_k": 20, "lambda_mult": 0.7},
        "reranker_top_n": 5,
    },
}


TFIDF_CONFIGS: dict[str, dict[str, Any]] = {
    # Chunk-size ablation, strict source span, no title/related expansion.
    "T01": {
        "name": "tfidf_320_64_k1_no_title_no_related_expand0",
        "chunk_tokens": 320,
        "chunk_overlap": 64,
        "top_k": 1,
        "use_title_boost": False,
        "related_sources": False,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },
    "T02": {
        "name": "tfidf_420_80_k1_no_title_no_related_expand0",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "top_k": 1,
        "use_title_boost": False,
        "related_sources": False,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },
    "T03": {
        "name": "tfidf_512_96_k1_no_title_no_related_expand0",
        "chunk_tokens": 512,
        "chunk_overlap": 96,
        "top_k": 1,
        "use_title_boost": False,
        "related_sources": False,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },

    # Title/API-symbol boost ablation.
    "T04": {
        "name": "tfidf_320_64_k1_title_no_related_expand0",
        "chunk_tokens": 320,
        "chunk_overlap": 64,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": False,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },
    "T05": {
        "name": "tfidf_420_80_k1_title_no_related_expand0",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": False,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },
    "T06": {
        "name": "tfidf_512_96_k1_title_no_related_expand0",
        "chunk_tokens": 512,
        "chunk_overlap": 96,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": False,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },

    # Related-source expansion ablation, still with strict source spans.
    "T07": {
        "name": "tfidf_320_64_k1_title_related_expand0",
        "chunk_tokens": 320,
        "chunk_overlap": 64,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": True,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },
    "T08": {
        "name": "tfidf_420_80_k1_title_related_expand0",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": True,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },
    "T09": {
        "name": "tfidf_512_96_k1_title_related_expand0",
        "chunk_tokens": 512,
        "chunk_overlap": 96,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": True,
        "source_expand_lines": 0,
        "context_expand_lines": 20,
    },

    # Conservative source expansion ablation.
    "T10": {
        "name": "tfidf_420_80_k1_title_related_expand5",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": True,
        "source_expand_lines": 5,
        "context_expand_lines": 20,
    },
    "T11": {
        "name": "tfidf_420_80_k1_title_related_expand10",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "top_k": 1,
        "use_title_boost": True,
        "related_sources": True,
        "source_expand_lines": 10,
        "context_expand_lines": 20,
    },

    # Top-k ablation under the strongest conservative lexical setting.
    "T12": {
        "name": "tfidf_420_80_k3_title_related_expand5",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "top_k": 3,
        "use_title_boost": True,
        "related_sources": True,
        "source_expand_lines": 5,
        "context_expand_lines": 20,
    },
    "T13": {
        "name": "tfidf_420_80_k5_title_related_expand5",
        "chunk_tokens": 420,
        "chunk_overlap": 80,
        "top_k": 5,
        "use_title_boost": True,
        "related_sources": True,
        "source_expand_lines": 5,
        "context_expand_lines": 20,
    },

    # # Wider expansion stress tests. Interpret carefully; do not choose solely
    # # because they inflate overlap-based retrieval.
    # "T14": {
    #     "name": "tfidf_420_80_k1_title_related_expand20_stress",
    #     "chunk_tokens": 420,
    #     "chunk_overlap": 80,
    #     "top_k": 1,
    #     "use_title_boost": True,
    #     "related_sources": True,
    #     "source_expand_lines": 20,
    #     "context_expand_lines": 20,
    # },
    # "T15": {
    #     "name": "tfidf_420_80_k1_title_related_expand50_stress",
    #     "chunk_tokens": 420,
    #     "chunk_overlap": 80,
    #     "top_k": 1,
    #     "use_title_boost": True,
    #     "related_sources": True,
    #     "source_expand_lines": 50,
    #     "context_expand_lines": 20,
    # },
}


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text.split()) * 1.3))


def ensure_corpus() -> None:
    if SOURCE_DIR.exists() and any(SOURCE_DIR.glob("*.rst")):
        return
    if not SOURCE_ZIP.exists():
        raise FileNotFoundError("Missing sourcedocs/ and sourcedocs.zip")
    subprocess.run(["unzip", "-q", "-o", str(SOURCE_ZIP), "-d", str(ROOT)], check=True)


def clean_rst(text: str) -> str:
    text = re.sub(r"^\s*\.\. image::.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r":(?:class|func|code|doc|param|type|py:[^:`]+):`([^`]+)`", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def singularize(term: str) -> str:
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def lexical_terms(text: str) -> set[str]:
    return {
        singularize(term)
        for term in re.findall(r"[a-z0-9_+-]+", text.lower())
        if len(term) > 1 and term not in QUERY_STOPWORDS
    }


def query_symbols(question: str) -> set[str]:
    return {
        symbol.lower()
        for symbol in re.findall(
            r"\b(?:RF[A-Za-z0-9_]+|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z]{2,}[A-Za-z0-9_]*)\b",
            question,
        )
        if len(symbol) > 2
    }


def first_document_title(lines: list[str], fallback: str) -> str:
    for i, line in enumerate(lines[:-1]):
        title = line.strip()
        underline = lines[i + 1].strip()
        if title and len(underline) >= 3 and set(underline) <= set("=-~^#*"):
            return title
    return fallback


def section_for_line(lines: list[str], idx: int, current: str) -> str:
    line = lines[idx].strip()
    if idx + 1 < len(lines):
        underline = lines[idx + 1].strip()
        if line and len(underline) >= max(3, len(line) // 2) and set(underline) <= set("=-~^#*"):
            return line
    return current


def load_rst_documents() -> list[dict[str, Any]]:
    ensure_corpus()
    docs: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DIR.glob("*.rst")):
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        docs.append({"file": path.name, "path": path, "text": raw_text, "lines": raw_text.splitlines()})
    if not docs:
        raise RuntimeError("No .rst files found in sourcedocs/")
    return docs


def load_qa(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("QA input must be a list of examples")
    rows = []
    for item in data:
        rows.append(
            {
                "question_id": int(item["question_id"]),
                "question": item["question"],
                "query": item["question"],
                "reference_answer": item.get("reference_answer", item.get("gold_answer", "")),
                "source_evidence": item.get("source_evidence", item.get("sources", [])),
            }
        )
    return rows


def source_file_from_doc(doc: Any) -> str:
    source = doc.metadata.get("source") or doc.metadata.get("file") or ""
    return Path(str(source)).name


def document_template(doc: Any) -> str:
    filename = source_file_from_doc(doc)
    return f"file={filename}\n{doc.page_content}"


def cap_context_by_estimated_tokens(text: str, max_estimated_tokens: int = 1850) -> str:
    if estimate_tokens(text) <= max_estimated_tokens:
        return text
    max_words = int(max_estimated_tokens / 1.3)
    return " ".join(text.split()[:max_words])


def safe_to_spans(items: Any) -> list[tuple[str, int, int]]:
    try:
        return to_spans(items)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Metric functions using Metrics/project1_eval.py
# ---------------------------------------------------------------------------

def project_compute_metrics_fn(batch: dict[str, list]) -> dict[str, dict[str, Any]]:
    """Compute retrieval + released generation metrics.

    Delegates retrieval and judge computation to
    Metrics/rapidfire_integration_example.py::sample_compute_metrics_fn,
    then adds Partial_Overall_Score and context-token diagnostics on top.
    """
    total = len(batch["query"])
    if total == 0:
        return {"Total": {"value": 0}}

    # Reuse sample_compute_metrics_fn from the course integration module.
    metrics = sample_compute_metrics_fn(batch)

    # sample_compute_metrics_fn does not compute Partial_Overall_Score.
    if "Retrieval Score" in metrics and "Generation_Score_3_released" in metrics:
        rs = metrics["Retrieval Score"]["value"]
        gen3 = metrics["Generation_Score_3_released"]["value"]
        metrics["Partial_Overall_Score"] = {"value": 0.5 * rs + 0.5 * gen3}

    # Context-token diagnostics (not in sample_compute_metrics_fn).
    context_tokens = [
        float(estimate_tokens(str(x)))
        for x in batch.get("serialized_context", [""] * total)
    ]
    metrics["MeanEstimatedContextTokens"] = {"value": sum(context_tokens) / total}
    metrics["MaxEstimatedContextTokens"] = {"value": max(context_tokens) if context_tokens else 0.0}

    return metrics


def project_accumulate_metrics_fn(aggregated: dict[str, list]) -> dict[str, dict[str, Any]]:
    """Weighted accumulation over batches.

    Delegates the standard metrics to
    Metrics/rapidfire_integration_example.py::sample_accumulate_metrics_fn,
    then adds Partial_Overall_Score and context-token diagnostics.
    """
    out = sample_accumulate_metrics_fn(aggregated)

    # sample_accumulate_metrics_fn omits is_distributive on Total and
    # is_distributive on Judge Failures; restore them for RapidFire internals.
    out["Total"]["is_distributive"] = True
    if "Judge Failures" in out:
        out["Judge Failures"]["is_distributive"] = True

    ns = [m.get("value", 0) for m in aggregated.get("Total", [])]
    total = sum(ns)

    # Metrics not covered by sample_accumulate_metrics_fn.
    extra: dict[str, tuple[int, int]] = {
        "Partial_Overall_Score": (0, 1),
        "MeanEstimatedContextTokens": (0, 2500),
        "MaxEstimatedContextTokens": (0, 2500),
    }
    for metric, value_range in extra.items():
        if metric not in aggregated:
            continue
        out[metric] = {
            "value": (
                sum(m["value"] * n for m, n in zip(aggregated[metric], ns)) / total
                if total else 0.0
            ),
            "is_algebraic": True,
            "value_range": value_range,
        }

    return out


# ---------------------------------------------------------------------------
# RapidFire dependencies
# ---------------------------------------------------------------------------

def require_rapidfire_dependencies() -> dict[str, Any]:
    missing: list[str] = []
    modules = [
        "rapidfireai",
        "datasets",
        "langchain_community",
        "langchain_text_splitters",
        "langchain_openai",
        "langchain_classic",
    ]
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        raise RuntimeError(
            "Missing RapidFire experiment dependencies: "
            + ", ".join(missing)
            + "\nInstall them in the project environment before running experiments."
        )

    # RapidFireAI 0.15.x may gate RFOpenAIAPIModelConfig behind vLLM imports
    # even when only using OpenAI-compatible API models. Avoid installing vLLM
    # for this CPU/API-only project by providing the minimal symbol if absent.
    try:
        __import__("vllm")
    except ImportError:
        vllm_stub = types.ModuleType("vllm")

        class SamplingParams:  # pragma: no cover
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        vllm_stub.SamplingParams = SamplingParams
        sys.modules["vllm"] = vllm_stub

    from datasets import Dataset
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rapidfireai import Experiment
    from rapidfireai.automl import RFGridSearch, RFLangChainRagSpec, RFOpenAIAPIModelConfig

    return {
        "Dataset": Dataset,
        "CrossEncoderReranker": CrossEncoderReranker,
        "DirectoryLoader": DirectoryLoader,
        "TextLoader": TextLoader,
        "OpenAIEmbeddings": OpenAIEmbeddings,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "Experiment": Experiment,
        "RFGridSearch": RFGridSearch,
        "RFLangChainRagSpec": RFLangChainRagSpec,
        "RFOpenAIAPIModelConfig": RFOpenAIAPIModelConfig,
    }


def make_openai_config(
    deps: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    rag: Any | None,
) -> Any:
    RFOpenAIAPIModelConfig = deps["RFOpenAIAPIModelConfig"]
    kwargs = {
        "client_config": {"api_key": api_key, "base_url": base_url, "max_retries": 2},
        "model_config": {"model": model, "max_completion_tokens": 512, "temperature": 0.0},
        "rpm_limit": 120,
        "tpm_limit": 1_000_000,
        "prompt_manager": None,
    }
    sig = inspect.signature(RFOpenAIAPIModelConfig)
    if "rag" in sig.parameters:
        kwargs["rag"] = rag
    elif rag is not None:
        raise RuntimeError("This RapidFire version does not expose RFOpenAIAPIModelConfig(rag=...).")
    return RFOpenAIAPIModelConfig(**kwargs)


# ---------------------------------------------------------------------------
# Dense RapidFire configs
# ---------------------------------------------------------------------------

def _char_to_line(text: str, char_index: int) -> int:
    if char_index <= 0:
        return 1
    return text[:char_index].count("\n") + 1


_DENSE_DOC_CACHE: dict[str, dict[str, Any]] | None = None


def dense_doc_cache() -> dict[str, dict[str, Any]]:
    global _DENSE_DOC_CACHE
    if _DENSE_DOC_CACHE is None:
        _DENSE_DOC_CACHE = {doc["file"]: doc for doc in load_rst_documents()}
    return _DENSE_DOC_CACHE


def infer_span_from_langchain_doc(doc: Any) -> tuple[str, int, int]:
    filename = source_file_from_doc(doc)
    cache = dense_doc_cache()
    source_doc = cache.get(filename)
    if not source_doc:
        return (filename, 1, 1)

    text = source_doc["text"]
    lines = source_doc["lines"]
    page_content = str(getattr(doc, "page_content", "") or "")
    metadata = getattr(doc, "metadata", {}) or {}

    # RecursiveCharacterTextSplitter(add_start_index=True) should provide this.
    start_index = metadata.get("start_index")
    if isinstance(start_index, int) and start_index >= 0:
        start_line = _char_to_line(text, start_index)
        end_line = _char_to_line(text, min(len(text), start_index + len(page_content)))
        return (filename, max(1, start_line), max(start_line, end_line))

    # Fallback: exact substring search.
    idx = text.find(page_content)
    if idx >= 0:
        start_line = _char_to_line(text, idx)
        end_line = _char_to_line(text, min(len(text), idx + len(page_content)))
        return (filename, max(1, start_line), max(start_line, end_line))

    # Fallback: match first non-empty line.
    nonempty = [line.strip() for line in page_content.splitlines() if line.strip()]
    if nonempty:
        needle = nonempty[0]
        for i, line in enumerate(lines, start=1):
            if needle in line.strip() or line.strip() in needle:
                approx_len = max(1, len(page_content.splitlines()))
                return (filename, i, min(len(lines), i + approx_len - 1))

    return (filename, 1, min(len(lines), 1))


def preprocess_dense_fn(batch: dict[str, list], rag: Any, prompt_manager: Any) -> dict[str, list]:
    all_docs = rag.get_context(batch_queries=batch["question"], serialize=False)
    serialized_context = [
        cap_context_by_estimated_tokens(text, max_estimated_tokens=1850)
        for text in rag.serialize_documents(all_docs)
    ]

    retrieved_spans = [
        [infer_span_from_langchain_doc(doc) for doc in docs]
        for docs in all_docs
    ]
    ground_truth_spans = [safe_to_spans(evidence) for evidence in batch["source_evidence"]]

    prompts = [
        [
            {"role": "system", "content": INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    f"{context}\n\n"
                    "Question:\n"
                    f"{question}\n\n"
                    "Answer using only the context."
                ),
            },
        ]
        for question, context in zip(batch["question"], serialized_context)
    ]

    return {
        "prompts": prompts,
        "query": batch["question"],
        "serialized_context": serialized_context,
        "retrieved_context": serialized_context,
        "retrieved_spans": retrieved_spans,
        "ground_truth_spans": ground_truth_spans,
        **batch,
    }


def postprocess_fn(batch: dict[str, list]) -> dict[str, list]:
    batch["answer"] = [str(text).strip() for text in batch.get("generated_text", [])]
    return batch


def build_dense_config_group(
    deps: dict[str, Any],
    dense_config: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
) -> Any:
    DirectoryLoader = deps["DirectoryLoader"]
    TextLoader = deps["TextLoader"]
    OpenAIEmbeddings = deps["OpenAIEmbeddings"]
    RecursiveCharacterTextSplitter = deps["RecursiveCharacterTextSplitter"]
    CrossEncoderReranker = deps["CrossEncoderReranker"]
    RFLangChainRagSpec = deps["RFLangChainRagSpec"]
    RFGridSearch = deps["RFGridSearch"]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2",
        chunk_size=int(dense_config["chunk_size"]),
        chunk_overlap=int(dense_config["chunk_overlap"]),
        add_start_index=True,
    )

    rag_spec = RFLangChainRagSpec(
        document_loader=DirectoryLoader(
            path=str(SOURCE_DIR),
            glob="*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            sample_seed=42,
        ),
        text_splitter=splitter,
        embedding_cfg={
            "class": OpenAIEmbeddings,
            "model": "api-tgpt-embeddings",
            "api_key": api_key,
            "base_url": base_url,
            "check_embedding_ctx_length": False,
        },
        vector_store_cfg={"type": "faiss", "batch_size": batch_size},
        search_cfg=dense_config["search_cfg"],
        reranker_cfg={
            "class": CrossEncoderReranker,
            "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"},
            "top_n": int(dense_config["reranker_top_n"]),
        },
        enable_gpu_search=False,
        document_template=document_template,
    )

    openai_config = make_openai_config(
        deps=deps,
        api_key=api_key,
        base_url=base_url,
        model=model,
        rag=rag_spec,
    )

    return RFGridSearch(
        {
            "openai_config": openai_config,
            "batch_size": batch_size,
            "preprocess_fn": preprocess_dense_fn,
            "postprocess_fn": postprocess_fn,
            "compute_metrics_fn": project_compute_metrics_fn,
            "accumulate_metrics_fn": project_accumulate_metrics_fn,
            "online_strategy_kwargs": {
                "strategy_name": "normal",
                "confidence_level": 0.95,
                "use_fpc": True,
            },
        }
    )


# ---------------------------------------------------------------------------
# TF-IDF configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file: str
    start_line: int
    end_line: int
    section: str
    text: str

    def source(self, expand_lines: int = 0, max_line: int | None = None) -> tuple[str, int, int]:
        start = max(1, self.start_line - expand_lines)
        end = self.end_line + expand_lines
        if max_line is not None:
            end = min(max_line, end)
        return (self.file, start, end)


def build_chunks(docs: list[dict[str, Any]], chunk_tokens: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        filename = doc["file"]
        lines = doc["lines"]
        section = filename
        units: list[tuple[int, str, str]] = []
        for i, line in enumerate(lines):
            section = section_for_line(lines, i, section)
            stripped = line.rstrip()
            if stripped:
                units.append((i + 1, stripped, section))

        start = 0
        while start < len(units):
            token_count = 0
            end = start
            while end < len(units):
                token_count += max(1, len(units[end][1].split()))
                end += 1
                if token_count >= chunk_tokens:
                    break

            block = units[start:end]
            if not block:
                break

            text = clean_rst("\n".join(item[1] for item in block))
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{filename}:{block[0][0]}-{block[-1][0]}",
                        file=filename,
                        start_line=block[0][0],
                        end_line=block[-1][0],
                        section=block[0][2],
                        text=text,
                    )
                )

            if end >= len(units):
                break

            overlap_words = 0
            next_start = end
            while next_start > start and overlap_words < chunk_overlap:
                next_start -= 1
                overlap_words += max(1, len(units[next_start][1].split()))
            start = max(start + 1, next_start)
    return chunks


class TfidfRag:
    def __init__(self, config: dict[str, Any]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.cosine_similarity = cosine_similarity
        self.config = config
        self.docs = load_rst_documents()
        self.doc_lines = {doc["file"]: doc["lines"] for doc in self.docs}
        self.line_counts = {doc["file"]: len(doc["lines"]) for doc in self.docs}
        self.doc_titles = {
            doc["file"]: first_document_title(doc["lines"], Path(doc["file"]).stem)
            for doc in self.docs
        }
        self.chunks = build_chunks(
            self.docs,
            chunk_tokens=int(config["chunk_tokens"]),
            chunk_overlap=int(config["chunk_overlap"]),
        )
        self.title_terms = {
            chunk.chunk_id: lexical_terms(
                f"{chunk.section} {Path(chunk.file).stem} {self.doc_titles.get(chunk.file, '')}"
            )
            for chunk in self.chunks
        }
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_df=0.92,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self._chunk_text(c) for c in self.chunks)

    @staticmethod
    def _chunk_text(chunk: Chunk) -> str:
        return f"{chunk.file} {chunk.section}\n{chunk.text}"

    def _query_scores(self, question: str) -> Any:
        qv = self.vectorizer.transform([question])
        return self.cosine_similarity(qv, self.matrix).ravel()

    def _title_boost(self, question: str, chunk: Chunk) -> float:
        if not self.config.get("use_title_boost", False):
            return 0.0
        q_terms = lexical_terms(question)
        if not q_terms:
            return 0.0

        boost = 0.0
        symbols = query_symbols(question)
        if symbols:
            haystack = f"{chunk.file} {chunk.section} {chunk.text}".lower()
            boost += 0.02 * sum(1 for symbol in symbols if symbol in haystack)
            if chunk.file == "glossary.rst":
                boost -= 0.02

        title_overlap = len(q_terms & self.title_terms.get(chunk.chunk_id, set()))
        if title_overlap:
            boost += 0.12 * title_overlap / len(q_terms)

        return boost

    def _chunk_covering_line(self, filename: str, line: int) -> Chunk | None:
        candidates = [
            chunk for chunk in self.chunks
            if chunk.file == filename and chunk.start_line <= line <= chunk.end_line
        ]
        if candidates:
            return sorted(candidates, key=lambda chunk: (chunk.end_line - chunk.start_line, chunk.chunk_id))[0]
        same_file = [chunk for chunk in self.chunks if chunk.file == filename]
        if not same_file:
            return None
        return sorted(same_file, key=lambda chunk: (abs(chunk.start_line - line), chunk.chunk_id))[0]

    def _select_related(self, question: str, ranked: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        if not ranked:
            return []
        if not self.config.get("related_sources", False):
            return ranked[: int(self.config["top_k"])]

        q_lower = question.lower()
        selected: list[tuple[Chunk, float]] = []
        seen_files: set[str] = set()
        max_sources = max(1, min(5, int(self.config.get("max_related_sources", 5))))

        def add(chunk: Chunk, score: float) -> None:
            if len(selected) >= max_sources or chunk.file in seen_files:
                return
            seen_files.add(chunk.file)
            selected.append((chunk, score))

        add(*ranked[0])

        if "tutorial" in q_lower and "use case" in q_lower:
            for chunk, score in ranked:
                if chunk.file == "tutorials.rst":
                    add(chunk, score)
                    break
            if "rag" in q_lower or "context engineering" in q_lower:
                for filename in ("rag_fiqa.rst", "rag_gsm8k.rst", "rag_scifact.rst"):
                    child = self._chunk_covering_line(filename, 8)
                    if child is not None:
                        add(child, ranked[0][1])
            elif "fine" in q_lower or "post-training" in q_lower or "post training" in q_lower:
                for filename in ("sft.rst", "dpo.rst", "grpo.rst"):
                    child = self._chunk_covering_line(filename, 8)
                    if child is not None:
                        add(child, ranked[0][1])

        if "vector store" in q_lower:
            for chunk, score in ranked:
                if chunk.file in {"ragspecs.rst", "rag_fiqa.rst"}:
                    add(chunk, score)

        if "why" in q_lower or "compared to" in q_lower or "compared with" in q_lower:
            for chunk, score in ranked:
                if chunk.file == "difference.rst":
                    add(chunk, score)
                    break

        if "terminology" in q_lower or re.search(r"\bwhat is an?\s+['\"]", q_lower):
            for chunk, score in ranked:
                if chunk.file == "glossary.rst":
                    add(chunk, score)
                    break

        return selected[:max_sources]

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        scores = self._query_scores(question)
        ranked: list[tuple[Chunk, float]] = []
        for idx, base_score in enumerate(scores):
            chunk = self.chunks[idx]
            score = float(base_score) + self._title_boost(question, chunk)
            ranked.append((chunk, score))
        ranked.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return self._select_related(question, ranked)

    def _context_text(self, chunk: Chunk) -> tuple[int, int, str]:
        expand_lines = int(self.config.get("context_expand_lines", 20))
        start = max(1, chunk.start_line - expand_lines)
        end = min(self.line_counts.get(chunk.file, chunk.end_line), chunk.end_line + expand_lines)
        lines = self.doc_lines.get(chunk.file)
        if not lines:
            return chunk.start_line, chunk.end_line, chunk.text

        core = "\n".join(line.rstrip() for line in lines[chunk.start_line - 1 : chunk.end_line] if line.rstrip())
        after = "\n".join(line.rstrip() for line in lines[chunk.end_line : end] if line.rstrip())
        before = "\n".join(line.rstrip() for line in lines[start - 1 : chunk.start_line - 1] if line.rstrip())
        ordered_parts = [
            f"Core lines {chunk.start_line}-{chunk.end_line}:\n{core}",
            f"Following context lines {chunk.end_line + 1}-{end}:\n{after}" if after else "",
            f"Preceding context lines {start}-{chunk.start_line - 1}:\n{before}" if before else "",
        ]
        return start, end, clean_rst("\n\n".join(part for part in ordered_parts if part))

    def build_context(self, retrieved: list[tuple[Chunk, float]], max_context_tokens: int = 1850) -> tuple[str, list[tuple[str, int, int]]]:
        used = estimate_tokens(INSTRUCTIONS)
        parts: list[str] = []
        retrieved_spans: list[tuple[str, int, int]] = []
        source_expand = int(self.config.get("source_expand_lines", 0))

        for rank, (chunk, score) in enumerate(retrieved, start=1):
            context_start, context_end, context_text = self._context_text(chunk)
            label = (
                f"[S{rank}] file={chunk.file} lines={context_start}-{context_end} "
                f"section={chunk.section} score={score:.4f}"
            )
            candidate = f"{label}\n{context_text}"
            cost = estimate_tokens(candidate)

            if used + cost > max_context_tokens:
                remaining_words = max(0, int((max_context_tokens - used) / 1.3) - 8)
                if remaining_words > 40:
                    shortened = " ".join(context_text.split()[:remaining_words])
                    candidate = f"{label}\n{shortened}"
                    parts.append(candidate)
                    retrieved_spans.append(
                        chunk.source(
                            expand_lines=source_expand,
                            max_line=self.line_counts.get(chunk.file),
                        )
                    )
                break

            parts.append(candidate)
            retrieved_spans.append(
                chunk.source(
                    expand_lines=source_expand,
                    max_line=self.line_counts.get(chunk.file),
                )
            )
            used += cost

        return "\n\n".join(parts), retrieved_spans

    def answer_inputs(self, questions: list[str]) -> tuple[list[list[tuple[str, int, int]]], list[str]]:
        all_spans: list[list[tuple[str, int, int]]] = []
        all_contexts: list[str] = []
        for question in questions:
            retrieved = self.retrieve(question)
            context, spans = self.build_context(retrieved)
            all_contexts.append(context)
            all_spans.append(spans)
        return all_spans, all_contexts


_PIPELINE_CACHE: dict[str, TfidfRag] = {}


class TfidfPreprocessor:
    """Picklable callable preprocessor for RapidFire single-config TF-IDF runs."""

    def __init__(self, config_id: str, config: dict[str, Any]) -> None:
        self.config_id = config_id
        self.config = dict(config)

    def _pipeline(self) -> TfidfRag:
        cache_key = json.dumps(self.config, sort_keys=True)
        if cache_key not in _PIPELINE_CACHE:
            _PIPELINE_CACHE[cache_key] = TfidfRag(self.config)
        return _PIPELINE_CACHE[cache_key]

    def __call__(self, batch: dict[str, list], rag: Any = None, prompt_manager: Any = None) -> dict[str, list]:
        pipeline = self._pipeline()
        retrieved_spans, contexts = pipeline.answer_inputs(batch["question"])
        ground_truth_spans = [safe_to_spans(evidence) for evidence in batch["source_evidence"]]

        prompts = [
            [
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        "Context:\n"
                        f"{context}\n\n"
                        "Question:\n"
                        f"{question}\n\n"
                        "Answer using only the context. If the question asks for a list, comparison, "
                        "API behavior, default, or constraint, include all supported details from the context."
                    ),
                },
            ]
            for question, context in zip(batch["question"], contexts)
        ]

        return {
            "prompts": prompts,
            "query": batch["question"],
            "serialized_context": contexts,
            "retrieved_context": contexts,
            "retrieved_spans": retrieved_spans,
            "ground_truth_spans": ground_truth_spans,
            "config_id": [self.config_id] * len(batch["question"]),
            "config_name": [self.config["name"]] * len(batch["question"]),
            **batch,
        }


def build_tfidf_config_group(
    deps: dict[str, Any],
    config_id: str,
    config: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
) -> Any:
    RFGridSearch = deps["RFGridSearch"]
    openai_config = make_openai_config(
        deps=deps,
        api_key=api_key,
        base_url=base_url,
        model=model,
        rag=None,
    )

    return RFGridSearch(
        {
            "openai_config": openai_config,
            "batch_size": batch_size,
            "preprocess_fn": TfidfPreprocessor(config_id, config),
            "postprocess_fn": postprocess_fn,
            "compute_metrics_fn": project_compute_metrics_fn,
            "accumulate_metrics_fn": project_accumulate_metrics_fn,
            "online_strategy_kwargs": {
                "strategy_name": "normal",
                "confidence_level": 0.95,
                "use_fpc": True,
            },
        }
    )


# ---------------------------------------------------------------------------
# MLflow and artifact helpers
# ---------------------------------------------------------------------------

def mlflow_health_url() -> str:
    host = os.environ.get("RF_MLFLOW_HOST", "127.0.0.1")
    port = os.environ.get("RF_MLFLOW_PORT", "8852")
    return f"http://{host}:{port}/health"


def mlflow_is_healthy() -> bool:
    try:
        with urllib.request.urlopen(mlflow_health_url(), timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def start_mlflow_if_needed(experiments_path: Path) -> subprocess.Popen[str] | None:
    if mlflow_is_healthy():
        return None

    host = os.environ.get("RF_MLFLOW_HOST", "127.0.0.1")
    port = os.environ.get("RF_MLFLOW_PORT", "8852")
    rf_home = Path.home() / "rapidfireai"
    artifact_root = rf_home / "mlflow_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiments_path.mkdir(parents=True, exist_ok=True)

    log_path = experiments_path / "mlflow_server.log"
    log_file = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            host,
            "--port",
            port,
            "--backend-store-uri",
            f"sqlite:///{rf_home / 'mlflow.db'}",
            "--default-artifact-root",
            str(artifact_root),
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._rapidfire_log_file = log_file  # type: ignore[attr-defined]

    for _ in range(30):
        if mlflow_is_healthy():
            return process
        if process.poll() is not None:
            log_file.close()
            raise RuntimeError(f"MLflow server exited early; see {log_path}")
        time.sleep(1)

    process.terminate()
    log_file.close()
    raise RuntimeError(f"MLflow server did not become healthy; see {log_path}")


def stop_started_mlflow(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    log_file = getattr(process, "_rapidfire_log_file", None)
    if log_file is not None:
        log_file.close()


def write_optional_frame(frame: Any, path: Path) -> None:
    if frame is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(frame, "to_csv"):
        frame.to_csv(path, index=False)
    else:
        path.write_text(str(frame), encoding="utf-8")


def try_write_experiment_frame(experiment: Any, method_name: str, path: Path) -> None:
    try:
        frame = getattr(experiment, method_name)()
    except Exception as exc:
        path.with_suffix(path.suffix + ".unavailable.txt").write_text(
            f"{method_name} unavailable for this RapidFire run: {exc}\n",
            encoding="utf-8",
        )
        return
    write_optional_frame(frame, path)


# ---------------------------------------------------------------------------
# Output JSON helpers
# ---------------------------------------------------------------------------

def load_run_results(experiments_path: Path) -> dict[str, list] | None:
    """Return the content of run_evals_return.json, or None if unavailable."""
    try:
        return json.loads(
            (experiments_path / "run_evals_return.json").read_text(encoding="utf-8")
        )
    except Exception:
        return None


def compute_retrieval_score_from_batches(batches: list[dict[str, Any]]) -> float:
    """Mean retrieval score (F1+P+R)/3 computed from per-question spans."""
    f1s: list[float] = []
    ps: list[float] = []
    rs: list[float] = []
    for batch in batches:
        for retrieved_raw, gt_raw in zip(
            batch.get("retrieved_spans", []),
            batch.get("ground_truth_spans", []),
        ):
            retrieved = [tuple(s) for s in (retrieved_raw or [])]
            gt = [tuple(s) for s in (gt_raw or [])]
            if not gt:
                continue
            f1s.append(f1_at_k(retrieved, gt, 5))
            ps.append(precision_at_k(retrieved, gt, 5))
            rs.append(recall_at_k(retrieved, gt, 5))
    n = len(f1s)
    return (sum(f1s) / n + sum(ps) / n + sum(rs) / n) / 3.0 if n else 0.0


def batches_to_output_list(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert RapidFire batch results to submission JSON format."""
    outputs: dict[Any, dict[str, Any]] = {}
    for batch in batches:
        n = len(batch.get("question_id", []))
        for i in range(n):
            qid = batch["question_id"][i]
            spans = (batch.get("retrieved_spans") or [[]] * n)[i]
            sources = [
                {"file": s[0], "lines": [int(s[1]), int(s[2])]}
                for s in (spans or [])
            ]
            outputs[qid] = {
                "question_id": qid,
                "answer": (batch.get("answer") or [""] * n)[i],
                "sources": sources,
                "retrieved_context": (batch.get("retrieved_context") or [""] * n)[i],
            }
    return sorted(outputs.values(), key=lambda x: int(x["question_id"]))


def write_config_output_json(out_dir: Path, config_id: str) -> None:
    """Write validation_answers.json inside out_dir from run_evals_return.json.

    Also prints a retrieval-score summary (computed deterministically from
    the stored spans, no extra API calls).  Partial_Overall_Score is computed
    by project_compute_metrics_fn during run_evals and logged to MLflow; check
    the MLflow UI or results.csv in out_dir for the full score including
    Generation_Score_3_released.
    """
    results = load_run_results(out_dir)
    if not results:
        print(f"[WARN] {config_id}: run_evals_return.json not found in {out_dir}; skipping output JSON.")
        return

    batches = next(iter(results.values()), [])
    retrieval_score = compute_retrieval_score_from_batches(batches)
    output_list = batches_to_output_list(batches)

    out_path = out_dir / "validation_answers.json"
    out_path.write_text(
        json.dumps(output_list, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[INFO] {config_id}: validation_answers.json written "
        f"({len(output_list)} items, retrieval_score={retrieval_score:.4f}) → {out_path}"
    )


def write_config_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dense_configs": DENSE_CONFIGS,
        "tfidf_configs": TFIDF_CONFIGS,
        "metric_note": (
            "Metrics use Metrics/project1_eval.py. Generation_Score_3_released "
            "uses only correctness, faithfulness, and completeness/5. "
            "Partial_Overall_Score = 0.5 * Retrieval Score + "
            "0.5 * Generation_Score_3_released. Hidden metrics are unavailable."
        ),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_config_table(path: Path, configs: dict[str, dict[str, Any]], family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted({key for cfg in configs.values() for key in cfg})
    fields = ["config_id", "family"] + all_keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config_id, config in configs.items():
            row = {"config_id": config_id, "family": family}
            for key in all_keys:
                value = config.get(key)
                row[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            writer.writerow(row)


def run_rapidfire_config_group(
    deps: dict[str, Any],
    config_group: Any,
    dataset: Any,
    experiment_name: str,
    experiments_path: Path,
    num_shards: int,
    num_actors: int,
    seed: int,
) -> None:
    experiments_path.mkdir(parents=True, exist_ok=True)
    mlflow_process = start_mlflow_if_needed(experiments_path)
    experiment = None

    try:
        experiment = deps["Experiment"](
            experiment_name=experiment_name,
            mode="evals",
            experiment_path=str(experiments_path),
        )
        results = experiment.run_evals(
            config_group=config_group,
            dataset=dataset,
            num_shards=num_shards,
            num_actors=num_actors,
            seed=seed,
        )
        (experiments_path / "run_evals_return.json").write_text(
            json.dumps(results, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        try_write_experiment_frame(experiment, "get_runs_info", experiments_path / "runs_info.csv")
        try_write_experiment_frame(experiment, "get_results", experiments_path / "results.csv")
    finally:
        if experiment is not None:
            try:
                experiment.end()
            except Exception:
                pass
        stop_started_mlflow(mlflow_process)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_configs() -> None:
    print("Dense configs:")
    for config_id, config in DENSE_CONFIGS.items():
        print(f"{config_id}: {config['name']} | {json.dumps(config, ensure_ascii=False)}")

    print("\nTF-IDF configs:")
    for config_id, config in TFIDF_CONFIGS.items():
        print(
            f"{config_id}: {config['name']} | "
            f"chunk={config['chunk_tokens']}/{config['chunk_overlap']} | "
            f"k={config['top_k']} | "
            f"title={config['use_title_boost']} | "
            f"related={config['related_sources']} | "
            f"source_expand={config['source_expand_lines']} | "
            f"context_expand={config['context_expand_lines']}"
        )


def prepare_out_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_config_manifest(path / "config_manifest.json")
    write_config_table(path / "dense_config_table.csv", DENSE_CONFIGS, "dense")
    write_config_table(path / "tfidf_config_table.csv", TFIDF_CONFIGS, "tfidf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RapidFire AI Project 1 experiments.")
    parser.add_argument(
        "--mode",
        choices=["list-configs", "dense-single", "dense-all", "tfidf-single", "tfidf-all", "all"],
        default="list-configs",
    )
    parser.add_argument("--dense-config-id", choices=sorted(DENSE_CONFIGS), help="Dense config to run")
    parser.add_argument("--tfidf-config-id", choices=sorted(TFIDF_CONFIGS), help="TF-IDF config to run")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Golden/validation QA JSON")
    parser.add_argument("--experiments-path", default=str(DEFAULT_EXPERIMENTS_PATH))
    parser.add_argument("--experiment-name", default="project1-rag")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu/v1"))
    parser.add_argument("--model", default=os.environ.get("RAG_MODEL", "api-gpt-oss-120b"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_determinism(args.seed)

    if args.mode == "list-configs":
        list_configs()
        return 0

    deps = require_rapidfire_dependencies()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running RapidFire experiments.")

    ensure_corpus()
    dataset = deps["Dataset"].from_list(load_qa(Path(args.input)))
    base_out = Path(args.experiments_path)
    prepare_out_root(base_out)

    if args.mode == "dense-single":
        if not args.dense_config_id:
            raise ValueError("--dense-config-id is required for --mode dense-single")
        configs_to_run = [(args.dense_config_id, DENSE_CONFIGS[args.dense_config_id])]
        out_parent = base_out
    elif args.mode == "dense-all":
        configs_to_run = list(DENSE_CONFIGS.items())
        out_parent = base_out
    elif args.mode == "all":
        configs_to_run = list(DENSE_CONFIGS.items())
        out_parent = base_out / "dense_configs"
    else:
        configs_to_run = []
        out_parent = base_out

    for config_id, config in configs_to_run:
        out_dir = out_parent / f"{config_id}_{config['name']}"
        print(f"[INFO] Running dense config {config_id}: {config['name']}")
        group = build_dense_config_group(
            deps=deps,
            dense_config=config,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            batch_size=args.batch_size,
        )
        run_rapidfire_config_group(
            deps=deps,
            config_group=group,
            dataset=dataset,
            experiment_name=f"{args.experiment_name}-{config_id}",
            experiments_path=out_dir,
            num_shards=args.num_shards,
            num_actors=args.num_actors,
            seed=args.seed,
        )
        write_config_output_json(out_dir, config_id)

    if args.mode == "tfidf-single":
        if not args.tfidf_config_id:
            raise ValueError("--tfidf-config-id is required for --mode tfidf-single")
        tfidf_to_run = [(args.tfidf_config_id, TFIDF_CONFIGS[args.tfidf_config_id])]
        tfidf_parent = base_out
    elif args.mode == "tfidf-all":
        tfidf_to_run = list(TFIDF_CONFIGS.items())
        tfidf_parent = base_out
    elif args.mode == "all":
        tfidf_to_run = list(TFIDF_CONFIGS.items())
        tfidf_parent = base_out / "tfidf_configs"
    else:
        tfidf_to_run = []
        tfidf_parent = base_out

    for config_id, config in tfidf_to_run:
        out_dir = tfidf_parent / f"{config_id}_{config['name']}"
        print(f"[INFO] Running TF-IDF config {config_id}: {config['name']}")
        group = build_tfidf_config_group(
            deps=deps,
            config_id=config_id,
            config=config,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            batch_size=args.batch_size,
        )
        run_rapidfire_config_group(
            deps=deps,
            config_group=group,
            dataset=dataset,
            experiment_name=f"{args.experiment_name}-{config_id}",
            experiments_path=out_dir,
            num_shards=args.num_shards,
            # TF-IDF preprocessing has a shared in-process cache; keep actors modest.
            num_actors=max(1, min(args.num_actors, 2)),
            seed=args.seed,
        )
        write_config_output_json(out_dir, config_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
