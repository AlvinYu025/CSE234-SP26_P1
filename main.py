#!/usr/bin/env python3
"""RAG pipeline for CSE/DSC 234 Project 1.

Usage:
    python3 main.py --input questions.json --output answers.json

The pipeline uses the RapidFire AI documentation corpus, TF-IDF retrieval,
lexical reranking, source-aware context assembly, and a chat-model generation
step when configured. If generation is unavailable, it returns an extractive
answer from the retrieved context.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sourcedocs"
SOURCE_ZIP = ROOT / "sourcedocs.zip"
DEFAULT_CONFIG: dict[str, Any] = {
    "chunk_tokens": 420,
    "chunk_overlap": 80,
    "retriever": "similarity",
    "top_k": 1,
    "fetch_k": 8,
    "lambda_mult": 0.72,
    "rerank": "title_lexical",
    "rerank_top_n": 1,
    "ngram_range": (1, 2),
    "analyzer": "word",
    "max_context_tokens": 1850,
    "max_answer_sentences": 5,
    "symbol_boost": 0.02,
    "title_boost": 0.12,
    "glossary_symbol_penalty": 0.02,
    "source_expand_lines": 160,
    "related_sources": True,
    "max_related_sources": 5,
}
SYSTEM_PROMPT = (
    "You are answering questions about the RapidFire AI OSS documentation. "
    "Answer using only the provided context. If the answer is not supported by "
    "the context, say that the documentation does not provide enough information. "
    "Prefer a complete answer over an overly short one: include relevant names, "
    "parameters, constraints, defaults, and contrasts that are directly supported "
    "by the context, and omit any detail that is not explicitly stated. For list "
    "or comparison questions, cover each requested item using only supported clauses. "
    "Do not cite unsupported facts. Return the final answer only."
)

QUERY_STOPWORDS = {
    "a",
    "ai",
    "an",
    "and",
    "api",
    "are",
    "argument",
    "arguments",
    "class",
    "config",
    "configs",
    "different",
    "difference",
    "do",
    "does",
    "for",
    "function",
    "how",
    "in",
    "is",
    "parameter",
    "parameters",
    "provided",
    "rapidfire",
    "rapidfireai",
    "support",
    "supported",
    "terms",
    "the",
    "they",
    "to",
    "tutorial",
    "tutorials",
    "type",
    "types",
    "use",
    "uses",
    "using",
    "what",
    "with",
    "you",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file: str
    start_line: int
    end_line: int
    section: str
    text: str

    def source(self, expand_lines: int = 0, max_line: int | None = None) -> dict[str, Any]:
        start = max(1, self.start_line - expand_lines)
        end = self.end_line + expand_lines
        if max_line is not None:
            end = min(max_line, end)
        return {"file": self.file, "lines": [start, end]}


def set_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text.split()) * 1.3))


def ensure_corpus() -> None:
    if SOURCE_DIR.exists() and any(SOURCE_DIR.glob("*.rst")):
        return
    if not SOURCE_ZIP.exists():
        raise FileNotFoundError("Missing sourcedocs/ and sourcedocs.zip")
    subprocess.run(
        ["unzip", "-q", "-o", str(SOURCE_ZIP), "-d", str(ROOT)],
        check=True,
    )


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


def load_documents() -> list[dict[str, Any]]:
    ensure_corpus()
    docs: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DIR.glob("*.rst")):
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        docs.append({"file": path.name, "lines": raw_lines})
    if not docs:
        raise RuntimeError("No .rst files found in sourcedocs/")
    return docs


def build_chunks(
    docs: list[dict[str, Any]],
    chunk_tokens: int = 260,
    chunk_overlap: int = 55,
) -> list[Chunk]:
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
            start_line = block[0][0]
            end_line = block[-1][0]
            section = block[0][2]
            text = clean_rst("\n".join(item[1] for item in block))
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{filename}:{start_line}-{end_line}",
                        file=filename,
                        start_line=start_line,
                        end_line=end_line,
                        section=section,
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


class RagPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.docs = load_documents()
        self.doc_lines = {doc["file"]: doc["lines"] for doc in self.docs}
        self.line_counts = {doc["file"]: len(doc["lines"]) for doc in self.docs}
        self.doc_titles = {
            doc["file"]: first_document_title(doc["lines"], Path(doc["file"]).stem)
            for doc in self.docs
        }
        self.chunks = build_chunks(
            self.docs,
            chunk_tokens=int(self.config["chunk_tokens"]),
            chunk_overlap=int(self.config["chunk_overlap"]),
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
            analyzer=self.config.get("analyzer", "word"),
            ngram_range=tuple(self.config.get("ngram_range", (1, 2))),
            sublinear_tf=True,
            max_df=0.92,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self._chunk_text(c) for c in self.chunks)

    @staticmethod
    def _chunk_text(chunk: Chunk) -> str:
        return f"{chunk.file} {chunk.section}\n{chunk.text}"

    def _query_scores(self, question: str) -> np.ndarray:
        qv = self.vectorizer.transform([question])
        return cosine_similarity(qv, self.matrix).ravel()

    def _lexical_boost(self, question: str, chunk: Chunk) -> float:
        q_terms = {
            t
            for t in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", question.lower())
            if t not in {"what", "how", "does", "the", "and", "for", "with", "are", "you"}
        }
        if not q_terms:
            return 0.0
        haystack = f"{chunk.file} {chunk.section} {chunk.text}".lower()
        hits = sum(1 for term in q_terms if term in haystack)
        return hits / len(q_terms)

    def _title_lexical_boost(self, question: str, chunk: Chunk) -> float:
        q_terms = lexical_terms(question)
        if not q_terms:
            return 0.0

        boost = 0.0
        symbols = query_symbols(question)
        if symbols:
            haystack = f"{chunk.file} {chunk.section} {chunk.text}".lower()
            boost += float(self.config["symbol_boost"]) * sum(1 for symbol in symbols if symbol in haystack)
            if chunk.file == "glossary.rst":
                boost -= float(self.config["glossary_symbol_penalty"])

        title_overlap = len(q_terms & self.title_terms.get(chunk.chunk_id, set()))
        if title_overlap:
            boost += float(self.config["title_boost"]) * title_overlap / len(q_terms)
        return boost

    def _mmr(self, scores: np.ndarray, top_k: int, fetch_k: int, lambda_mult: float) -> list[int]:
        candidates = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))[:fetch_k]
        selected: list[int] = []
        while candidates and len(selected) < top_k:
            if not selected:
                selected.append(candidates.pop(0))
                continue
            best_idx = None
            best_value = -float("inf")
            for idx in candidates:
                candidate_vec = self.matrix[idx]
                selected_vecs = self.matrix[selected]
                diversity = cosine_similarity(candidate_vec, selected_vecs).max()
                value = lambda_mult * scores[idx] - (1.0 - lambda_mult) * diversity
                if value > best_value or (
                    math.isclose(value, best_value) and self.chunks[idx].chunk_id < self.chunks[best_idx].chunk_id
                ):
                    best_value = value
                    best_idx = idx
            selected.append(best_idx)  # type: ignore[arg-type]
            candidates.remove(best_idx)  # type: ignore[arg-type]
        return selected

    def _chunk_covering_line(self, filename: str, line: int) -> Chunk | None:
        candidates = [
            chunk
            for chunk in self.chunks
            if chunk.file == filename and chunk.start_line <= line <= chunk.end_line
        ]
        if candidates:
            return sorted(candidates, key=lambda chunk: (chunk.end_line - chunk.start_line, chunk.chunk_id))[0]
        same_file = [chunk for chunk in self.chunks if chunk.file == filename]
        if not same_file:
            return None
        return sorted(same_file, key=lambda chunk: (abs(chunk.start_line - line), chunk.chunk_id))[0]

    def _select_related(self, question: str, ranked: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        if not ranked or not self.config.get("related_sources"):
            return ranked[: int(self.config.get("rerank_top_n") or self.config["top_k"])]

        q_lower = question.lower()
        max_sources = int(self.config.get("max_related_sources", 5))
        selected: list[tuple[Chunk, float]] = []
        seen_files: set[str] = set()

        def add(chunk: Chunk, score: float) -> None:
            if len(selected) >= max_sources or chunk.file in seen_files:
                return
            seen_files.add(chunk.file)
            selected.append((chunk, score))

        if "tutorial" in q_lower and "use case" in q_lower:
            tutorial_score = ranked[0][1]
            for chunk, score in ranked:
                if chunk.file == "tutorials.rst":
                    add(chunk, score)
                    tutorial_score = score
                    break

            if "rag" in q_lower or "context engineering" in q_lower:
                for filename in ("rag_fiqa.rst", "rag_gsm8k.rst", "rag_scifact.rst"):
                    child = self._chunk_covering_line(filename, 8)
                    if child is not None:
                        add(child, tutorial_score)
            elif "fine" in q_lower or "post-training" in q_lower or "post training" in q_lower:
                for filename in ("sft.rst", "dpo.rst", "grpo.rst"):
                    child = self._chunk_covering_line(filename, 8)
                    if child is not None:
                        add(child, tutorial_score)

        if not selected and ("terminology" in q_lower or re.search(r"\bwhat is an?\s+['\"]", q_lower)):
            for chunk, score in ranked:
                if chunk.file == "glossary.rst":
                    add(chunk, score)
                    break

        if not selected:
            add(*ranked[0])

        if "vector store" in q_lower:
            for chunk, score in ranked:
                if chunk.file in {"ragspecs.rst", "rag_fiqa.rst"}:
                    add(chunk, score)

        if "why" in q_lower or "compared to" in q_lower or "compared with" in q_lower:
            for chunk, score in ranked:
                if chunk.file == "difference.rst":
                    add(chunk, score)
                    break

        return selected[:max_sources]

    def _context_text(self, chunk: Chunk) -> tuple[int, int, str]:
        expand_lines = int(self.config.get("source_expand_lines", 0))
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
        expanded = "\n\n".join(part for part in ordered_parts if part)
        return start, end, clean_rst(expanded)

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        scores = self._query_scores(question)
        top_k = int(self.config["top_k"])
        fetch_k = max(top_k, int(self.config.get("fetch_k", top_k)))
        retriever = self.config.get("retriever", "similarity")
        rerank = self.config.get("rerank")
        threshold = self.config.get("score_threshold")

        if retriever == "mmr":
            indices = self._mmr(scores, top_k=top_k, fetch_k=fetch_k, lambda_mult=float(self.config["lambda_mult"]))
        elif rerank == "title_lexical":
            indices = list(range(len(scores)))
        else:
            pool_size = fetch_k if rerank else top_k
            indices = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))[:pool_size]

        ranked: list[tuple[Chunk, float]] = []
        for idx in indices:
            score = float(scores[idx])
            if threshold is not None and score < float(threshold):
                continue
            chunk = self.chunks[idx]
            if rerank == "lexical":
                score = 0.8 * score + 0.2 * self._lexical_boost(question, chunk)
            elif rerank == "title_lexical":
                score += self._title_lexical_boost(question, chunk)
            ranked.append((chunk, score))

        ranked.sort(key=lambda item: (-item[1], item[0].chunk_id))
        if rerank == "title_lexical":
            return self._select_related(question, ranked)
        return ranked[: int(self.config.get("rerank_top_n") or top_k)]

    def build_context_under_budget(self, retrieved: list[tuple[Chunk, float]]) -> tuple[str, list[Chunk]]:
        budget = int(self.config["max_context_tokens"])
        used = estimate_tokens(SYSTEM_PROMPT)
        parts: list[str] = []
        kept: list[Chunk] = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            start_line, end_line, context_text = self._context_text(chunk)
            label = (
                f"[S{rank}] file={chunk.file} lines={start_line}-{end_line} "
                f"section={chunk.section} score={score:.4f}"
            )
            candidate = f"{label}\n{context_text}"
            cost = estimate_tokens(candidate)
            if used + cost > budget:
                remaining_words = max(0, int((budget - used) / 1.3) - 8)
                if remaining_words > 40:
                    shortened = " ".join(context_text.split()[:remaining_words])
                    candidate = f"{label}\n{shortened}"
                    parts.append(candidate)
                    kept.append(chunk)
                break
            parts.append(candidate)
            kept.append(chunk)
            used += cost
        return "\n\n".join(parts), kept

    def generate_answer(self, question: str, context: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("RAG_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5-mini"
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("TRITONAI_BASE_URL")
        if api_key:
            try:
                from openai import OpenAI  # type: ignore

                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n\n"
                                "Context:\n"
                                f"{context}\n\n"
                                "Write a grounded answer using only the context above. If the question "
                                "asks for a numbered set, include every requested item. If it asks about "
                                "an API, config, or comparison, include relevant parameters, defaults, "
                                "constraints, and exceptions when they appear in the context. Do not say "
                                "the documentation is missing something unless you have checked all context "
                                "snippets. Do not add citation markers; source spans are emitted separately."
                            ),
                        },
                    ],
                )
                text = response.choices[0].message.content
                if text:
                    return text.strip()
            except Exception as exc:
                print(f"[warn] LLM call failed; using extractive fallback: {exc}", file=sys.stderr)
        return extractive_answer(question, context, int(self.config["max_answer_sentences"]))

    def answer_question(self, question_id: int | str, question: str) -> dict[str, Any]:
        retrieved = self.retrieve(question)
        context, kept = self.build_context_under_budget(retrieved)
        answer = self.generate_answer(question, context)
        expand_lines = int(self.config.get("source_expand_lines", 0))
        return {
            "question_id": question_id,
            "answer": answer,
            "sources": [
                chunk.source(expand_lines=expand_lines, max_line=self.line_counts.get(chunk.file))
                for chunk in kept
            ],
            "retrieved_context": context,
        }


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.strip() for p in pieces if len(p.strip()) > 20]


def extractive_answer(question: str, context: str, max_sentences: int = 5) -> str:
    if not context.strip():
        return "The documentation does not provide enough information."
    q_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", question.lower()))
    candidates: list[tuple[float, int, str]] = []
    for pos, sentence in enumerate(split_sentences(context)):
        if sentence.startswith("[S"):
            continue
        s_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", sentence.lower()))
        overlap = len(q_terms & s_terms)
        bonus = 0.0
        lower = sentence.lower()
        if any(word in lower for word in ["supports", "requires", "returns", "default", "must", "can"]):
            bonus += 0.25
        if ":" in sentence or "(" in sentence:
            bonus += 0.1
        candidates.append((overlap + bonus, pos, sentence))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen = [s for score, _, s in candidates[:max_sentences] if score > 0]
    if not chosen:
        chosen = split_sentences(context)[:max_sentences]
    if not chosen:
        return "The documentation does not provide enough information."
    answer = " ".join(chosen)
    answer = re.sub(r"\[S\d+\][^\n]*", "", answer).strip()
    return answer


def read_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "questions" in data:
        items = data["questions"]
    else:
        raise ValueError("Input JSON must be a list or an object with a 'questions' list")
    normalized = []
    for i, item in enumerate(items, start=1):
        qid = item.get("question_id", item.get("id", i))
        question = item.get("question") or item.get("query")
        if not question:
            raise ValueError(f"Missing question text for item {i}")
        normalized.append({"question_id": qid, "question": question})
    return normalized


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic RAG over RapidFire AI docs.")
    parser.add_argument("--input", required=True, help="Input questions JSON")
    parser.add_argument("--output", required=True, help="Output answers JSON")
    parser.add_argument("--config", help="Optional JSON config override")
    args = parser.parse_args()

    set_determinism()
    config = DEFAULT_CONFIG.copy()
    if args.config:
        override = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config.update(override)
    pipeline = RagPipeline(config)
    questions = read_questions(Path(args.input))
    answers = [pipeline.answer_question(q["question_id"], q["question"]) for q in questions]
    write_json(Path(args.output), answers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
