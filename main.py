#!/usr/bin/env python3
"""RAG pipeline for CSE/DSC 234 Project 1.

TF-IDF retrieval with config T10 (chunk=420/80, top_k=1, title boost,
related sources, source_expand=5, context_expand=20). Deterministic.

Usage:
    python3 main.py \
        --input questions.json \
        --output answers.json \
        --corpus-dir sourcedocs \
        --apikey-txt ~/api-key.txt \
        --generation-model api-gpt-oss-120b
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
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT / "sourcedocs"
SOURCE_DIR = DEFAULT_SOURCE_DIR
SOURCE_ZIP = ROOT / "sourcedocs.zip"
TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"

# T10 config
CHUNK_TOKENS = 420
CHUNK_OVERLAP = 80
TOP_K = 1
SOURCE_EXPAND_LINES = 5
CONTEXT_EXPAND_LINES = 20
MAX_RELATED_SOURCES = 5
RELATED_SOURCES = True
MAX_CONTEXT_TOKENS = 1850
MAX_ANSWER_SENTENCES = 5

SYSTEM_PROMPT = (
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


def load_api_key(api_key_txt: Path | None) -> str | None:
    if api_key_txt is not None:
        key_file = api_key_txt.expanduser()
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
            if key:
                return key
            raise ValueError(f"API key file is empty: {key_file}")
        if str(api_key_txt) not in {"~/api-key.txt", str(Path.home() / "api-key.txt")}:
            raise FileNotFoundError(f"API key file not found: {key_file}")

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    return None


def configure_corpus(corpus_dir: Path) -> None:
    global SOURCE_DIR
    candidate = corpus_dir.expanduser()
    if any(candidate.glob("*.rst")):
        SOURCE_DIR = candidate
        return
    nested = candidate / "sourcedocs"
    if nested.exists() and any(nested.glob("*.rst")):
        SOURCE_DIR = nested
        return
    SOURCE_DIR = candidate


def ensure_corpus() -> None:
    if SOURCE_DIR.exists() and any(SOURCE_DIR.glob("*.rst")):
        return
    if SOURCE_DIR != DEFAULT_SOURCE_DIR:
        raise FileNotFoundError(f"No .rst files found under corpus dir: {SOURCE_DIR}")
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


def load_documents() -> list[dict[str, Any]]:
    ensure_corpus()
    docs: list[dict[str, Any]] = []
    for path in sorted(SOURCE_DIR.glob("*.rst")):
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        docs.append({"file": path.name, "text": raw_text, "lines": raw_text.splitlines()})
    if not docs:
        raise RuntimeError("No .rst files found in sourcedocs/")
    return docs


def build_chunks(docs: list[dict[str, Any]]) -> list[Chunk]:
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
                if token_count >= CHUNK_TOKENS:
                    break
            block = units[start:end]
            if not block:
                break
            text = clean_rst("\n".join(item[1] for item in block))
            if text:
                chunks.append(Chunk(
                    chunk_id=f"{filename}:{block[0][0]}-{block[-1][0]}",
                    file=filename,
                    start_line=block[0][0],
                    end_line=block[-1][0],
                    section=block[0][2],
                    text=text,
                ))
            if end >= len(units):
                break
            overlap_words = 0
            next_start = end
            while next_start > start and overlap_words < CHUNK_OVERLAP:
                next_start -= 1
                overlap_words += max(1, len(units[next_start][1].split()))
            start = max(start + 1, next_start)
    return chunks


class RagPipeline:
    def __init__(
        self,
        api_key: str | None = None,
        generation_model: str = "api-gpt-oss-120b",
        base_url: str = TRITONAI_BASE_URL,
    ) -> None:
        self.api_key = api_key
        self.generation_model = generation_model
        self.base_url = base_url
        self.docs = load_documents()
        self.doc_lines = {doc["file"]: doc["lines"] for doc in self.docs}
        self.line_counts = {doc["file"]: len(doc["lines"]) for doc in self.docs}
        self.doc_titles = {
            doc["file"]: first_document_title(doc["lines"], Path(doc["file"]).stem)
            for doc in self.docs
        }
        self.chunks = build_chunks(self.docs)
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
        self.matrix = self.vectorizer.fit_transform(
            f"{c.file} {c.section}\n{c.text}" for c in self.chunks
        )

    def _query_scores(self, question: str) -> Any:
        qv = self.vectorizer.transform([question])
        return cosine_similarity(qv, self.matrix).ravel()

    def _title_boost(self, question: str, chunk: Chunk) -> float:
        q_terms = lexical_terms(question)
        if not q_terms:
            return 0.0
        boost = 0.0
        symbols = query_symbols(question)
        if symbols:
            haystack = f"{chunk.file} {chunk.section} {chunk.text}".lower()
            boost += 0.02 * sum(1 for s in symbols if s in haystack)
            if chunk.file == "glossary.rst":
                boost -= 0.02
        title_overlap = len(q_terms & self.title_terms.get(chunk.chunk_id, set()))
        if title_overlap:
            boost += 0.12 * title_overlap / len(q_terms)
        return boost

    def _chunk_covering_line(self, filename: str, line: int) -> Chunk | None:
        candidates = [
            c for c in self.chunks
            if c.file == filename and c.start_line <= line <= c.end_line
        ]
        if candidates:
            return sorted(candidates, key=lambda c: (c.end_line - c.start_line, c.chunk_id))[0]
        same_file = [c for c in self.chunks if c.file == filename]
        if not same_file:
            return None
        return sorted(same_file, key=lambda c: (abs(c.start_line - line), c.chunk_id))[0]

    def _select_related(self, question: str, ranked: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        if not ranked:
            return []
        if not RELATED_SOURCES:
            return ranked[:TOP_K]
        q_lower = question.lower()
        selected: list[tuple[Chunk, float]] = []
        seen_files: set[str] = set()

        def add(chunk: Chunk, score: float) -> None:
            if len(selected) >= MAX_RELATED_SOURCES or chunk.file in seen_files:
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
                for fn in ("rag_fiqa.rst", "rag_gsm8k.rst", "rag_scifact.rst"):
                    child = self._chunk_covering_line(fn, 8)
                    if child is not None:
                        add(child, ranked[0][1])
            elif "fine" in q_lower or "post-training" in q_lower or "post training" in q_lower:
                for fn in ("sft.rst", "dpo.rst", "grpo.rst"):
                    child = self._chunk_covering_line(fn, 8)
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

        return selected[:MAX_RELATED_SOURCES]

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        scores = self._query_scores(question)
        ranked = [
            (self.chunks[i], float(scores[i]) + self._title_boost(question, self.chunks[i]))
            for i in range(len(scores))
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return self._select_related(question, ranked)

    def _context_text(self, chunk: Chunk) -> tuple[int, int, str]:
        start = max(1, chunk.start_line - CONTEXT_EXPAND_LINES)
        end = min(self.line_counts.get(chunk.file, chunk.end_line), chunk.end_line + CONTEXT_EXPAND_LINES)
        lines = self.doc_lines.get(chunk.file)
        if not lines:
            return chunk.start_line, chunk.end_line, chunk.text
        core = "\n".join(l.rstrip() for l in lines[chunk.start_line - 1 : chunk.end_line] if l.rstrip())
        after = "\n".join(l.rstrip() for l in lines[chunk.end_line : end] if l.rstrip())
        before = "\n".join(l.rstrip() for l in lines[start - 1 : chunk.start_line - 1] if l.rstrip())
        parts = [f"Core lines {chunk.start_line}-{chunk.end_line}:\n{core}"]
        if after:
            parts.append(f"Following context lines {chunk.end_line + 1}-{end}:\n{after}")
        if before:
            parts.append(f"Preceding context lines {start}-{chunk.start_line - 1}:\n{before}")
        return start, end, clean_rst("\n\n".join(parts))

    def build_context(self, retrieved: list[tuple[Chunk, float]]) -> tuple[str, list[dict[str, Any]]]:
        used = estimate_tokens(SYSTEM_PROMPT)
        parts: list[str] = []
        sources: list[dict[str, Any]] = []

        for rank, (chunk, score) in enumerate(retrieved, start=1):
            ctx_start, ctx_end, ctx_text = self._context_text(chunk)
            label = (
                f"[S{rank}] file={chunk.file} lines={ctx_start}-{ctx_end} "
                f"section={chunk.section} score={score:.4f}"
            )
            candidate = f"{label}\n{ctx_text}"
            cost = estimate_tokens(candidate)

            if used + cost > MAX_CONTEXT_TOKENS:
                remaining_words = max(0, int((MAX_CONTEXT_TOKENS - used) / 1.3) - 8)
                if remaining_words > 40:
                    candidate = f"{label}\n{' '.join(ctx_text.split()[:remaining_words])}"
                    parts.append(candidate)
                    sources.append(chunk.source(expand_lines=SOURCE_EXPAND_LINES,
                                                 max_line=self.line_counts.get(chunk.file)))
                break

            parts.append(candidate)
            sources.append(chunk.source(expand_lines=SOURCE_EXPAND_LINES,
                                         max_line=self.line_counts.get(chunk.file)))
            used += cost

        return "\n\n".join(parts), sources

    def generate_answer(self, question: str, context: str) -> str:
        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                response = client.chat.completions.create(
                    model=self.generation_model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{context}\n\n"
                                f"Question:\n{question}\n\n"
                                "Answer using only the context. If the question asks for a list, comparison, "
                                "API behavior, default, or constraint, include all supported details from the context."
                            ),
                        },
                    ],
                )
                text = response.choices[0].message.content
                if text:
                    return text.strip()
            except Exception as exc:
                print(f"[warn] LLM call failed; using extractive fallback: {exc}", file=sys.stderr)
        return _extractive_answer(question, context)

    def answer_question(self, question_id: int | str, question: str) -> dict[str, Any]:
        retrieved = self.retrieve(question)
        context, sources = self.build_context(retrieved)
        return {
            "question_id": question_id,
            "answer": self.generate_answer(question, context),
            "sources": sources,
            "retrieved_context": context,
        }


def _extractive_answer(question: str, context: str) -> str:
    if not context.strip():
        return "The documentation does not provide enough information."
    q_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", question.lower()))
    candidates: list[tuple[float, int, str]] = []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", re.sub(r"\s+", " ", context))
    for pos, sentence in enumerate(s.strip() for s in sentences if len(s.strip()) > 20):
        if sentence.startswith("[S"):
            continue
        s_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", sentence.lower()))
        overlap = float(len(q_terms & s_terms))
        lower = sentence.lower()
        if any(w in lower for w in ["supports", "requires", "returns", "default", "must", "can"]):
            overlap += 0.25
        if ":" in sentence or "(" in sentence:
            overlap += 0.1
        candidates.append((overlap, pos, sentence))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen = [s for score, _, s in candidates[:MAX_ANSWER_SENTENCES] if score > 0]
    if not chosen:
        chosen = [s.strip() for s in sentences[:MAX_ANSWER_SENTENCES] if len(s.strip()) > 20]
    if not chosen:
        return "The documentation does not provide enough information."
    return re.sub(r"\[S\d+\][^\n]*", "", " ".join(chosen)).strip()


def read_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("questions", [])
    if not items:
        raise ValueError("Input JSON must be a list or an object with a 'questions' list")
    normalized = []
    for i, item in enumerate(items, start=1):
        qid = item.get("question_id", item.get("id", i))
        question = item.get("question") or item.get("query")
        if not question:
            raise ValueError(f"Missing question text for item {i}")
        normalized.append({"question_id": qid, "question": question})
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic TF-IDF RAG over RapidFire AI docs.")
    parser.add_argument("--input", required=True, help="Input questions JSON")
    parser.add_argument("--output", required=True, help="Output answers JSON")
    parser.add_argument(
        "--corpus-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Root directory containing the RapidFire .rst documentation corpus",
    )
    parser.add_argument(
        "--apikey-txt",
        default="~/api-key.txt",
        help="Path to the TritonAI/OpenAI-compatible gateway API key text file",
    )
    parser.add_argument(
        "--generation-model",
        default="api-gpt-oss-120b",
        help="Gateway model id used for answer generation",
    )
    parser.add_argument(
        "--base-url",
        default=TRITONAI_BASE_URL,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    set_determinism()
    configure_corpus(Path(args.corpus_dir))
    api_key = load_api_key(Path(args.apikey_txt) if args.apikey_txt else None)
    pipeline = RagPipeline(
        api_key=api_key,
        generation_model=args.generation_model,
        base_url=args.base_url,
    )
    questions = read_questions(Path(args.input))
    answers = [pipeline.answer_question(q["question_id"], q["question"]) for q in questions]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(answers)} answers to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
