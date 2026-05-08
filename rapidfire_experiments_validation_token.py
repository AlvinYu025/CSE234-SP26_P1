#!/usr/bin/env python3
"""Token-budget ablations for the Project 1 validation set.

This script keeps the current TF-IDF retrieval design from main.py, but varies
chunk size, related-source count, and context packing. It measures prompt tokens
with the generation model itself by making OpenAI-compatible calls with
max_tokens=1 and reading response.usage.prompt_tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

import main as project_main


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "validation-set-golden-qa-pairs.json"
DEFAULT_OUTPUT_ROOT = ROOT / "rapidfire_experiments_validation_token"
DEFAULT_CORPUS_DIR = ROOT / "sourcedocs"
TRITONAI_BASE_URL = "https://tritonai-api.ucsd.edu/v1"

USER_SUFFIX = (
    "Answer using only the context. If the question asks for a list, comparison, "
    "API behavior, default, or constraint, include all supported details from the context."
)


@dataclass(frozen=True)
class TokenExperimentConfig:
    config_id: str
    chunk_tokens: int
    chunk_overlap: int
    max_related_sources: int
    pack_strategy: str
    token_cap: int
    source_expand_lines: int = 3
    context_expand_lines: int = 3

    @property
    def name(self) -> str:
        return (
            f"{self.config_id}_chunk{self.chunk_tokens}_overlap{self.chunk_overlap}_"
            f"rel{self.max_related_sources}_{self.pack_strategy}_cap{self.token_cap}"
        )


CONFIGS = [
    TokenExperimentConfig("TC01", 420, 80, 5, "sequential", 1950),
    TokenExperimentConfig("TC02", 420, 80, 5, "sequential", 1900),
    TokenExperimentConfig("TC03", 384, 64, 5, "sequential", 1900),
    TokenExperimentConfig("TC04", 320, 64, 5, "sequential", 1900),
    TokenExperimentConfig("TC05", 256, 48, 5, "sequential", 1900),
    TokenExperimentConfig("TC06", 420, 80, 4, "sequential", 1900),
    TokenExperimentConfig("TC07", 420, 80, 3, "sequential", 1900),
    TokenExperimentConfig("TC08", 320, 64, 5, "balanced", 1900),
    TokenExperimentConfig("TC09", 320, 64, 5, "question_aware", 1900),
    TokenExperimentConfig("TC10", 256, 48, 5, "staged", 1900),
]
CONFIG_BY_ID = {cfg.config_id: cfg for cfg in CONFIGS}


@dataclass
class ContextBlock:
    rank: int
    chunk: project_main.Chunk
    score: float
    label: str
    text: str
    source: dict[str, Any]
    units: list[str]


def retry_call(label: str, fn: Any, max_attempts: int = 6) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry gateway/rate-limit failures.
            last_exc = exc
            if attempt == max_attempts:
                break
            wait = min(30.0, 1.5 * (2 ** (attempt - 1)))
            print(f"[warn] {label} failed on attempt {attempt}; retrying in {wait:.1f}s: {exc}",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after {max_attempts} attempts: {last_exc}") from last_exc


def split_units(text: str) -> list[str]:
    """Split context text into paragraph/sentence units for token-safe trimming."""
    text = text.strip()
    if not text:
        return []
    units: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= 70:
            units.append(paragraph)
            continue
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'])", re.sub(r"\s+", " ", paragraph))
            if s.strip()
        ]
        if len(sentences) <= 1:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            units.extend(lines if lines else [paragraph])
        else:
            units.extend(sentences)
    return units


def lexical_terms(text: str) -> set[str]:
    return project_main.lexical_terms(text)


def unit_score(question_terms: set[str], unit: str) -> float:
    terms = lexical_terms(unit)
    if not terms:
        return 0.0
    score = 2.0 * len(question_terms & terms)
    lower = unit.lower()
    for keyword in (
        "workflow", "tutorial", "rag", "context engineering", "default", "requires",
        "supports", "vector store", "generator", "evaluation", "retrieval",
    ):
        if keyword in lower:
            score += 0.6
    if re.search(r"\b(?:RF[A-Za-z0-9_]+|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z]{2,}[A-Za-z0-9_]*)\b", unit):
        score += 0.5
    return score


def make_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": project_main.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\n{USER_SUFFIX}",
        },
    ]


class TokenBudgetPipeline(project_main.RagPipeline):
    def __init__(
        self,
        cfg: TokenExperimentConfig,
        api_key: str,
        generation_model: str,
        base_url: str,
    ) -> None:
        self.cfg = cfg
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.token_cache: dict[str, int] = {}
        self.debug_rows: list[dict[str, Any]] = []
        self.token_rows: list[dict[str, Any]] = []
        super().__init__(api_key=api_key, generation_model=generation_model, base_url=base_url)

    def _prompt_tokens(self, question: str, context: str) -> int:
        messages = make_messages(question, context)
        raw = json.dumps(
            {"model": self.generation_model, "messages": messages},
            sort_keys=True,
            ensure_ascii=False,
        )
        cache_key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        if cache_key in self.token_cache:
            return self.token_cache[cache_key]

        def call() -> Any:
            return self.client.chat.completions.create(
                model=self.generation_model,
                temperature=0.0,
                max_tokens=1,
                messages=messages,
            )

        response = retry_call("prompt token preflight", call)
        if response.usage is None or response.usage.prompt_tokens is None:
            raise RuntimeError("Gateway response did not include usage.prompt_tokens")
        tokens = int(response.usage.prompt_tokens)
        self.token_cache[cache_key] = tokens
        return tokens

    def _generate_with_usage(self, question: str, context: str) -> tuple[str, dict[str, Any]]:
        messages = make_messages(question, context)

        def call() -> Any:
            return self.client.chat.completions.create(
                model=self.generation_model,
                temperature=0.0,
                messages=messages,
            )

        response = retry_call("answer generation", call)
        text = response.choices[0].message.content or ""
        usage: dict[str, Any] = {}
        if response.usage is not None:
            usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
        return text.strip(), usage

    def _make_blocks(self, retrieved: list[tuple[project_main.Chunk, float]]) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            ctx_start, ctx_end, ctx_text = self._context_text(chunk)
            label = (
                f"[S{rank}] file={chunk.file} lines={ctx_start}-{ctx_end} "
                f"section={chunk.section} score={score:.4f}"
            )
            blocks.append(ContextBlock(
                rank=rank,
                chunk=chunk,
                score=score,
                label=label,
                text=ctx_text,
                source=chunk.source(
                    expand_lines=self.cfg.source_expand_lines,
                    max_line=self.line_counts.get(chunk.file),
                ),
                units=split_units(ctx_text),
            ))
        return blocks

    @staticmethod
    def _part(block: ContextBlock, units: list[str]) -> str:
        return f"{block.label}\n" + "\n\n".join(unit for unit in units if unit.strip())

    @staticmethod
    def _join_parts(parts: list[str]) -> str:
        return "\n\n".join(part for part in parts if part.strip())

    def _longest_prefix_part(
        self,
        question: str,
        prefix_parts: list[str],
        block: ContextBlock,
    ) -> tuple[str | None, int, int]:
        units = block.units
        if not units:
            return None, self._prompt_tokens(question, self._join_parts(prefix_parts)), 0

        low, high = 0, len(units)
        best_part: str | None = None
        best_tokens = self._prompt_tokens(question, self._join_parts(prefix_parts))
        best_units = 0
        while low <= high:
            mid = (low + high) // 2
            candidate_part = self._part(block, units[:mid])
            candidate_context = self._join_parts(prefix_parts + [candidate_part])
            tokens = self._prompt_tokens(question, candidate_context)
            if tokens <= self.cfg.token_cap and mid > 0:
                best_part = candidate_part
                best_tokens = tokens
                best_units = mid
                low = mid + 1
            elif tokens <= self.cfg.token_cap and mid == 0:
                low = mid + 1
            else:
                high = mid - 1
        return best_part, best_tokens, best_units

    def _pack_sequential(self, question: str, blocks: list[ContextBlock]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        included_files: list[str] = []
        dropped_files: list[str] = []
        trimmed = False

        for block in blocks:
            full_part = self._part(block, block.units)
            candidate = self._join_parts(parts + [full_part])
            tokens = self._prompt_tokens(question, candidate)
            if tokens <= self.cfg.token_cap:
                parts.append(full_part)
                sources.append(block.source)
                included_files.append(block.chunk.file)
                continue

            trimmed_part, _, n_units = self._longest_prefix_part(question, parts, block)
            if trimmed_part is not None:
                parts.append(trimmed_part)
                sources.append(block.source)
                included_files.append(block.chunk.file)
                trimmed = n_units < len(block.units)
            else:
                dropped_files.append(block.chunk.file)
            dropped_files.extend(b.chunk.file for b in blocks[block.rank:])
            break

        context = self._join_parts(parts)
        final_tokens = self._prompt_tokens(question, context)
        return context, sources, {
            "included_files": included_files,
            "dropped_files": dropped_files,
            "trimmed": trimmed,
            "final_prompt_tokens": final_tokens,
        }

    def _pack_balanced(self, question: str, blocks: list[ContextBlock]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        selected: dict[int, list[str]] = {i: [] for i in range(len(blocks))}
        included: set[int] = set()
        changed = True
        trimmed = False

        def current_parts() -> list[str]:
            return [self._part(blocks[i], selected[i]) for i in range(len(blocks)) if selected[i]]

        # Seed every source with a short opening snippet when it fits.
        for i, block in enumerate(blocks):
            seed_units = block.units[: min(2, len(block.units))]
            if not seed_units:
                continue
            candidate_selected = {idx: list(vals) for idx, vals in selected.items()}
            candidate_selected[i].extend(seed_units)
            parts = [self._part(blocks[idx], candidate_selected[idx])
                     for idx in range(len(blocks)) if candidate_selected[idx]]
            if self._prompt_tokens(question, self._join_parts(parts)) <= self.cfg.token_cap:
                selected[i].extend(seed_units)
                included.add(i)

        # Round-robin fill remaining units so one long source does not consume everything.
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if len(selected[i]) >= len(block.units):
                    continue
                next_unit = block.units[len(selected[i])]
                candidate_parts = current_parts()
                if selected[i]:
                    candidate_selected = {idx: list(vals) for idx, vals in selected.items()}
                    candidate_selected[i].append(next_unit)
                    candidate_parts = [
                        self._part(blocks[idx], candidate_selected[idx])
                        for idx in range(len(blocks)) if candidate_selected[idx]
                    ]
                else:
                    candidate_parts.append(self._part(block, [next_unit]))
                if self._prompt_tokens(question, self._join_parts(candidate_parts)) <= self.cfg.token_cap:
                    selected[i].append(next_unit)
                    included.add(i)
                    changed = True
                else:
                    trimmed = True

        parts = current_parts()
        context = self._join_parts(parts)
        final_tokens = self._prompt_tokens(question, context)
        included_files = [blocks[i].chunk.file for i in range(len(blocks)) if selected[i]]
        dropped_files = [blocks[i].chunk.file for i in range(len(blocks)) if not selected[i]]
        sources = [blocks[i].source for i in range(len(blocks)) if selected[i]]
        return context, sources, {
            "included_files": included_files,
            "dropped_files": dropped_files,
            "trimmed": trimmed,
            "final_prompt_tokens": final_tokens,
        }

    def _pack_question_aware(
        self,
        question: str,
        blocks: list[ContextBlock],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        q_terms = lexical_terms(question)
        selected: dict[int, set[int]] = {i: set() for i in range(len(blocks))}
        candidates: list[tuple[float, int, int]] = []
        for i, block in enumerate(blocks):
            for j, unit in enumerate(block.units):
                score = unit_score(q_terms, unit) + (0.1 / max(1, block.rank))
                candidates.append((score, i, j))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        def parts_from_selected() -> list[str]:
            parts: list[str] = []
            for i, block in enumerate(blocks):
                if selected[i]:
                    ordered = [block.units[j] for j in sorted(selected[i])]
                    parts.append(self._part(block, ordered))
            return parts

        # Coverage pass: best unit per source first.
        for i, block in enumerate(blocks):
            if not block.units:
                continue
            best_j = max(range(len(block.units)), key=lambda j: unit_score(q_terms, block.units[j]))
            selected[i].add(best_j)
            if self._prompt_tokens(question, self._join_parts(parts_from_selected())) > self.cfg.token_cap:
                selected[i].remove(best_j)

        trimmed = False
        for _, i, j in candidates:
            if j in selected[i]:
                continue
            selected[i].add(j)
            if self._prompt_tokens(question, self._join_parts(parts_from_selected())) > self.cfg.token_cap:
                selected[i].remove(j)
                trimmed = True

        parts = parts_from_selected()
        context = self._join_parts(parts)
        final_tokens = self._prompt_tokens(question, context)
        included_files = [blocks[i].chunk.file for i in range(len(blocks)) if selected[i]]
        dropped_files = [blocks[i].chunk.file for i in range(len(blocks)) if not selected[i]]
        sources = [blocks[i].source for i in range(len(blocks)) if selected[i]]
        return context, sources, {
            "included_files": included_files,
            "dropped_files": dropped_files,
            "trimmed": trimmed,
            "final_prompt_tokens": final_tokens,
        }

    def _pack_staged(self, question: str, blocks: list[ContextBlock]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        q_terms = lexical_terms(question)
        selected: dict[int, set[int]] = {i: set() for i in range(len(blocks))}

        def parts_from_selected() -> list[str]:
            parts: list[str] = []
            for i, block in enumerate(blocks):
                if selected[i]:
                    ordered = [block.units[j] for j in sorted(selected[i])]
                    parts.append(self._part(block, ordered))
            return parts

        # Stage 1: one compact relevant snippet per source.
        for i, block in enumerate(blocks):
            if not block.units:
                continue
            first = max(range(len(block.units)), key=lambda j: unit_score(q_terms, block.units[j]))
            selected[i].add(first)
            if self._prompt_tokens(question, self._join_parts(parts_from_selected())) > self.cfg.token_cap:
                selected[i].remove(first)

        # Stage 2: fill by retrieval rank, then by local relevance.
        trimmed = False
        for i, block in enumerate(blocks):
            ranked_units = sorted(
                range(len(block.units)),
                key=lambda j: (-unit_score(q_terms, block.units[j]), j),
            )
            for j in ranked_units:
                if j in selected[i]:
                    continue
                selected[i].add(j)
                if self._prompt_tokens(question, self._join_parts(parts_from_selected())) > self.cfg.token_cap:
                    selected[i].remove(j)
                    trimmed = True

        parts = parts_from_selected()
        context = self._join_parts(parts)
        final_tokens = self._prompt_tokens(question, context)
        included_files = [blocks[i].chunk.file for i in range(len(blocks)) if selected[i]]
        dropped_files = [blocks[i].chunk.file for i in range(len(blocks)) if not selected[i]]
        sources = [blocks[i].source for i in range(len(blocks)) if selected[i]]
        return context, sources, {
            "included_files": included_files,
            "dropped_files": dropped_files,
            "trimmed": trimmed,
            "final_prompt_tokens": final_tokens,
        }

    def build_context_for_question(
        self,
        question: str,
        retrieved: list[tuple[project_main.Chunk, float]],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        blocks = self._make_blocks(retrieved)
        base_tokens = self._prompt_tokens(question, "")

        if self.cfg.pack_strategy == "sequential":
            context, sources, debug = self._pack_sequential(question, blocks)
        elif self.cfg.pack_strategy == "balanced":
            context, sources, debug = self._pack_balanced(question, blocks)
        elif self.cfg.pack_strategy == "question_aware":
            context, sources, debug = self._pack_question_aware(question, blocks)
        elif self.cfg.pack_strategy == "staged":
            context, sources, debug = self._pack_staged(question, blocks)
        else:
            raise ValueError(f"Unknown pack strategy: {self.cfg.pack_strategy}")

        prompt_tokens = self._prompt_tokens(question, context)
        debug.update({
            "base_prompt_tokens_without_context": base_tokens,
            "remaining_budget_for_context": self.cfg.token_cap - base_tokens,
            "prompt_tokens": prompt_tokens,
            "token_cap": self.cfg.token_cap,
            "over_cap": prompt_tokens > self.cfg.token_cap,
            "retrieved_files": [chunk.file for chunk, _ in retrieved],
            "source_count": len(sources),
            "context_chars": len(context),
            "context_words": len(context.split()),
        })
        return context, sources, debug

    def answer_question(self, question_id: int | str, question: str) -> dict[str, Any]:
        retrieved = self.retrieve(question)
        context, sources, debug = self.build_context_for_question(question, retrieved)
        answer, generation_usage = self._generate_with_usage(question, context)
        prompt_tokens = int(generation_usage.get("prompt_tokens") or debug["prompt_tokens"])

        debug_row = {
            "question_id": question_id,
            "config_id": self.cfg.config_id,
            **debug,
            "generation_usage": generation_usage,
        }
        token_row = {
            "question_id": question_id,
            "prompt_tokens": prompt_tokens,
            "preflight_prompt_tokens": debug["prompt_tokens"],
            "token_cap": self.cfg.token_cap,
            "over_cap": prompt_tokens > self.cfg.token_cap,
            "completion_tokens": generation_usage.get("completion_tokens"),
            "total_tokens": generation_usage.get("total_tokens"),
            "context_chars": len(context),
            "context_words": len(context.split()),
        }
        self.debug_rows.append(debug_row)
        self.token_rows.append(token_row)

        return {
            "question_id": question_id,
            "answer": answer,
            "sources": sources,
            "retrieved_context": context,
        }


def apply_config(cfg: TokenExperimentConfig, corpus_dir: Path) -> None:
    project_main.configure_corpus(corpus_dir)
    project_main.CHUNK_TOKENS = cfg.chunk_tokens
    project_main.CHUNK_OVERLAP = cfg.chunk_overlap
    project_main.TOP_K = 1
    project_main.SOURCE_EXPAND_LINES = cfg.source_expand_lines
    project_main.CONTEXT_EXPAND_LINES = cfg.context_expand_lines
    project_main.MAX_RELATED_SOURCES = cfg.max_related_sources
    project_main.RELATED_SOURCES = True


def read_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    questions = project_main.read_questions(path)
    if limit is not None:
        return questions[:limit]
    return questions


def load_api_key(path: Path) -> str:
    key = project_main.load_api_key(path)
    if not key:
        raise RuntimeError("An API key is required for token-budget experiments")
    return key


def run_metric_commands(
    out_dir: Path,
    output_file: Path,
    validation_file: Path,
    api_key: str,
    judge_model: str,
    base_url: str,
    skip_judge: bool,
) -> None:
    retrieval = subprocess.run(
        [
            sys.executable,
            "Metrics/evaluate_retrieval.py",
            "--output",
            str(output_file),
            "--validation",
            str(validation_file),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (out_dir / "retrieval_metrics.json").write_text(retrieval.stdout, encoding="utf-8")
    (out_dir / "retrieval_metrics.stderr").write_text(retrieval.stderr, encoding="utf-8")
    (out_dir / "retrieval_metrics.txt").write_text(retrieval.stderr + retrieval.stdout, encoding="utf-8")
    if retrieval.returncode != 0:
        raise RuntimeError(f"retrieval evaluator failed for {output_file}: {retrieval.stderr}")

    if skip_judge:
        return

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = api_key
    judge = subprocess.run(
        [
            sys.executable,
            "Metrics/run_judge.py",
            "--output",
            str(output_file),
            "--validation",
            str(validation_file),
            "--base-url",
            base_url,
            "--model",
            judge_model,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    (out_dir / "judge_metrics.json").write_text(judge.stdout, encoding="utf-8")
    (out_dir / "judge_metrics.stderr").write_text(judge.stderr, encoding="utf-8")
    if judge.returncode != 0:
        raise RuntimeError(f"judge failed for {output_file}: {judge.stderr}")


def summarize_token_rows(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    prompts = [int(row["prompt_tokens"]) for row in rows]
    return {
        "model": model,
        "n": len(rows),
        "mean_prompt_tokens": sum(prompts) / len(prompts) if prompts else 0.0,
        "max_prompt_tokens": max(prompts) if prompts else 0,
        "min_prompt_tokens": min(prompts) if prompts else 0,
        "num_prompt_tokens_over_cap": sum(
            int(row["prompt_tokens"]) > int(row["token_cap"]) for row in rows
        ),
        "num_prompt_tokens_over_2000": sum(int(row["prompt_tokens"]) > 2000 for row in rows),
        "top5_prompt_tokens": sorted(rows, key=lambda row: int(row["prompt_tokens"]), reverse=True)[:5],
        "rows": rows,
    }


def parse_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return json.loads(text)


def run_config(
    cfg: TokenExperimentConfig,
    args: argparse.Namespace,
    api_key: str,
) -> None:
    out_dir = args.output_root / cfg.name
    output_file = out_dir / "validation_answers.json"
    done_file = out_dir / "DONE"
    if done_file.exists() and output_file.exists() and not args.force:
        print(f"[skip] {cfg.config_id}: existing run found at {out_dir}", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2) + "\n", encoding="utf-8")
    print(f"[run] {cfg.config_id}: {cfg.name}", file=sys.stderr)

    project_main.set_determinism()
    apply_config(cfg, args.corpus_dir)
    pipeline = TokenBudgetPipeline(
        cfg=cfg,
        api_key=api_key,
        generation_model=args.generation_model,
        base_url=args.base_url,
    )
    questions = read_questions(args.input, limit=args.limit)
    answers: list[dict[str, Any]] = []
    for idx, question in enumerate(questions, start=1):
        answer = pipeline.answer_question(question["question_id"], question["question"])
        answers.append(answer)
        token_row = pipeline.token_rows[-1]
        print(
            f"[{cfg.config_id}] {idx:02d}/{len(questions)} qid={question['question_id']} "
            f"tokens={token_row['prompt_tokens']} sources={len(answer['sources'])}",
            file=sys.stderr,
            flush=True,
        )

    output_file.write_text(json.dumps(answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "context_debug.json").write_text(
        json.dumps(pipeline.debug_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    token_summary = summarize_token_rows(pipeline.token_rows, args.generation_model)
    (out_dir / "prompt_token_usage.json").write_text(
        json.dumps(token_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not args.no_metrics:
        run_metric_commands(
            out_dir=out_dir,
            output_file=output_file,
            validation_file=args.input,
            api_key=api_key,
            judge_model=args.judge_model,
            base_url=args.base_url,
            skip_judge=args.skip_judge,
        )

    done_file.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")
    print(f"[done] {cfg.config_id}: {out_dir}", file=sys.stderr)


def load_retrieval_summary(path: Path) -> dict[str, Any] | None:
    data = parse_json_file(path)
    return data.get("summary") if data else None


def build_summary(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        out_dir = output_root / cfg.name
        token_summary = parse_json_file(out_dir / "prompt_token_usage.json")
        retrieval = load_retrieval_summary(out_dir / "retrieval_metrics.json")
        judge = load_retrieval_summary(out_dir / "judge_metrics.json")
        if not token_summary or not retrieval:
            continue
        generation_score = None
        correctness = None
        faithfulness = None
        completeness = None
        if judge:
            generation_score = judge.get("Partial Generation Score")
            correctness = judge.get("Correctness (pass rate)")
            faithfulness = judge.get("Faithfulness (pass rate)")
            completeness = judge.get("Completeness (raw, 0-5)")
        retrieval_score = retrieval.get("Retrieval Score")
        overall = None
        if retrieval_score is not None and generation_score is not None:
            overall = (retrieval_score + generation_score) / 2
        rows.append({
            "config_id": cfg.config_id,
            "name": cfg.name,
            "chunk_tokens": cfg.chunk_tokens,
            "chunk_overlap": cfg.chunk_overlap,
            "max_related_sources": cfg.max_related_sources,
            "pack_strategy": cfg.pack_strategy,
            "token_cap": cfg.token_cap,
            "retrieval_score": retrieval_score,
            "generation_score": generation_score,
            "overall_mean": overall,
            "correctness": correctness,
            "faithfulness": faithfulness,
            "completeness_raw_0_to_5": completeness,
            "mean_prompt_tokens": token_summary.get("mean_prompt_tokens"),
            "max_prompt_tokens": token_summary.get("max_prompt_tokens"),
            "over_cap": token_summary.get("num_prompt_tokens_over_cap"),
            "over_2000": token_summary.get("num_prompt_tokens_over_2000"),
            "top_prompt_question": (
                token_summary.get("top5_prompt_tokens", [{}])[0].get("question_id")
                if token_summary.get("top5_prompt_tokens") else None
            ),
        })

    summary = {
        "description": "Token-budget context-packing ablation on the released validation set.",
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "token_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    best = max(
        (row for row in rows if row["overall_mean"] is not None),
        key=lambda row: row["overall_mean"],
        default=None,
    )
    lines = [
        "# Token-Budget Context Ablation on Validation Set",
        "",
        "All runs measure prompt tokens by calling the generation model with `max_tokens=1` and reading `response.usage.prompt_tokens`.",
        "",
    ]
    if best is not None:
        lines.extend([
            f"Best validation overall: **{best['config_id']}** "
            f"(`{best['name']}`), overall `{fmt(best['overall_mean'])}`, "
            f"retrieval `{fmt(best['retrieval_score'])}`, generation `{fmt(best['generation_score'])}`, "
            f"max prompt tokens `{best['max_prompt_tokens']}`.",
            "",
        ])
    lines.extend([
        "| Config | Chunk | Related | Strategy | Cap | Retrieval | Generation | Overall | Max prompt tokens | > cap | > 2000 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['config_id']} | {row['chunk_tokens']}/{row['chunk_overlap']} | "
            f"{row['max_related_sources']} | {row['pack_strategy']} | {row['token_cap']} | "
            f"{fmt(row['retrieval_score'])} | "
            f"{fmt(row['generation_score'])} | "
            f"{fmt(row['overall_mean'])} | "
            f"{row['max_prompt_tokens']} | {row['over_cap']} | {row['over_2000']} |"
        )
    lines.extend([
        "",
        "Notes:",
        "",
        "- `Retrieval` is the released span-overlap retrieval score.",
        "- `Generation` is the released partial judge score using correctness, faithfulness, and completeness.",
        "- `Overall` is the mean of retrieval and generation, used only as a validation comparison heuristic.",
        "- Every listed prompt-token value is returned by the generation model API, not estimated locally.",
    ])
    (output_root / "token_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run token-budget ablations on validation data.")
    parser.add_argument("--mode", choices=["list-configs", "run", "summarize"], default="run")
    parser.add_argument("--config-id", default="all", help="Config id such as TC01, or all")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--apikey-txt", type=Path, default=Path("~/api-key.txt"))
    parser.add_argument("--generation-model", default="api-gpt-oss-120b")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6")
    parser.add_argument("--base-url", default=TRITONAI_BASE_URL)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.corpus_dir = args.corpus_dir.expanduser().resolve()
    args.apikey_txt = args.apikey_txt.expanduser()

    if args.mode == "list-configs":
        for cfg in CONFIGS:
            print(json.dumps(cfg.__dict__, sort_keys=True))
        return 0

    if args.mode == "summarize":
        build_summary(args.output_root)
        return 0

    selected = CONFIGS if args.config_id == "all" else [CONFIG_BY_ID[args.config_id]]
    api_key = load_api_key(args.apikey_txt)
    for cfg in selected:
        run_config(cfg, args, api_key)
        build_summary(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
