#!/usr/bin/env python3
"""Re-run judge scoring on already-completed dense config runs.

Reads each D*/run_evals_return.json, calls call_judge on the stored answers,
and writes updated generation metrics back into the same file.

Usage:
    export OPENAI_API_KEY="$(cat ~/api-key.txt)"
    export JUDGE_BASE_URL="https://tritonai-api.ucsd.edu/v1"
    export JUDGE_MODEL="claude-sonnet-4-6"
    python3 rejudge_dense.py [--experiments-path rapidfire_experiments/dense_all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "Metrics"

_metrics_str = str(METRICS_DIR)
if _metrics_str not in sys.path:
    sys.path.insert(0, _metrics_str)
os.environ["PYTHONPATH"] = (
    _metrics_str + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
)

from project1_eval import call_judge, f1_at_k, precision_at_k, recall_at_k  # noqa: E402

if not os.environ.get("JUDGE_BASE_URL"):
    os.environ["JUDGE_BASE_URL"] = "https://tritonai-api.ucsd.edu/v1"
if not os.environ.get("JUDGE_MODEL"):
    os.environ["JUDGE_MODEL"] = "claude-sonnet-4-6"


def rejudge_config(result_path: Path, dry_run: bool = False) -> dict:
    data = json.loads(result_path.read_text())

    # Structure: {shard_id: [data_batch, metrics_batch]}
    shard_key = list(data.keys())[0]
    batches = data[shard_key]

    # Find the data batch (has question_id list) and metrics batch (has Judge Failures)
    data_batch = next((b for b in batches if b.get("question_id")), None)
    metrics_batch_idx = next((i for i, b in enumerate(batches) if "Judge Failures" in b), None)

    if data_batch is None:
        print(f"  SKIP: no data batch found in {result_path}")
        return {}

    questions = data_batch["query"]
    refs = data_batch["reference_answer"]
    contexts = data_batch["serialized_context"]
    answers = data_batch.get("generated_text") or data_batch.get("answer")
    n = len(questions)

    model = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
    base_url = os.environ.get("JUDGE_BASE_URL") or None

    corr, faith, comp, failures = [], [], [], 0
    for i, (q, ref, ctx, ans) in enumerate(zip(questions, refs, contexts, answers)):
        print(f"  [{i+1}/{n}] judging q_id={data_batch['question_id'][i]} ...", end=" ", flush=True)
        if dry_run:
            print("DRY RUN")
            corr.append(1.0); faith.append(1.0); comp.append(0.6)
            continue
        r = call_judge(q, ref, ctx, ans, model=model, base_url=base_url)
        if r.get("failed"):
            failures += 1
            print(f"FAIL: {r.get('error', '')[:80]}")
        else:
            print(f"C={int(r['correctness'])} F={int(r['faithfulness'])} Comp={int(r['completeness'])}")
        corr.append(r["correctness"])
        faith.append(r["faithfulness"])
        comp.append(r["completeness"] / 5.0)

    mean_c  = sum(corr) / n
    mean_fa = sum(faith) / n
    mean_cp = sum(comp) / n
    gen3    = (mean_c + mean_fa + mean_cp) / 3

    # Recompute retrieval metrics too (for consistency)
    f1s = [f1_at_k([tuple(s) for s in r], [tuple(s) for s in g])
           for r, g in zip(data_batch["retrieved_spans"], data_batch["ground_truth_spans"])]
    ps  = [precision_at_k([tuple(s) for s in r], [tuple(s) for s in g])
           for r, g in zip(data_batch["retrieved_spans"], data_batch["ground_truth_spans"])]
    rs  = [recall_at_k([tuple(s) for s in r], [tuple(s) for s in g])
           for r, g in zip(data_batch["retrieved_spans"], data_batch["ground_truth_spans"])]
    mean_f1 = sum(f1s) / n
    mean_p  = sum(ps) / n
    mean_r  = sum(rs) / n
    ret_score = (mean_f1 + mean_p + mean_r) / 3

    partial = 0.5 * ret_score + 0.5 * gen3

    new_metrics = {
        "F1_at_5":                    {"value": mean_f1,  "is_algebraic": True, "value_range": [0, 1]},
        "Precision_at_5":             {"value": mean_p,   "is_algebraic": True, "value_range": [0, 1]},
        "Recall_at_5":                {"value": mean_r,   "is_algebraic": True, "value_range": [0, 1]},
        "Retrieval Score":            {"value": ret_score,"is_algebraic": True, "value_range": [0, 1]},
        "Correctness_pass_rate":      {"value": mean_c,   "is_algebraic": True, "value_range": [0, 1]},
        "Faithfulness_pass_rate":     {"value": mean_fa,  "is_algebraic": True, "value_range": [0, 1]},
        "Completeness_normalized":    {"value": mean_cp,  "is_algebraic": True, "value_range": [0, 1]},
        "Generation_Score_3_released":{"value": gen3,     "is_algebraic": True, "value_range": [0, 1]},
        "Judge Failures":             {"value": float(failures), "is_distributive": True},
        "Partial_Overall_Score":      {"value": partial,  "is_algebraic": True, "value_range": [0, 1]},
    }

    # Update metrics batch in-place (preserving config fields like chunk_size etc.)
    if metrics_batch_idx is not None:
        data[shard_key][metrics_batch_idx].update(new_metrics)
    else:
        # No existing metrics batch — append one
        data[shard_key].append(new_metrics)

    if not dry_run:
        result_path.write_text(json.dumps(data, indent=2))

    summary = {
        "F1": mean_f1, "P": mean_p, "R": mean_r,
        "RetScore": ret_score,
        "Correctness": mean_c, "Faithfulness": mean_fa,
        "Completeness_norm": mean_cp, "Gen3": gen3,
        "Partial": partial, "JudgeFailures": failures,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-path", default="rapidfire_experiments/dense_all")
    parser.add_argument("--configs", nargs="*", help="e.g. D01 D03 (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.experiments_path)
    if not exp_dir.is_absolute():
        exp_dir = ROOT / exp_dir

    # Collect config dirs
    all_dirs = sorted(d for d in exp_dir.iterdir()
                      if d.is_dir() and d.name[:1] == "D")
    if args.configs:
        prefix_set = set(args.configs)
        all_dirs = [d for d in all_dirs if any(d.name.startswith(p) for p in prefix_set)]

    results = {}
    for cfg_dir in all_dirs:
        result_file = cfg_dir / "run_evals_return.json"
        if not result_file.exists():
            print(f"\n{cfg_dir.name}: no run_evals_return.json, skipping")
            continue

        cfg_id = cfg_dir.name.split("_")[0]
        print(f"\n{'='*60}")
        print(f"{cfg_id}  ({cfg_dir.name})")
        t0 = time.time()
        summary = rejudge_config(result_file, dry_run=args.dry_run)
        elapsed = time.time() - t0
        if summary:
            results[cfg_id] = summary
            print(f"  RetScore={summary['RetScore']:.4f}  "
                  f"Gen3={summary['Gen3']:.4f}  "
                  f"Partial={summary['Partial']:.4f}  "
                  f"JudgeFail={summary['JudgeFailures']}  "
                  f"({elapsed:.0f}s)")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Config':<6}  {'RetScore':>9}  {'Correct':>7}  {'Faith':>7}  {'Comp':>7}  {'Gen3':>7}  {'Partial':>9}  {'JFail':>5}")
    print("-" * 70)
    for cfg_id, s in sorted(results.items()):
        print(f"{cfg_id:<6}  {s['RetScore']:>9.4f}  {s['Correctness']:>7.4f}  "
              f"{s['Faithfulness']:>7.4f}  {s['Completeness_norm']:>7.4f}  "
              f"{s['Gen3']:>7.4f}  {s['Partial']:>9.4f}  {s['JudgeFailures']:>5}")

    # Save summary CSV
    if results and not args.dry_run:
        csv_path = exp_dir / "judge_results.csv"
        with open(csv_path, "w") as f:
            f.write("config_id,RetScore,F1,Precision,Recall,Correctness,Faithfulness,Completeness_norm,Gen3,Partial,JudgeFailures\n")
            for cfg_id, s in sorted(results.items()):
                f.write(f"{cfg_id},{s['RetScore']:.6f},{s['F1']:.6f},{s['P']:.6f},{s['R']:.6f},"
                        f"{s['Correctness']:.6f},{s['Faithfulness']:.6f},{s['Completeness_norm']:.6f},"
                        f"{s['Gen3']:.6f},{s['Partial']:.6f},{s['JudgeFailures']}\n")
        print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
