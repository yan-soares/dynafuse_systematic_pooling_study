"""
Extract per-split inner-CV scores from SentEval evaluation logs.

The SentEval toolkit runs cross-validation per transfer task. We parse the
log file using the actual structure observed:

    Per CONFIG (1 of N inside the log), the toolkit evaluates 7 tasks in this
    fixed order:
        MR    -> 10 'Best param found at split X' lines
        CR    -> 10 'Best param found at split X' lines
        SUBJ  -> 10 'Best param found at split X' lines
        MPQA  -> 10 'Best param found at split X' lines
        SST2  -> ONE 'Validation : best param found' line (fixed train/dev/test)
        TREC  -> 'Transfer task : TREC' + ONE 'Cross-validation : best param' line
        MRPC  -> 'Transfer task : MRPC' + ONE 'Cross-validation : best param' line

We therefore identify task boundaries by counting the consecutive blocks of
10 'Best param at split' lines (= the first 4 tasks), then advance through the
explicit markers for the rest.

Configs are detected by the cumulative count of tasks (every 7 tasks starts
a new config).

Input: cl_results_*_log.txt files under <root>/results_<model>/<fusion>/<pooling>/
Output: per-task split arrays + summary CSV + per-task variability stats

Note: This is NOT a substitute for proper paired statistical tests across
seeds. It quantifies the uncertainty of the single SentEval evaluation,
which is a different (weaker) question than 'is DynaFuse significantly
better than baseline X'. Use the seeded re-run for the latter.

Usage:
    python extract_inner_cv_from_logs.py \\
        --root /path/to/train_scripts/results_classification_weights \\
        --out_dir inner_cv_extraction
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# SentEval evaluation order, observed empirically from the logs
TASK_ORDER = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]
TASKS_PER_CONFIG = len(TASK_ORDER)
SPLITS_PER_KFOLD_TASK = 10


# Patterns
RE_SPLIT_SCORE = re.compile(
    r"Best param found at split (\d+): l2reg = \S+\s+with score\s+([\d.]+)"
)
RE_VALIDATION_SCORE = re.compile(
    r"Validation : best param found is reg = \S+ with score\s+([\d.]+)"
)
RE_CV_SCORE = re.compile(
    r"Cross-validation : best param found is reg = \S+\s+with score\s+([\d.]+)"
)


def parse_log(log_path: Path) -> dict:
    """Parse one SentEval log file using the observed task structure.

    Returns:
        {
            'task_splits':  {(config_idx, task): list[float]},
            'n_configs':    int,
        }
    """
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()

    # We walk linearly, accumulating split scores. A "block end" is detected
    # either when split numbering resets (new split=1 after split>1) OR when
    # a SST2/TREC/MRPC summary line appears. Each block end advances the
    # task index.

    task_splits = defaultdict(list)  # (cfg_idx, task) -> [scores]
    current_block = []         # accumulating scores for current task
    last_split_id = 0          # the split id seen most recently
    global_task_idx = 0        # how many tasks have been completed in total

    def finish_block(task_idx_at_finish):
        nonlocal current_block
        if not current_block:
            return
        cfg = task_idx_at_finish // TASKS_PER_CONFIG
        local_task = task_idx_at_finish % TASKS_PER_CONFIG
        task_name = TASK_ORDER[local_task]
        task_splits[(cfg, task_name)].extend(current_block)
        current_block = []

    for line in lines:
        m = RE_SPLIT_SCORE.search(line)
        if m:
            split_id = int(m.group(1))
            score = float(m.group(2))
            # If we observe split=1 after we already had a higher split id,
            # it means the previous task block has ended.
            if split_id == 1 and last_split_id > 0:
                finish_block(global_task_idx)
                global_task_idx += 1
            current_block.append(score)
            last_split_id = split_id
            continue

        # SST2 single-validation case: one number, treated as a single-split task
        m = RE_VALIDATION_SCORE.search(line)
        if m:
            # First close any pending KFold block
            if current_block:
                finish_block(global_task_idx)
                global_task_idx += 1
                last_split_id = 0
            score = float(m.group(1))
            current_block = [score]
            finish_block(global_task_idx)
            global_task_idx += 1
            last_split_id = 0
            continue

        # TREC/MRPC: 'Cross-validation : best param found ... with score X'
        m = RE_CV_SCORE.search(line)
        if m:
            if current_block:
                finish_block(global_task_idx)
                global_task_idx += 1
                last_split_id = 0
            score = float(m.group(1))
            current_block = [score]
            finish_block(global_task_idx)
            global_task_idx += 1
            last_split_id = 0
            continue

    # Flush any trailing block
    if current_block:
        finish_block(global_task_idx)
        global_task_idx += 1

    n_configs = (global_task_idx + TASKS_PER_CONFIG - 1) // TASKS_PER_CONFIG
    return {"task_splits": dict(task_splits), "n_configs": n_configs}


def parse_csv_for_config_names(csv_path: Path) -> list:
    """Read the cl_results_*.csv and return the ordered list of 'pooling' values."""
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    return df["pooling"].tolist()


def discover_logs(root: Path) -> list:
    out = []
    for log in root.rglob("cl_results_*_log.txt"):
        parts = log.relative_to(root).parts
        if len(parts) < 4:
            continue
        model_dir, fusion, pooling = parts[0], parts[1], parts[2]
        model = re.sub(r"^results_", "", model_dir)
        out.append({
            "log": log,
            "csv": log.parent / log.name.replace("_log.txt", ".csv"),
            "model": model,
            "fusion": fusion,
            "pooling": pooling,
        })
    return out


def build_summary(logs: list) -> pd.DataFrame:
    """Aggregate split-level scores across all logs into one long-form frame."""
    rows = []
    for item in logs:
        parsed = parse_log(item["log"])
        config_names = parse_csv_for_config_names(item["csv"])
        for (cfg_idx, task), scores in parsed["task_splits"].items():
            cfg_name = config_names[cfg_idx] if cfg_idx < len(config_names) else f"cfg_{cfg_idx}"
            arr = np.array(scores, dtype=float)
            rows.append({
                "model":   item["model"],
                "fusion":  item["fusion"],
                "pooling_train": item["pooling"],
                "config_name":   cfg_name,
                "task":          task,
                "n_splits":      len(arr),
                "mean":          float(arr.mean()),
                "std":           float(arr.std()) if len(arr) > 1 else 0.0,
                "min":           float(arr.min()),
                "max":           float(arr.max()),
                "scores":        arr.tolist(),
            })
    return pd.DataFrame(rows)


def write_summary_table(df: pd.DataFrame, out_path: Path):
    """Mean of inner-CV std per (model, task), across configs."""
    # Only tasks with real per-split variability (KFold tasks: MR, CR, SUBJ, MPQA)
    kfold = df[df["n_splits"] >= 10].copy()
    if kfold.empty:
        print("WARN: no KFold-style tasks parsed")
        return
    overall = (
        kfold.groupby(["model", "task"])
        .agg(mean_std=("std", "mean"),
             mean_score=("mean", "mean"),
             n_configs=("std", "count"))
        .reset_index()
        .sort_values(["model", "task"])
    )
    overall.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", default="inner_cv_extraction")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = discover_logs(Path(args.root))
    if not logs:
        print(f"No logs found under {args.root}")
        sys.exit(1)
    print(f"Discovered {len(logs)} log files.")

    df = build_summary(logs)
    df.to_csv(out_dir / "long_form.csv", index=False)
    print(f"wrote long_form.csv ({len(df)} rows)")

    write_summary_table(df, out_dir / "per_model_task.csv")

    # Print top-level number for the response letter
    kfold = df[df["n_splits"] >= 10]
    print("\n=== Inner-CV split variability (KFold tasks only) ===")
    print(f"(MR, CR, SUBJ, MPQA — these are the tasks with 10 inner splits each)")
    print(f"\nAcross {len(kfold)} (config, task) pairs:")
    print(f"  Mean inner-CV std across splits: {kfold['std'].mean():.3f}")
    print(f"  Median inner-CV std:             {kfold['std'].median():.3f}")
    print(f"  Max inner-CV std observed:       {kfold['std'].max():.3f}")
    print(f"  Configs covered:                 "
          f"{kfold.groupby(['model','fusion','pooling_train']).ngroups}")

    print(f"\nPer-task average std:")
    for task in ["MR", "CR", "SUBJ", "MPQA"]:
        sub = kfold[kfold["task"] == task]
        if len(sub) > 0:
            print(f"  {task}: mean_std = {sub['std'].mean():.3f}, "
                  f"mean_score = {sub['mean'].mean():.2f} ({len(sub)} configs)")


if __name__ == "__main__":
    main()

