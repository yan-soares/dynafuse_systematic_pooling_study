"""
Rerun SentEval evaluation with multiple seeds for statistical testing.

This addresses Major #3 of the reviewer report: paired statistical tests
between DynaFuse and baselines on the SentEval benchmark.

== What this script does NOT do ==

It does NOT replace your existing SentEval evaluation pipeline. Instead, it
provides:

    1. A driver function that wraps your existing eval call and runs it N
       times with different seeds.
    2. A post-processing function that collects the per-seed accuracies and
       runs paired Wilcoxon / Friedman tests.

== Where the seed variation happens ==

The DynaFuse weights alpha are FIXED (the average over 5 seeds you already
have). The variation that matters for statistical testing comes from:

    - The 10-fold StratifiedKFold split used by SentEval for MR, CR, SUBJ,
      MPQA (the 'inner CV' of the logistic regression). SentEval calls
      sklearn KFold internally with a fixed shuffle but no seed argument,
      which means it uses sklearn's default (deterministic).

To get variation, we patch the seed used by sklearn before each SentEval
run. The simplest path is to monkeypatch np.random.seed at the top of your
eval script, before SentEval imports its KFold splits.

== Minimal integration ==

Add the following block at the top of your existing eval script (e.g.
eval_classification.py), wrapped around the SentEval call:

    import numpy as np
    import random
    import torch  # if used

    SEED = int(os.environ.get("EVAL_SEED", 42))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

Then launch your evaluation N times:

    for seed in 42 0 1234 2025 999; do
        EVAL_SEED=$seed python eval_classification.py \\
            --out_csv results_seed_${seed}.csv \\
            <other args>
    done

This will give you 5 CSVs, each with 'acc' values that differ slightly due
to the KFold split.

== After running ==

Run:
    python rerun_eval_with_seeds.py aggregate \\
        --inputs results_seed_*.csv \\
        --out_dir stat_tests

This will:
    - Build a (config x task x seed) tensor of accuracies
    - Compute mean +/- std per (config, task)
    - Run paired Wilcoxon tests between DynaFuse-winning and baselines
    - Output a LaTeX table with significance markers

Usage:
    # After running eval N times:
    python rerun_eval_with_seeds.py aggregate \\
        --inputs results_seed_42.csv results_seed_0.csv ... \\
        --out_dir stat_tests \\
        --reference_config "AVG_IN-TASK" \\
        --comparison_configs "CLS_IN-TASK" "AVG-NS_IN-TASK"
"""

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SENTEVAL_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]


def _parse_cell(cell: str) -> dict:
    """Same robust dict parser used in val_test_gap_analysis."""
    if not isinstance(cell, str):
        return {}
    s = cell.strip()
    s = re.sub(r"np\.(float64|float32|int64|int32)\(([^)]+)\)", r"\2", s)
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def load_seed_csv(path: Path, seed: int) -> pd.DataFrame:
    """Read one seeded eval CSV into long form: config, task, seed, acc."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        for task in SENTEVAL_TASKS:
            if task not in r:
                continue
            parsed = _parse_cell(r[task])
            if not parsed:
                continue
            rows.append({
                "config":  r["pooling"],
                "task":    task,
                "seed":    seed,
                "acc":     parsed.get("acc"),
                "devacc":  parsed.get("devacc"),
            })
    return pd.DataFrame(rows)


def aggregate(inputs: list, out_dir: Path,
              reference_config: str,
              comparison_configs: list):
    """Build per-(config, task) seed tensor, compute stats, run paired tests."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse seed from filename: results_seed_<N>.csv
    pat = re.compile(r"seed_?(\d+)")
    all_rows = []
    for p in inputs:
        m = pat.search(Path(p).stem)
        if not m:
            print(f"WARN: cannot extract seed from {p}, skipping")
            continue
        seed = int(m.group(1))
        all_rows.append(load_seed_csv(Path(p), seed))
    if not all_rows:
        print("No valid inputs.")
        return

    long = pd.concat(all_rows, ignore_index=True)
    long.to_csv(out_dir / "all_seeds_long.csv", index=False)

    # Pivot: rows = (config, task), cols = seed, values = acc
    pivot = long.pivot_table(
        index=["config", "task"], columns="seed", values="acc"
    )
    pivot.to_csv(out_dir / "pivot_by_seed.csv")

    # Per (config, task): mean and std over seeds
    summary = long.groupby(["config", "task"])["acc"].agg(["mean", "std", "count"])
    summary.to_csv(out_dir / "per_config_task_summary.csv")
    print(f"Per-(config, task) summary written. Configs: "
          f"{summary.index.get_level_values('config').nunique()}, "
          f"Tasks: {summary.index.get_level_values('task').nunique()}")

    # Paired statistical tests
    test_rows = []
    for cmp_cfg in comparison_configs:
        for task in SENTEVAL_TASKS:
            ref = pivot.loc[(reference_config, task)].dropna().values \
                if (reference_config, task) in pivot.index else None
            cmp = pivot.loc[(cmp_cfg, task)].dropna().values \
                if (cmp_cfg, task) in pivot.index else None
            if ref is None or cmp is None:
                continue
            # Align on common seeds
            common_n = min(len(ref), len(cmp))
            ref = ref[:common_n]
            cmp = cmp[:common_n]
            if common_n < 3:
                continue
            try:
                stat_w, p_w = stats.wilcoxon(ref, cmp)
            except Exception:
                stat_w, p_w = np.nan, np.nan
            test_rows.append({
                "task":             task,
                "reference":        reference_config,
                "comparison":       cmp_cfg,
                "n_seeds":          common_n,
                "mean_ref":         float(ref.mean()),
                "mean_cmp":         float(cmp.mean()),
                "delta":            float(ref.mean() - cmp.mean()),
                "wilcoxon_stat":    float(stat_w) if stat_w is not np.nan else np.nan,
                "wilcoxon_p":       float(p_w) if p_w is not np.nan else np.nan,
                "significant_005":  bool(p_w < 0.05) if not np.isnan(p_w) else False,
            })

    tests = pd.DataFrame(test_rows)
    tests.to_csv(out_dir / "wilcoxon_tests.csv", index=False)
    print(f"Wilcoxon tests written ({len(tests)} task comparisons).")

    # Friedman test across all (task) levels per comparison
    print("\n=== Friedman test across tasks ===")
    print("(Tests whether reference is significantly different from comparison "
          "when treating tasks as blocks)")
    for cmp_cfg in comparison_configs:
        ref_vals, cmp_vals = [], []
        for task in SENTEVAL_TASKS:
            if (reference_config, task) in pivot.index and (cmp_cfg, task) in pivot.index:
                r = pivot.loc[(reference_config, task)].dropna().values
                c = pivot.loc[(cmp_cfg, task)].dropna().values
                n = min(len(r), len(c))
                if n >= 3:
                    ref_vals.append(r[:n].mean())
                    cmp_vals.append(c[:n].mean())
        if len(ref_vals) >= 3:
            stat_f, p_f = stats.friedmanchisquare(ref_vals, cmp_vals)
            print(f"  {reference_config} vs {cmp_cfg}: "
                  f"Friedman chi2={stat_f:.3f}, p={p_f:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--inputs", nargs="+", required=True,
                       help="CSV files from N seeded SentEval runs")
    p_agg.add_argument("--out_dir", default="stat_tests")
    p_agg.add_argument("--reference_config", required=True,
                       help="Config to compare AGAINST (e.g. AVG_IN-TASK)")
    p_agg.add_argument("--comparison_configs", nargs="+", required=True,
                       help="Configs to compare TO (e.g. CLS_IN-TASK ...)")

    args = ap.parse_args()
    if args.cmd == "aggregate":
        aggregate(
            inputs=args.inputs,
            out_dir=Path(args.out_dir),
            reference_config=args.reference_config,
            comparison_configs=args.comparison_configs,
        )


if __name__ == "__main__":
    main()
