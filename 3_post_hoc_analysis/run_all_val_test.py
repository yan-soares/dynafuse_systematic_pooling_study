"""
Orchestrator: run val_test_gap_analysis across all (model, fusion, pooling)
combinations.

Expected directory layout:
    <root>/results_<model>/<fusion_technique>/<pooling>/
        cl_results_<fusion_technique>-<pooling>.csv

Examples:
    results_microsoft_deberta-v3-base/dynamic_layer_weighted_mean/avg/
        cl_results_dynamic_layer_weighted_mean-avg.csv

This script auto-discovers every CSV under <root> matching the pattern
'cl_results_*.csv', infers (model, fusion, pooling) from the path, and
runs val_test_gap_analysis on each one.

Output: <out_root>/<model>__<fusion>__<pooling>/{long_form.csv, summary_gap.csv, top_10.csv, ...}

Usage:
    python run_all_val_test.py \\
        --root /path/to/train_scripts/results_classification_weights \\
        --out_root all_val_test_results
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def discover_csvs(root: Path) -> list:
    """Find all cl_results_*.csv files and parse their (model, fusion, pooling)."""
    rows = []
    for csv in root.rglob("cl_results_*.csv"):
        # Skip auxiliary outputs (_processado_acc, _processado_devacc, etc.)
        if "_processado" in csv.name or "_log" in csv.name:
            continue
        parts = csv.relative_to(root).parts
        # Expected: results_<model>/<fusion>/<pooling>/cl_results_*.csv
        if len(parts) < 4:
            print(f"WARN: unexpected depth for {csv}, skipping")
            continue
        model_dir, fusion, pooling = parts[0], parts[1], parts[2]
        # Strip the 'results_' prefix from model directory
        model = re.sub(r"^results_", "", model_dir)
        rows.append({
            "csv": csv,
            "model": model,
            "fusion": fusion,
            "pooling": pooling,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root containing results_<model>/ subdirs")
    ap.add_argument("--out_root", default="all_val_test_results")
    ap.add_argument("--script", default="val_test_gap_analysis.py",
                    help="Path to the per-config script")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only list discovered configs, do not execute")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = discover_csvs(root)
    if not configs:
        print(f"No CSVs found under {root}")
        return

    print(f"Discovered {len(configs)} configurations:")
    for c in configs:
        print(f"  {c['model']} | {c['fusion']} | {c['pooling']}")

    if args.dry_run:
        return

    failures = []
    for c in configs:
        tag = f"{c['model']}__{c['fusion']}__{c['pooling']}"
        out_dir = out_root / tag
        if (out_dir / "summary_gap.csv").exists():
            print(f"SKIP existing {tag}")
            continue
        print(f"\n>>> Running {tag}")
        cmd = [
            sys.executable, args.script,
            "--csv", str(c["csv"]),
            "--out_dir", str(out_dir),
            "--top_k", "10",
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            failures.append(tag)
            print(f"FAILED: {tag} (exit code {rc})")

    print(f"\nDone. {len(configs) - len(failures)} succeeded, "
          f"{len(failures)} failed.")
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")


if __name__ == "__main__":
    main()
