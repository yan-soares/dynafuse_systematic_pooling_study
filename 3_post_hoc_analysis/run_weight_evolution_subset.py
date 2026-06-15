"""
Orchestrator: run weight_evolution_analysis on a curated subset of configs.

For Figure-X (weight convergence across backbones), we do NOT want all 36
configurations. The narrative is "different backbones converge to different
layer-weight distributions, and the convergence is stable in all of them".
Four panels (one per backbone) with the recommended configuration suffice.

Default subset:
    - All 4 backbones (BERT, RoBERTa, DeBERTa, SBERT)
    - Weighted Average (the method recommended in the paper)
    - AVG training pooling (the recommended training pooling)

This script discovers the all_histories.json file for each backbone under
that fixed (fusion, pooling) configuration, then runs the per-config
analysis.

Usage:
    python run_weight_evolution_subset.py \\
        --root /path/to/train_scripts/results_classification_weights \\
        --fusion dynamic_layer_weighted_mean \\
        --pooling avg \\
        --out_root weight_evolution_per_backbone
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def discover_histories(root: Path, fusion: str, pooling: str) -> list:
    """Find all_histories.json files under results_<model>/<fusion>/<pooling>/."""
    rows = []
    for hist in root.rglob("all_histories.json"):
        parts = hist.relative_to(root).parts
        if len(parts) < 4:
            continue
        model_dir, found_fusion, found_pooling = parts[0], parts[1], parts[2]
        if found_fusion != fusion or found_pooling != pooling:
            continue
        model = re.sub(r"^results_", "", model_dir)
        rows.append({
            "histories": hist,
            "model": model,
            "fusion": found_fusion,
            "pooling": found_pooling,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fusion", default="dynamic_layer_weighted_mean")
    ap.add_argument("--pooling", default="avg")
    ap.add_argument("--out_root", default="weight_evolution_per_backbone")
    ap.add_argument("--script", default="weight_evolution_analysis.py")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = discover_histories(root, args.fusion, args.pooling)
    if not configs:
        print(f"No matching configs found for fusion={args.fusion} "
              f"pooling={args.pooling}")
        return

    print(f"Discovered {len(configs)} configurations:")
    for c in configs:
        print(f"  {c['model']}: {c['histories']}")

    for c in configs:
        out_dir = out_root / c["model"]
        if (out_dir / "convergence_grid.png").exists():
            print(f"SKIP existing {c['model']}")
            continue
        print(f"\n>>> Running for {c['model']}")
        cmd = [
            sys.executable, args.script,
            "--histories", str(c["histories"]),
            "--out_dir", str(out_dir),
            "--tol", str(args.tol),
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"FAILED: {c['model']} (exit code {rc})")


if __name__ == "__main__":
    main()
