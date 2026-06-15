"""
Build the combined Figure 12 of the paper: average learned weights per
backbone, stacked into a single heatmap (one row per backbone).

The original paper Figure 12 shows BERT, DeBERTa, RoBERTa, SBERT in one
heatmap, with each backbone occupying one row and the IN-TASK profile
weights averaged across the 7 SentEval tasks.

This script reads the per-backbone dynamic_weights_mean.json files
(produced by the training pipeline for fusion=weighted_mean, pooling=avg)
and stitches them together.

Usage:
    python build_combined_figure12.py \\
        --root /path/to/train_scripts/results_classification_weights \\
        --fusion dynamic_layer_weighted_mean \\
        --pooling avg \\
        --out_path fig12_combined.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SENTEVAL_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]


# Display names: filesystem dir -> paper label
MODEL_DISPLAY = {
    "google-bert_bert-base-uncased":       "BERT",
    "bert-base-uncased":                   "BERT",
    "FacebookAI_roberta-base":             "RoBERTa",
    "roberta-base":                        "RoBERTa",
    "microsoft_deberta-v3-base":           "DeBERTa",
    "microsoft_deberta-v3-base-base":      "DeBERTa",  # in case of typo
    "sentence-transformers_all-mpnet-base-v2": "SBERT",
    "all-mpnet-base-v2":                   "SBERT",
}

# Order to display
DISPLAY_ORDER = ["BERT", "DeBERTa", "RoBERTa", "SBERT"]


def discover_weight_files(root: Path, fusion: str, pooling: str) -> dict:
    """Return {model_display_name: path_to_dynamic_weights_mean.json}."""
    out = {}
    for jf in root.rglob("dynamic_weights_mean.json"):
        parts = jf.relative_to(root).parts
        if len(parts) < 4:
            continue
        model_dir, found_fusion, found_pooling = parts[0], parts[1], parts[2]
        if found_fusion != fusion or found_pooling != pooling:
            continue
        model_key = re.sub(r"^results_", "", model_dir)
        display = MODEL_DISPLAY.get(model_key, model_key)
        out[display] = jf
    return out


def average_senteval_rows(data: dict) -> np.ndarray:
    """Given the per-task weight dict, return the average across SentEval tasks."""
    rows = []
    for t in SENTEVAL_TASKS:
        if t in data:
            rows.append(np.array(data[t]))
    if not rows:
        raise ValueError("No SentEval task keys found in weights file")
    return np.mean(np.stack(rows, axis=0), axis=0)


def build_figure(weight_files: dict, out_path: Path):
    matrix_rows = []
    labels = []
    for display in DISPLAY_ORDER:
        if display not in weight_files:
            print(f"WARN: missing {display}")
            continue
        with open(weight_files[display]) as f:
            data = json.load(f)
        # Prefer the AVG_SENTEVAL aggregate if present, else compute it
        if "AVG_SENTEVAL" in data:
            row = np.array(data["AVG_SENTEVAL"])
        else:
            row = average_senteval_rows(data)
        matrix_rows.append(row)
        labels.append(display)

    matrix = np.stack(matrix_rows, axis=0)
    n_layers = matrix.shape[1]
    layer_cols = [f"L{i+1}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(
        matrix,
        annot=True, fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Mean weight (softmax-normalized)"},
        annot_kws={"size": 11},
        linewidths=0.5, linecolor="white",
        xticklabels=layer_cols,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        "Mean learned layer weights per backbone "
        "(averaged over 7 SentEval tasks)",
        fontsize=12, pad=10,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path} and {out_path.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fusion", default="dynamic_layer_weighted_mean")
    ap.add_argument("--pooling", default="avg")
    ap.add_argument("--out_path", default="fig12_combined.png")
    args = ap.parse_args()

    files = discover_weight_files(Path(args.root), args.fusion, args.pooling)
    if not files:
        print(f"No weight files found. Check root, fusion, pooling.")
        return
    print(f"Found {len(files)} backbone(s): {list(files.keys())}")
    build_figure(files, Path(args.out_path))


if __name__ == "__main__":
    main()
