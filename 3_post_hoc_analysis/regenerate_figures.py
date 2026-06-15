"""
Regenerate Figures 9 and 12 of the paper in English.

Figure 9: Standard deviation of learned weights across seeds (training
instability). Source: dynamic_weights_std.csv.

Figure 12: Mean learned weights per task (interpretability heatmap).
Source: dynamic_weights_mean.json.

The originals are rendered in Portuguese; this script reproduces them with
English labels and a cleaner layout for the paper resubmission.

Usage:
    python regenerate_figures.py \\
        --std_csv dynamic_weights_std.csv \\
        --mean_json dynamic_weights_mean.json \\
        --backbone "microsoft/deberta-v3-base" \\
        --out_dir figures_en
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Order of tasks to display (matches the paper)
TASK_ORDER_SENTEVAL = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]
TASK_ORDER_WITH_NLI = TASK_ORDER_SENTEVAL + ["NLI"]


# ----------------------------------------------------------------------------
# Figure 9: weight standard deviation across seeds
# ----------------------------------------------------------------------------

def plot_figure9(std_csv: Path, backbone: str, out_path: Path):
    df = pd.read_csv(std_csv, index_col=0)
    # Keep only the rows we want, in the paper's order
    rows = [t for t in TASK_ORDER_WITH_NLI if t in df.index]
    df = df.loc[rows]

    # Layer columns should already be L1..L12; ensure that order
    layer_cols = [c for c in df.columns if c.startswith("L")]
    layer_cols = sorted(layer_cols, key=lambda x: int(x[1:]))
    df = df[layer_cols]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        cbar_kws={"label": "Standard deviation"},
        annot_kws={"size": 11},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(
        f"Weight standard deviation across seeds (training instability) — {backbone}",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path} and {out_path.with_suffix('.pdf')}")


# ----------------------------------------------------------------------------
# Figure 12: mean learned weights per task
# ----------------------------------------------------------------------------

def plot_figure12(
    mean_json: Path,
    backbone: str,
    out_path: Path,
    include_aggregates: bool = True,
    include_nli: bool = True,
):
    """Render the mean-weights heatmap.

    The JSON contains per-task weights plus two aggregates (AVG_ALL,
    AVG_SENTEVAL). We render the same rows the paper currently shows.
    """
    with open(mean_json) as f:
        data = json.load(f)

    # Build display order
    rows = list(TASK_ORDER_SENTEVAL)
    if include_nli and "NLI" in data:
        rows.append("NLI")
    if include_aggregates:
        if "AVG_ALL" in data:
            rows.append("AVG_ALL")
        if "AVG_SENTEVAL" in data:
            rows.append("AVG_SENTEVAL")

    rows = [r for r in rows if r in data]
    matrix = np.array([data[r] for r in rows])
    n_layers = matrix.shape[1]
    layer_labels = [f"L{i+1}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Mean weight (softmax-normalized)"},
        annot_kws={"size": 11},
        linewidths=0.5,
        linecolor="white",
        xticklabels=layer_labels,
        yticklabels=rows,
        ax=ax,
    )
    ax.set_title(
        f"Mean learned layer weights — {backbone}",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Task / profile", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11, rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path} and {out_path.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std_csv", default="dynamic_weights_std.csv")
    ap.add_argument("--mean_json", default="dynamic_weights_mean.json")
    ap.add_argument("--backbone", default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", default="figures_en")
    ap.add_argument("--no_aggregates", action="store_true",
                    help="Omit AVG_ALL/AVG_SENTEVAL rows from Figure 12.")
    ap.add_argument("--no_nli", action="store_true",
                    help="Omit NLI row from Figure 12.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_figure9(
        Path(args.std_csv),
        args.backbone,
        out_dir / "fig9_weight_std_sbert.png",
    )
    plot_figure12(
        Path(args.mean_json),
        args.backbone,
        out_dir / "fig12_weight_mean.png",
        include_aggregates=not args.no_aggregates,
        include_nli=not args.no_nli,
    )

    # Also produce a "clean" Figure 12 version with only the SentEval tasks
    # (drops aggregates and NLI). Useful when the paper's narrative needs
    # focus on the 7 SentEval tasks.
    plot_figure12(
        Path(args.mean_json),
        args.backbone,
        out_dir / "fig12_weight_mean_senteval_only.png",
        include_aggregates=False,
        include_nli=False,
    )


if __name__ == "__main__":
    main()
