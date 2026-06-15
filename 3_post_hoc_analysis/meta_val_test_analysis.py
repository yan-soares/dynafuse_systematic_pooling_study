"""
Meta-analysis: consolidate val-test gap results across all (model, fusion,
pooling) combinations into a single appendix table for the paper.

After running run_all_val_test.py, this script reads every summary_gap.csv
under <results_root>/<tag>/ and produces:

    1. meta_summary.csv: one row per (model, fusion, pooling) with
       mean_dev, mean_test, mean_gap, std_gap, % configs with positive gap.

    2. meta_table.tex: a compact LaTeX table for the appendix grouped by
       model and fusion technique.

    3. meta_overall.png: bar chart of mean gap per (model, fusion) for
       quick visual inspection.

    4. meta_overall_stats.txt: top-level numbers for the response letter:
       'across N configurations, the mean test-val gap is +X.XX +/- Y.YY,
       and the gap is positive in Z% of configurations'.

Usage:
    python meta_val_test_analysis.py \\
        --results_root all_val_test_results \\
        --out_dir meta_val_test
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_tag(tag: str):
    """tag = '<model>__<fusion>__<pooling>'."""
    parts = tag.split("__")
    if len(parts) != 3:
        return None
    return {"model": parts[0], "fusion": parts[1], "pooling": parts[2]}


def collect(results_root: Path) -> pd.DataFrame:
    """Aggregate per-config summary_gap.csv files into one frame."""
    rows = []
    for tag_dir in sorted(results_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        summary = tag_dir / "summary_gap.csv"
        if not summary.exists():
            continue
        meta = parse_tag(tag_dir.name)
        if meta is None:
            continue
        df = pd.read_csv(summary)
        # df has columns: pooling, profile, mean_dev, mean_test, mean_gap, std_gap, n
        # Compute one summary number per config (model, fusion, pooling_train)
        rows.append({
            **meta,
            "n_configs": len(df),
            "mean_dev":      df["mean_dev"].mean(),
            "mean_test":     df["mean_test"].mean(),
            "mean_gap":      df["mean_gap"].mean(),
            "std_gap":       df["mean_gap"].std(),
            "min_gap":       df["mean_gap"].min(),
            "max_gap":       df["mean_gap"].max(),
            "pct_positive":  (df["mean_gap"] > 0).mean() * 100,
            # the row with best test acc inside this config
            "best_test":     df["mean_test"].max(),
            "best_test_pooling": df.loc[df["mean_test"].idxmax(), "pooling"],
            "best_test_profile": df.loc[df["mean_test"].idxmax(), "profile"],
        })
    return pd.DataFrame(rows)


def write_overall_stats(df: pd.DataFrame, out_path: Path):
    """Top-level numbers ready to drop into the response letter."""
    lines = []
    lines.append("=" * 60)
    lines.append("Meta val-test gap analysis (Major #7)")
    lines.append("=" * 60)
    lines.append(f"Configurations analyzed: {len(df)}")
    lines.append(f"Backbones: {df['model'].nunique()}")
    lines.append(f"Fusion techniques: {df['fusion'].nunique()}")
    lines.append(f"Training poolings: {df['pooling'].nunique()}")
    lines.append("")
    lines.append(f"Mean (test - val) gap: {df['mean_gap'].mean():+.3f}")
    lines.append(f"Std of mean gap across configs: {df['mean_gap'].std():.3f}")
    lines.append(f"Min mean gap: {df['mean_gap'].min():+.3f}")
    lines.append(f"Max mean gap: {df['mean_gap'].max():+.3f}")
    lines.append("")
    pos = (df['mean_gap'] > 0).sum()
    lines.append(f"Configs with positive mean gap: {pos}/{len(df)} "
                 f"({pos/len(df)*100:.1f}%)")
    lines.append(f"Mean fraction of (pool, profile) pairs with positive gap, "
                 f"averaged over configs: {df['pct_positive'].mean():.1f}%")
    txt = "\n".join(lines)
    out_path.write_text(txt)
    print(txt)


def plot_gap_overview(df: pd.DataFrame, out_path: Path):
    """Bar chart of mean gap per (model, fusion), with pooling as hue."""
    pretty_fusion = {
        "dynamic_layer_weighted_mean":  "Weighted Avg",
        "dynamic_layer_weighted_sum":   "Weighted Sum",
        "dynamic_layer_cnn":            "1D-CNN",
        "dynamic_layer_1d_cnn":         "1D-CNN",
    }
    df = df.copy()
    df["fusion_label"] = df["fusion"].map(lambda x: pretty_fusion.get(x, x))

    models = sorted(df["model"].unique())
    poolings = sorted(df["pooling"].unique())
    fusions = sorted(df["fusion_label"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    width = 0.8 / max(len(poolings), 1)
    cmap = plt.get_cmap("tab10")

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        x = np.arange(len(fusions))
        for i, pool in enumerate(poolings):
            ssub = sub[sub["pooling"] == pool].set_index("fusion_label")
            ssub = ssub.reindex(fusions)
            vals = ssub["mean_gap"].fillna(0).values
            errs = ssub["std_gap"].fillna(0).values
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   yerr=errs, capsize=3,
                   label=pool, color=cmap(i % 10), alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fusions, rotation=15)
        ax.set_title(model, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Mean test - val gap (acc points)")
    axes[-1].legend(title="Train pooling", loc="best", fontsize=9)
    fig.suptitle("Test - validation gap across all configurations "
                 "(positive = no overfit on selection)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def to_latex_compact(df: pd.DataFrame, out_path: Path):
    """LaTeX table grouped by model and fusion, showing mean gap and pos%."""
    pretty_fusion = {
        "dynamic_layer_weighted_mean":  "W-Avg",
        "dynamic_layer_weighted_sum":   "W-Sum",
        "dynamic_layer_cnn":            "1D-CNN",
        "dynamic_layer_1d_cnn":         "1D-CNN",
    }
    df = df.copy()
    df["fusion_label"] = df["fusion"].map(lambda x: pretty_fusion.get(x, x))
    df = df.sort_values(["model", "fusion_label", "pooling"])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Validation--test gap across all backbone, fusion, "
                 r"and training-pooling configurations. Gap is computed as "
                 r"(test -- validation) accuracy averaged over the 42 "
                 r"(pooling, profile) pairs per configuration. Positive "
                 r"values indicate no overfitting on hyperparameter "
                 r"selection.}")
    lines.append(r"\label{tab:meta_val_test_gap}")
    lines.append(r"\begin{tabular}{lllccc}")
    lines.append(r"\toprule")
    lines.append(r"Backbone & Fusion & Train pool. & Mean gap & "
                 r"Std gap & \% positive \\")
    lines.append(r"\midrule")
    prev_model = None
    prev_fusion = None
    for _, r in df.iterrows():
        model = r["model"] if r["model"] != prev_model else ""
        fusion = r["fusion_label"] if r["fusion_label"] != prev_fusion or model else ""
        lines.append(
            f"{model} & {fusion} & {r['pooling']} & "
            f"{r['mean_gap']:+.2f} & {r['std_gap']:.2f} & "
            f"{r['pct_positive']:.1f} \\\\"
        )
        prev_model = r["model"]
        prev_fusion = r["fusion_label"]
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{3}}{{l}}{{\\textbf{{Overall}}}} & "
        f"\\textbf{{{df['mean_gap'].mean():+.2f}}} & "
        f"{df['mean_gap'].std():.2f} & "
        f"{(df['mean_gap'] > 0).mean()*100:.1f} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="all_val_test_results")
    ap.add_argument("--out_dir", default="meta_val_test")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect(Path(args.results_root))
    if df.empty:
        print(f"No summary_gap.csv files found under {args.results_root}")
        return

    df.to_csv(out_dir / "meta_summary.csv", index=False)
    print(f"Collected {len(df)} configurations.")

    write_overall_stats(df, out_dir / "meta_overall_stats.txt")
    plot_gap_overview(df, out_dir / "meta_overall.png")
    to_latex_compact(df, out_dir / "meta_table.tex")


if __name__ == "__main__":
    main()
