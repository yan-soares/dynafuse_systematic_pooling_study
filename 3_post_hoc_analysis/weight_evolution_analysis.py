"""
Weight evolution analysis (supports rewrite of Section 4.8, Major #5).

The reviewer pointed out that the interpretability claim is only
correlational: showing that DynaFuse weights concentrate on layers
identified as semantic by probing studies does not prove the weights
reflect semantic content. This script provides additional empirical
material to refine the discussion:

    1. Convergence plot: how weights evolve over training epochs, per task.
       If weights converge smoothly and consistently, they likely reflect
       a property of the encoder, not classifier noise.

    2. Cross-seed stability: variance of the final converged weights across
       seeds, as a function of layer. Complements the existing Figure 9.

    3. Time-to-convergence: epoch at which the weights stop changing
       meaningfully. Lower values suggest the optimization landscape is
       well-behaved (few local minima).

Input: all_histories.json with keys like "MR_seed_42", each containing
weight_evolution (list of L weights per epoch).

Usage:
    python weight_evolution_analysis.py \\
        --histories all_histories.json \\
        --out_dir weight_evolution
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


SENTEVAL_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_histories(path: Path) -> dict:
    """Return {task: {seed: weight_evolution_array}} where each array has
    shape (n_epochs, n_layers).
    """
    with open(path) as f:
        h = json.load(f)
    out = defaultdict(dict)
    pat = re.compile(r"^(.+)_seed_(\w+)$")
    for k, v in h.items():
        m = pat.match(k)
        if not m:
            continue
        task, seed = m.group(1), m.group(2)
        we = np.array(v["weight_evolution"])
        # The values are already softmax-normalized (rows sum to 1).
        # We keep both labels for backward compatibility with the rest of
        # the code.
        out[task][seed] = {"raw": we, "softmax": we}
    return dict(out)


# ----------------------------------------------------------------------------
# Plot 1: convergence trajectories (one panel per task)
# ----------------------------------------------------------------------------

def plot_convergence_grid(histories: dict, out_path: Path,
                          tasks=None, max_seeds=5):
    tasks = tasks or [t for t in SENTEVAL_TASKS if t in histories]
    if "NLI" in histories:
        tasks = list(tasks) + ["NLI"]

    n = len(tasks)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             sharex=False, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    cmap = plt.get_cmap("viridis")

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        seeds = list(histories[task].keys())[:max_seeds]
        # We average across seeds for the displayed curves, but show std band
        n_layers = histories[task][seeds[0]]["softmax"].shape[1]
        # Build a per-epoch per-layer mean / std across seeds.
        # Different seeds may converge at different epochs (due to early
        # stopping). Pad to the max epoch with the last value held constant.
        max_epochs = max(histories[task][s]["softmax"].shape[0] for s in seeds)
        stacks = []
        for s in seeds:
            arr = histories[task][s]["softmax"]  # (E_s, L)
            if arr.shape[0] < max_epochs:
                last = arr[-1:, :]
                pad = np.repeat(last, max_epochs - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            stacks.append(arr)
        st = np.stack(stacks, axis=0)  # (S, E, L)
        mean = st.mean(axis=0)
        std = st.std(axis=0)

        epochs = np.arange(1, max_epochs + 1)
        for ell in range(n_layers):
            color = cmap(ell / (n_layers - 1))
            ax.plot(epochs, mean[:, ell], color=color, linewidth=1.5,
                    label=f"L{ell+1}" if idx == 0 else None)
            ax.fill_between(
                epochs, mean[:, ell] - std[:, ell], mean[:, ell] + std[:, ell],
                color=color, alpha=0.12,
            )
        ax.set_title(task, fontsize=12)
        ax.set_xlabel("Epoch")
        if idx % ncols == 0:
            ax.set_ylabel("Softmax weight")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(tasks), len(axes)):
        axes[j].axis("off")

    # Single legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9,
               bbox_to_anchor=(1.02, 0.5), ncol=1, title="Layer")
    fig.suptitle(
        "Layer weight evolution during training (mean ± std over seeds)",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------------
# Plot 2: time-to-stabilization
# ----------------------------------------------------------------------------

def compute_stabilization(histories: dict, tol: float = 1e-3) -> pd.DataFrame:
    """For each (task, seed), the epoch at which the L1 distance between
    consecutive weight vectors first drops below `tol` and stays there.
    """
    rows = []
    for task, by_seed in histories.items():
        for seed, vals in by_seed.items():
            arr = vals["softmax"]  # (E, L)
            E = arr.shape[0]
            stab_epoch = E  # default: never stabilized within run
            diffs = np.abs(arr[1:] - arr[:-1]).sum(axis=1)
            for e in range(len(diffs)):
                if diffs[e] < tol:
                    # check it stays
                    if (diffs[e:] < tol).all():
                        stab_epoch = e + 1
                        break
            rows.append({
                "task": task, "seed": seed,
                "stab_epoch": stab_epoch,
                "total_epochs": E,
                "final_diff": float(diffs[-1]) if len(diffs) else 0.0,
            })
    return pd.DataFrame(rows)


def plot_stabilization(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    tasks = sorted(df["task"].unique())
    means = [df[df["task"] == t]["stab_epoch"].mean() for t in tasks]
    stds = [df[df["task"] == t]["stab_epoch"].std() for t in tasks]
    ax.bar(tasks, means, yerr=stds, capsize=4, color="#5b9bd5", alpha=0.85)
    ax.set_ylabel("Epoch of weight stabilization")
    ax.set_xlabel("Task")
    ax.set_title("Time to weight stabilization (tol = 1e-3, mean ± std over seeds)",
                 fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------------
# Plot 3: final weight cross-seed stability (per layer)
# ----------------------------------------------------------------------------

def plot_final_stability(histories: dict, out_path: Path):
    """For each task, std of final weight vector across seeds, per layer."""
    rows = []
    for task, by_seed in histories.items():
        finals = np.stack(
            [vals["softmax"][-1] for vals in by_seed.values()], axis=0
        )  # (S, L)
        std_per_layer = finals.std(axis=0)
        for ell, val in enumerate(std_per_layer):
            rows.append({"task": task, "layer": ell + 1, "std": val})
    df = pd.DataFrame(rows)

    pivot = df.pivot(index="task", columns="layer", values="std")
    order = [t for t in SENTEVAL_TASKS if t in pivot.index]
    if "NLI" in pivot.index:
        order.append("NLI")
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(13, 5))
    import seaborn as sns
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="Reds",
        cbar_kws={"label": "Std of final weight across seeds"},
        ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Task")
    ax.set_title("Cross-seed stability of converged weights "
                 "(complement to Figure 9)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--histories", default="all_histories.json")
    ap.add_argument("--out_dir", default="weight_evolution")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="Stabilization tolerance (L1 distance between epochs).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(Path(args.histories))
    print(f"Loaded {sum(len(v) for v in histories.values())} runs "
          f"across {len(histories)} tasks.")

    plot_convergence_grid(
        histories, out_dir / "convergence_grid.png",
    )

    stab_df = compute_stabilization(histories, tol=args.tol)
    stab_df.to_csv(out_dir / "stabilization.csv", index=False)
    plot_stabilization(stab_df, out_dir / "stabilization.png")

    plot_final_stability(histories, out_dir / "final_stability.png")

    # Quick report
    print("\n=== Quick stats ===")
    print(f"Mean stabilization epoch (over all task/seed runs): "
          f"{stab_df['stab_epoch'].mean():.1f}")
    print(f"Tasks where mean stabilization < 10 epochs: "
          f"{(stab_df.groupby('task')['stab_epoch'].mean() < 10).sum()}")


if __name__ == "__main__":
    main()
