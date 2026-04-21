"""
Generate the data-distribution figure for the report.
Two panels: (left) classification class counts, (right) regression label histogram.

Usage: python plot_data_distribution.py --data data/features.npz --out plots/data_dist.pdf
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features.npz")
    ap.add_argument("--out",  default="plots/data_dist.pdf")
    args = ap.parse_args()

    d = np.load(args.data)
    yc = d["ytrainclassif"].astype(int)
    yr = d["ytrainreg"]

    # --- style: clean, paper-friendly, no heavy gridlines ---
    plt.rcParams.update({
        "font.size":       8,
        "axes.labelsize":  8,
        "axes.titlesize":  9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    # figsize tuned for IEEE double-column: 3.3" fits one column,
    # 7.0" spans both columns via \begin{figure*}. Default: one column.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.8))

    # --- Left: class distribution ---
    classes = ["Low", "Medium", "High"]
    counts  = [int((yc == i).sum()) for i in range(3)]
    total   = sum(counts)
    # emphasize the minority class
    colors  = ["#4C78A8", "#4C78A8", "#E45756"]
    bars = ax1.bar(classes, counts, color=colors, edgecolor="black", linewidth=0.4)
    for b, c in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, c + total * 0.015,
                 f"{c}\n({100 * c / total:.1f}%)",
                 ha="center", va="bottom", fontsize=6.5)
    ax1.set_ylabel("training samples")
    ax1.set_title("(a) Classification labels")
    ax1.set_ylim(0, max(counts) * 1.25)
    ax1.tick_params(axis="y", length=2)

    # --- Right: regression label histogram ---
    edges = np.arange(-0.5, 11.5, 1.0)
    ax2.hist(yr, bins=edges, color="#4C78A8",
             edgecolor="black", linewidth=0.4)
    ax2.axvline(yr.mean(),      color="#E45756", linestyle="--",
                linewidth=1.0, label=f"mean={yr.mean():.2f}")
    ax2.axvline(np.median(yr),  color="black",   linestyle=":",
                linewidth=1.0, label=f"median={np.median(yr):.0f}")
    ax2.set_xlabel("addiction score")
    ax2.set_ylabel("training samples")
    ax2.set_title("(b) Regression labels")
    ax2.set_xticks(range(0, 11, 2))
    ax2.legend(frameon=False, loc="upper right",
               handlelength=1.2, borderpad=0.2)

    fig.tight_layout(pad=0.3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {args.out}")
    # also save a PNG for quick preview
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved preview  {png}")


if __name__ == "__main__":
    main()
