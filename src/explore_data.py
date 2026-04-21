"""
Data exploration for the Gaming & Mental Health dataset.
Computes the numbers needed to justify (or skip) visualizations in the report:
  1. Class balance        -> does macro-F1 story hold?
  2. Feature scales       -> is z-scoring justified?
  3. Feature correlations -> is a heatmap worth the space?
  4. Regression label distribution -> any skew worth mentioning?

Usage: python explore_data.py --data path/to/features.npz
"""

import argparse
import numpy as np


def load(path):
    d = np.load(path)
    return {
        "Xtr": d["xtrain"], "Xte": d["xtest"],
        "yr_tr": d["ytrainreg"], "yr_te": d["ytestreg"],
        "yc_tr": d["ytrainclassif"].astype(int),
        "yc_te": d["ytestclassif"].astype(int),
    }


def class_balance(y, name):
    print(f"\n=== Class balance: {name} ===")
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    for c, n in zip(classes, counts):
        label = {0: "Low", 1: "Medium", 2: "High"}.get(int(c), str(c))
        print(f"  class {int(c)} ({label:6s}): {n:4d}  ({100*n/total:5.2f}%)")
    ratio = counts.max() / counts.min()
    print(f"  max/min ratio: {ratio:.2f}x")
    verdict = "IMBALANCED (macro-F1 justified)" if ratio >= 1.5 else \
              "mildly imbalanced" if ratio >= 1.2 else "balanced"
    print(f"  verdict: {verdict}")


def feature_scales(X, name):
    print(f"\n=== Feature scales: {name} ===")
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    mins  = X.min(axis=0)
    maxs  = X.max(axis=0)
    print(f"  {'feat':>4}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
    for i in range(X.shape[1]):
        print(f"  {i:>4}  {means[i]:>10.3f}  {stds[i]:>10.3f}  {mins[i]:>10.3f}  {maxs[i]:>10.3f}")
    std_ratio = stds.max() / stds.min()
    range_ratio = (maxs - mins).max() / (maxs - mins).min()
    print(f"\n  std range:   [{stds.min():.3f}, {stds.max():.3f}]  ratio {std_ratio:.1f}x")
    print(f"  range ratio: {range_ratio:.1f}x (max feature range / min feature range)")
    if std_ratio >= 10:
        print("  verdict: scales differ strongly -> z-scoring clearly justified")
    elif std_ratio >= 3:
        print("  verdict: scales differ moderately -> z-scoring still justified")
    else:
        print("  verdict: scales are comparable -> z-scoring is a nicety, not a necessity")


def correlations(X, name, top_k=5, threshold=0.7):
    print(f"\n=== Feature correlations: {name} ===")
    C = np.corrcoef(X, rowvar=False)
    n = C.shape[0]
    # upper triangle, off-diagonal
    iu = np.triu_indices(n, k=1)
    pairs = sorted(zip(iu[0], iu[1], C[iu]), key=lambda t: -abs(t[2]))
    print(f"  top {top_k} |corr| pairs:")
    for i, j, c in pairs[:top_k]:
        print(f"    feat {i:>2} <-> feat {j:>2}:  {c:+.3f}")
    max_abs = max(abs(c) for _, _, c in pairs)
    strong = [p for p in pairs if abs(p[2]) >= threshold]
    print(f"  max |corr|: {max_abs:.3f}")
    print(f"  pairs with |corr| >= {threshold}: {len(strong)}")
    if max_abs >= threshold:
        print("  verdict: nontrivial correlation -> a heatmap could earn its place")
    else:
        print("  verdict: nothing strong -> skip the heatmap")


def regression_labels(y, name):
    print(f"\n=== Regression labels: {name} ===")
    print(f"  min={y.min():.2f}  max={y.max():.2f}  mean={y.mean():.2f}  std={y.std():.2f}")
    print(f"  median={np.median(y):.2f}  skew={((y-y.mean())**3).mean()/y.std()**3:.3f}")
    # quick histogram by integer bin
    edges = np.arange(int(y.min()), int(y.max())+2) - 0.5
    counts, _ = np.histogram(y, bins=edges)
    print("  histogram (integer bins):")
    for v, c in zip(range(int(y.min()), int(y.max())+1), counts):
        bar = "#" * int(50 * c / counts.max())
        print(f"    {v:>2}: {c:>4}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/user-data/uploads/features.npz")
    args = ap.parse_args()

    d = load(args.data)
    print(f"Loaded: Xtrain {d['Xtr'].shape}, Xtest {d['Xte'].shape}")

    class_balance(d["yc_tr"], "train (classification)")
    class_balance(d["yc_te"], "test  (classification)")
    feature_scales(d["Xtr"], "train")
    correlations(d["Xtr"], "train")
    regression_labels(d["yr_tr"], "train")


if __name__ == "__main__":
    main()
