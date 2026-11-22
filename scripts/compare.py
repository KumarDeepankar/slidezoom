"""Paired comparison of two evaluated models on the same slides.

    python scripts/compare.py runs/flat512/test_best_t512_kall_v1 runs/zoom/test_best_tall_kall_v1

Each argument is a results prefix written by scripts/test.py (…_answers.csv and …_nll.csv).
Accuracy: difference (B - A), paired bootstrap 95% CI over slides, exact McNemar p-value.
NLL: mean per-slide difference (B - A, negative = B better) with a paired bootstrap 95% CI.
"""
import sys
from math import comb

import numpy as np
import pandas as pd


def bootstrap_ci(diff: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired booleans."""
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(pa: str, pb: str):
    A, B = pd.read_csv(pa + "_answers.csv"), pd.read_csv(pb + "_answers.csv")
    m = A.merge(B, on=["file_id", "kind"], suffixes=("_a", "_b"))
    print(f"A = {pa}\nB = {pb}\n{m.file_id.nunique()} paired slides\n")
    print(f"{'question':22s} {'A':>5s} {'B':>5s} {'B-A':>6s}   95% CI           McNemar p")
    for k, g in m.groupby("kind", sort=False):
        a, b = g.correct_a.values.astype(bool), g.correct_b.values.astype(bool)
        lo, hi = bootstrap_ci(b.astype(float) - a.astype(float))
        print(f"{k:22s} {a.mean():5.2f} {b.mean():5.2f} {b.mean() - a.mean():+6.2f}   [{lo:+.2f}, {hi:+.2f}]   {mcnemar(a, b):.3f}")
    na, nb = pd.read_csv(pa + "_nll.csv"), pd.read_csv(pb + "_nll.csv")
    n = na.merge(nb, on="file_id", suffixes=("_a", "_b"))
    d = (n.nll_b - n.nll_a).values
    lo, hi = bootstrap_ci(d)
    print(f"\nanswer NLL  A {n.nll_a.mean():.3f}  B {n.nll_b.mean():.3f}  B-A {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"tiles read  A {n.tiles_read_a.mean():.0f}  B {n.tiles_read_b.mean():.0f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
