"""Are the duration models' intervals honest? Checked against the true pair averages.

For each amount of history (a typical 3, 15, 50 and 200 past cases per surgeon-procedure pair) and
each of ``--worlds`` synthetic worlds, fit the hierarchical (empirical Bayes) model and the grid
Bayesian model, and record how often each one's 90% interval for a pair's typical log time
contains the true value, how wide the intervals are, and the estimate of tau (true value 0.08).

Writes results/coverage.csv and results/coverage.md.

Usage:  python scripts/run_coverage.py [--worlds 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theatre_sched import WorldConfig, fit_hierarchical, make_world  # noqa: E402
from theatre_sched.durations import fit_bayes  # noqa: E402

HISTORY = [3, 15, 50, 200]
Z90 = 1.6449


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=30)
    args = ap.parse_args()
    rows = []
    for hist in HISTORY:
        for seed in range(args.worlds):
            w = make_world(WorldConfig(history_median_cases=hist), seed=seed)
            eb = fit_hierarchical(w.history).table.merge(w.true_params, on=["surgeon", "procedure"])
            lo, hi = eb.mu_hat - Z90 * np.sqrt(eb.post_var), eb.mu_hat + Z90 * np.sqrt(eb.post_var)
            b = fit_bayes(w.history, seed=seed)
            iv = b.interval(0.90).merge(w.true_params, on=["surgeon", "procedure"])
            rows += [
                {"history_cases": hist, "world": seed, "model": "Hierarchical (empirical Bayes)",
                 "covered": float(((lo <= eb.mu) & (eb.mu <= hi)).mean()), "width": float((hi - lo).mean()),
                 "tau": float(fit_hierarchical(w.history).tau)},
                {"history_cases": hist, "world": seed, "model": "Bayesian (grid)",
                 "covered": float(((iv.low <= iv.mu) & (iv.mu <= iv.high)).mean()),
                 "width": float((iv.high - iv.low).mean()), "tau": float(np.median(b.tau))},
            ]
        print(f"history {hist} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    df.to_csv(out / "coverage.csv", index=False)
    t = (df.groupby(["history_cases", "model"], sort=False)
         .agg(coverage=("covered", "mean"), width=("width", "mean"), tau=("tau", "median")).reset_index())
    t["coverage"] = (100 * t["coverage"]).round(1)
    t = t.round({"width": 3, "tau": 3})
    t.columns = ["Typical past cases", "Model", "90% intervals containing the truth (%)",
                 "Average width (log scale)", "Median estimate of tau (true 0.08)"]
    (out / "coverage.md").write_text(f"{args.worlds} synthetic worlds per row.\n\n" + t.to_markdown(index=False) + "\n")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
