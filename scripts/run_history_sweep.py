"""How much does the duration model matter, as the amount of history changes?

Reruns the Part 1 comparison with a typical (median) 3, 15, 50 and 200 past cases per
surgeon-procedure pair, holding everything else in each world fixed, and reports for the
chance-constrained planners (plug-in, hierarchical, true durations) at a 10% overrun limit:

  * the realised overrun probability of the average list and of the riskiest 10% of lists;
  * extra patients booked compared with booking to average (ratio order) at the same overrun rate.

Writes results/history_sweep.csv, results/history_sweep.md and results/history_sweep.png.

Usage:  python scripts/run_history_sweep.py [--worlds 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import (AXIS, COLOURS, INK, LABELS, MARKERS, MUTED, SURFACE,  # noqa: E402
                            extra_at_matched_risk, run_world, style_axes)

from theatre_sched import WorldConfig  # noqa: E402

HISTORY = [3, 15, 50, 200]
ALPHA = 0.10
METHODS = ["cc-plugin", "cc-hier", "cc-oracle"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = ap.parse_args()

    rows = []
    for hist in HISTORY:
        cfg = WorldConfig(history_median_cases=hist)
        runs, lists = [], []
        for seed in range(args.worlds):
            r, lr = run_world(seed, cfg, 1.0, 2000, 20000)
            runs += r
            lists += lr
        runs, lists = pd.DataFrame(runs), pd.DataFrame(lists)
        for m in METHODS:
            sel = (runs.method == m) & np.isclose(runs.setting, ALPHA)
            li = lists[(lists.method == m) & np.isclose(lists.setting, ALPHA) & (lists.n_cases > 0)]
            extra = extra_at_matched_risk(runs, "avg-ratio", m)
            extra = extra[np.isclose(extra.setting, ALPHA)]
            rows.append({"history_cases": hist, "method": m,
                         "overrun_rate": runs.loc[sel, "overrun_rate"].mean(),
                         "p90_list_risk": li["p_overrun"].quantile(0.9),
                         "patients_booked": runs.loc[sel, "patients_booked"].mean(),
                         "extra_vs_avg_ratio": extra.extra_patients.mean(),
                         "extra_se": extra.extra_patients.std(ddof=1) / np.sqrt(len(extra)),
                         "worlds_matched": len(extra)})
        print(f"history {hist} done")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "history_sweep.csv", index=False)

    t = df.copy()
    t["method"] = t["method"].map(LABELS)
    t[["overrun_rate", "p90_list_risk"]] = (100 * t[["overrun_rate", "p90_list_risk"]]).round(1)
    t = t.round({"patients_booked": 1, "extra_vs_avg_ratio": 2, "extra_se": 2})
    (args.out / "history_sweep.md").write_text(
        f"Overrun limit {ALPHA:.0%}; {args.worlds} worlds per row. Overrun figures are percentages.\n\n"
        + t.to_markdown(index=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    x = np.arange(len(HISTORY))
    for m in METHODS:
        d = df[df.method == m]
        ls = "--" if m == "cc-oracle" else "-"
        kw = dict(color=COLOURS[m], lw=2, marker=MARKERS[m], ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        axes[0].plot(x, 100 * d["p90_list_risk"], ls, label=LABELS[m], **kw)
        axes[1].errorbar(x, d["extra_vs_avg_ratio"], yerr=2 * d["extra_se"], fmt=ls, capsize=3, label=LABELS[m], **kw)
    axes[0].axhline(100 * ALPHA, color=AXIS, lw=1)
    axes[0].annotate("target 10%", (len(HISTORY) - 1, 100 * ALPHA), xytext=(0, 6), textcoords="offset points",
                     ha="right", fontsize=8, color=MUTED)
    axes[0].set_ylabel("Realised P(overrun), riskiest 10% of lists (%)", color=INK, fontsize=9.5)
    axes[0].set_title("Does the planner hit its 10% limit?", loc="left", color=INK, fontsize=11, fontweight="bold")
    axes[1].axhline(0, color=AXIS, lw=1)
    axes[1].set_ylabel("Extra patients vs booking to average, same overrun rate", color=INK, fontsize=9.5)
    axes[1].set_title("What it's worth in patients booked", loc="left", color=INK, fontsize=11, fontweight="bold")
    for ax in axes:
        style_axes(ax)
        ax.set_xticks(x, [str(h) for h in HISTORY])
        ax.set_xlabel("Typical past cases per surgeon and procedure", color=INK, fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.text(0.01, 0.01, f"Mean of {args.worlds} synthetic worlds per point; error bars are 2 standard errors across "
             "worlds. Four weeks of lists for 5 surgeons.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out / "history_sweep.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
