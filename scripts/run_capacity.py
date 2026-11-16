"""Capacity scenarios: how many extra lists does each specialty need to meet the 6-week target?

For each specialty and each synthetic world (seed), the same patients are run through every
combination of:

  extra lists per week   0, 0.25, ..., 2.0 (0.25 = one extra all-day list every four weeks)
  booking policy         in_turn, ratio_fill, touch_fill
  changeover time        15 or 5 minutes (5 ~ next patient anaesthetised in an anaesthetic room)
  booking caution        alpha = 0.10 or 0.30 (limit on each list's estimated P(overrun))

Each run is 26 weeks of warm-up then 52 measured weeks. Writes to results/capacity/.

Usage:  python scripts/run_capacity.py [--seeds 10] [--processes 2]
"""

from __future__ import annotations

import argparse
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theatre_sched import WorldConfig, make_world
from theatre_sched.rolling import POLICIES, RollingConfig, make_patients, simulate, summarise_run

SPECIALTIES = ["general_surgery", "orthopaedics", "gynaecology", "ent"]
NAMES = {"general_surgery": "General surgery", "orthopaedics": "Orthopaedics",
         "gynaecology": "Gynaecology", "ent": "ENT"}
EXTRA = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TURNAROUND = [15.0, 5.0]
ALPHA = [0.10, 0.30]
N_SURGEONS = 6
MEASURES = {"waiting_within_6wk": "Still waiting: within 6 weeks (national-style)",
            "treated_within_6wk": "Treated: had waited 6 weeks or less",
            "cohort_within_6wk": "Added to list: treated within 6 weeks"}

INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def run_specialty_seed(job: tuple[str, int]) -> list[dict]:
    specialty, rep = job
    seed = 1000 * SPECIALTIES.index(specialty) + rep
    world = make_world(WorldConfig(specialty=specialty, n_surgeons=N_SURGEONS), seed=seed)
    pts = make_patients(world, RollingConfig(), seed)
    rows = []
    for extra, policy, turnaround, alpha in product(EXTRA, POLICIES, TURNAROUND, ALPHA):
        cfg = RollingConfig(extra_lists=extra, policy=policy, turnaround=turnaround, alpha=alpha)
        out = simulate(world, cfg, seed, pts)
        rows.append({"specialty": specialty, "seed": rep, "extra_lists": extra, "policy": policy,
                     "turnaround": turnaround, "alpha": alpha, **summarise_run(out, cfg)})
    return rows


def lists_needed(summary: pd.DataFrame, measure: str, threshold: float = 0.95) -> pd.DataFrame:
    """Smallest number of extra lists per week at which the mean of ``measure`` reaches ``threshold``."""
    out = []
    keys = ["specialty", "policy", "turnaround", "alpha"]
    for k, g in summary.groupby(keys):
        g = g.sort_values("extra_lists")
        ok = g[g[measure] >= threshold]
        row = dict(zip(keys, k))
        if len(ok):
            first = ok.iloc[0]
            row.update(extra_lists_needed=first["extra_lists"], touch_utilisation_there=first["touch_utilisation"])
        else:
            row.update(extra_lists_needed=np.nan, touch_utilisation_there=np.nan)
        out.append(row)
    return pd.DataFrame(out)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_measures(summary: pd.DataFrame, path: Path) -> None:
    s = summary[(summary.policy == "in_turn") & (summary.turnaround == 15) & (summary.alpha == 0.10)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, sp in zip(axes.flat, SPECIALTIES):
        style(ax)
        g = s[s.specialty == sp].sort_values("extra_lists")
        for (m, label), c, mk in zip(MEASURES.items(), [C1, C2, C3], ["o", "s", "D"]):
            ax.plot(g.extra_lists, 100 * g[m], color=c, lw=2, marker=mk, ms=5, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=label)
        ax.axhline(95, color=MUTED, lw=1, ls="--")
        ax.set_title(NAMES[sp], loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_ylim(40, 101)
    axes[0, 0].annotate("95% target", (2.0, 95), xytext=(0, -11), textcoords="offset points", ha="right",
                        fontsize=8, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("Extra all-day lists per week", color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Within 6 weeks (%)", color=INK, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Three ways of measuring the same 6-week target", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_utilisation(summary: pd.DataFrame, path: Path) -> None:
    s = summary[summary.policy == "in_turn"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True, facecolor=SURFACE)
    combos = [(15, 0.10, C1, "o", "15-min changeover, overrun limit 10%"),
              (5, 0.30, C2, "s", "5-min changeover, overrun limit 30%")]
    for ax, sp in zip(axes.flat, SPECIALTIES):
        style(ax)
        for t, a, c, mk, label in combos:
            g = s[(s.specialty == sp) & (s.turnaround == t) & (s.alpha == a)].sort_values("extra_lists")
            ax.plot(g.extra_lists, 100 * g.touch_utilisation_full_lists, color=c, ls="--", lw=1.5, marker=mk,
                    ms=4, markeredgecolor=SURFACE, markeredgewidth=1, label=label + ": full lists")
            ax.plot(g.extra_lists, 100 * g.touch_utilisation, color=c, ls="-", lw=2, marker=mk, ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label + ": all lists")
        ax.axhline(85, color=MUTED, lw=1, ls=":")
        ax.set_title(NAMES[sp], loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_ylim(40, 90)
    axes[0, 0].annotate("85% target", (2.0, 85), xytext=(0, 4), textcoords="offset points", ha="right",
                        fontsize=8, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("Extra all-day lists per week", color=INK, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Touch-time utilisation (%)", color=INK, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Utilisation falls as lists are added", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "capacity")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(sp, r) for sp in SPECIALTIES for r in range(args.seeds)]
    with Pool(args.processes) as pool:
        rows = [row for chunk in pool.map(run_specialty_seed, jobs) for row in chunk]
    runs = pd.DataFrame(rows)
    runs.to_csv(args.out / "runs.csv", index=False)

    keys = ["specialty", "extra_lists", "policy", "turnaround", "alpha"]
    metrics = [c for c in runs.columns if c not in keys + ["seed"]]
    summary = runs.groupby(keys)[metrics].mean().reset_index()
    sem = runs.groupby(keys)[list(MEASURES)].sem().add_suffix("_se").reset_index()
    summary = summary.merge(sem, on=keys)
    summary.to_csv(args.out / "summary.csv", index=False)

    needed = pd.concat([lists_needed(summary, m).assign(measure=m) for m in MEASURES], ignore_index=True)
    needed.to_csv(args.out / "lists_needed.csv", index=False)

    plot_measures(summary, args.out / "measures.png")
    plot_utilisation(summary, args.out / "utilisation.png")

    with open(args.out / "summary.md", "w") as fh:
        base = needed[(needed.policy == "in_turn") & (needed.turnaround == 15) & (needed.alpha == 0.10)]
        t = base.pivot(index="specialty", columns="measure", values="extra_lists_needed")[list(MEASURES)]
        fh.write("Extra all-day lists per week needed to reach 95% (in turn, 15-min changeover, overrun limit 10%)\n\n")
        fh.write(t.rename(index=NAMES).to_markdown())
        fh.write("\n\nPolicies at 0.5 extra lists per week (15-min changeover, overrun limit 10%)\n\n")
        p = summary[(summary.extra_lists == 0.5) & (summary.turnaround == 15) & (summary.alpha == 0.10)]
        cols = ["specialty", "policy", *MEASURES, "median_wait_days", "p95_wait_days_treated",
                "touch_utilisation", "touch_utilisation_full_lists", "share_lists_short_of_patients",
                "cancelled_per_100_booked"]
        fh.write(p[cols].round(3).to_markdown(index=False))
        fh.write("\n")
    print(open(args.out / "summary.md").read())


if __name__ == "__main__":
    main()
