"""Compare booking methods across many synthetic worlds.

For each world:
  * fit the plug-in duration model to the historical cases;
  * book four weeks of lists with each method and setting;
  * score every schedule against fresh draws from the TRUE duration
    distributions, which no planner sees.

Methods:
  avg-queue  book to average case times, conventional queue order (priority, then wait)
  avg-ratio  book to average case times, most weight per expected minute first
  cc-plugin  chance-constrained, using the plug-in duration model (what a real planner could do)
  cc-hier    chance-constrained, using the hierarchical (partial pooling) duration model
  cc-oracle  chance-constrained, using the true durations (an upper reference: perfect knowledge)

Writes results/runs.csv, results/summary.csv, results/summary.md,
results/frontier.png and results/calibration.png.

Usage:  python scripts/run_experiment.py [--worlds 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theatre_sched import (
    LognormalPairModel,
    fit_hierarchical,
    WorldConfig,
    add_weights,
    book_to_average,
    chance_constrained,
    evaluate_schedule,
    fit_plugin,
    make_world,
    summarise,
)

FILLS = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
ALPHAS = [0.02, 0.05, 0.10, 0.20, 0.30]
LONG_CASES = {"lap_cholecystectomy", "incisional_hernia_repair"}
LABELS = {
    "avg-queue": "Book to average (queue order)",
    "avg-ratio": "Book to average (ratio order)",
    "cc-plugin": "Chance-constrained (plug-in model)",
    "cc-hier": "Chance-constrained (hierarchical model)",
    "cc-oracle": "Chance-constrained (true durations)",
}

# Colours: reference categorical palette, slots 1-4; the oracle is a neutral reference line.
COLOURS = {"avg-queue": "#2a78d6", "avg-ratio": "#eb6834", "cc-plugin": "#1baf7a", "cc-hier": "#eda100",
           "cc-oracle": "#52514e"}
MARKERS = {"avg-queue": "o", "avg-ratio": "s", "cc-plugin": "D", "cc-hier": "v", "cc-oracle": "^"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def run_world(seed: int, cfg_in: WorldConfig, lam: float, n_plan: int, n_eval: int) -> tuple[list[dict], list[dict]]:
    w = make_world(cfg_in, seed=seed)
    cfg = w.config
    p = add_weights(w.patients, lam=lam)
    fit = fit_plugin(w.history)
    truth = LognormalPairModel(w.true_params)

    plan = fit.model.sample(p, n_plan, np.random.default_rng([seed, 1]))
    hier = fit_hierarchical(w.history).sample(p, n_plan, np.random.default_rng([seed, 4]))
    oracle = truth.sample(p, n_plan, np.random.default_rng([seed, 2]))
    reality = truth.sample(p, n_eval, np.random.default_rng([seed, 3]))
    hist_mean = fit.historical_mean(p)

    jobs = []
    for f in FILLS:
        jobs.append(("avg-queue", f, lambda f=f: book_to_average(p, w.sessions, hist_mean, cfg.turnaround_minutes, f, "queue")))
        jobs.append(("avg-ratio", f, lambda f=f: book_to_average(p, w.sessions, hist_mean, cfg.turnaround_minutes, f, "ratio")))
    for a in ALPHAS:
        jobs.append(("cc-plugin", a, lambda a=a: chance_constrained(p, w.sessions, plan, cfg.turnaround_minutes, a)))
        jobs.append(("cc-hier", a, lambda a=a: chance_constrained(p, w.sessions, hier, cfg.turnaround_minutes, a)))
        jobs.append(("cc-oracle", a, lambda a=a: chance_constrained(p, w.sessions, oracle, cfg.turnaround_minutes, a)))

    rows, list_rows = [], []
    for method, setting, fn in jobs:
        sched = fn()
        per_session = evaluate_schedule(sched, w.sessions, reality, cfg.turnaround_minutes)
        rows.append({"world": seed, "method": method, "setting": setting,
                     **summarise(sched, per_session, p, LONG_CASES)})
        for r in per_session.itertuples(index=False):
            list_rows.append({"world": seed, "method": method, "setting": setting,
                              "session_id": r.session_id, "p_overrun": r.p_overrun, "n_cases": r.n_cases})
    return rows, list_rows


def extra_at_matched_risk(runs: pd.DataFrame, baseline: str, method: str) -> pd.DataFrame:
    """Per world: patients booked by ``method`` minus the baseline's, at the same realised overrun rate.

    The baseline's patients-vs-overrun curve (over its fill fractions) is linearly
    interpolated at each overrun rate the method achieved. Rates outside the
    baseline's range are left out rather than extrapolated.
    """
    out = []
    for world, g in runs.groupby("world"):
        b = g[g.method == baseline].sort_values("overrun_rate")
        m = g[g.method == method]
        x, y = b["overrun_rate"].to_numpy(), b["patients_booked"].to_numpy()
        for r in m.itertuples(index=False):
            if x.min() <= r.overrun_rate <= x.max():
                out.append({"world": world, "setting": r.setting,
                            "extra_patients": r.patients_booked - np.interp(r.overrun_rate, x, y)})
    return pd.DataFrame(out)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_frontier(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style_axes(ax)
    for method in ["avg-queue", "avg-ratio", "cc-plugin", "cc-hier", "cc-oracle"]:
        s = summary[summary.method == method].sort_values("overrun_rate")
        ls = "--" if method == "cc-oracle" else "-"
        ax.plot(100 * s["overrun_rate"], s["patients_booked"], ls, color=COLOURS[method], lw=2,
                marker=MARKERS[method], ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[method])
    ax.set_xlabel("Lists that overrun (%), scored on true durations", color=INK, fontsize=10)
    ax.set_ylabel("Patients booked in four weeks (5 surgeons)", color=INK, fontsize=10)
    ax.set_title("Patients booked against overrun risk", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    ax.set_xlim(left=0)
    fig.text(0.01, 0.01, f"Mean of {summary['n_worlds'].iloc[0]} synthetic worlds. Points along each line are different "
             "buffers (book to average) or overrun limits (chance-constrained).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_calibration(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=SURFACE)
    style_axes(ax)
    lim = 100 * max(ALPHAS) * 1.15
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1, zorder=1)
    ax.annotate("realised = target", (lim * 0.80, lim * 0.80), xytext=(-8, 10), textcoords="offset points",
                ha="right", fontsize=8, color=MUTED)
    for method in ["cc-plugin", "cc-hier", "cc-oracle"]:
        s = summary[summary.method == method].sort_values("setting")
        ls = "--" if method == "cc-oracle" else "-"
        ax.plot(100 * s["setting"], 100 * s["overrun_rate"], ls, color=COLOURS[method], lw=2, marker=MARKERS[method],
                ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[method] + ": average list")
        ax.plot(100 * s["setting"], 100 * s["p90_list_risk"], ":", color=COLOURS[method], lw=1.5,
                label=LABELS[method] + ": riskiest 10% of lists")
    ax.set_xlabel("Target: maximum P(overrun) per list (%)", color=INK, fontsize=10)
    ax.set_ylabel("Realised P(overrun) on true durations (%)", color=INK, fontsize=10)
    ax.set_title("Does the planner hit its own target?", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--lam", type=float, default=1.0, help="weight on urgency relative to throughput")
    ap.add_argument("--history-cases", type=float, default=15.0,
                    help="typical number of past cases per surgeon-procedure pair (median)")
    ap.add_argument("--plan-samples", type=int, default=2000)
    ap.add_argument("--eval-samples", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = WorldConfig(history_median_cases=args.history_cases)
    rows, list_rows = [], []
    for seed in range(args.worlds):
        r, lr = run_world(seed, cfg, args.lam, args.plan_samples, args.eval_samples)
        rows += r
        list_rows += lr
        print(f"world {seed + 1}/{args.worlds} done")
    runs = pd.DataFrame(rows)
    lists = pd.DataFrame(list_rows)
    runs.to_csv(args.out / "runs.csv", index=False)

    # Per-list risk: the 90th percentile of realised P(overrun) across all non-empty lists.
    p90 = (lists[lists.n_cases > 0].groupby(["method", "setting"])["p_overrun"]
           .quantile(0.9).rename("p90_list_risk").reset_index())
    metrics = ["patients_booked", "weight_booked", "must_book_unbooked", "overrun_rate",
               "overtime_per_list", "utilisation", "touch_utilisation", "long_case_share"]
    summary = runs.groupby(["method", "setting"])[metrics].mean().reset_index()
    summary = summary.merge(p90, on=["method", "setting"])
    summary["n_worlds"] = args.worlds
    summary.to_csv(args.out / "summary.csv", index=False)

    # Headline: extra patients at matched realised overrun rate, mean and standard error across worlds.
    head = []
    for base in ["avg-queue", "avg-ratio"]:
        for m in ["cc-plugin", "cc-hier", "cc-oracle"]:
            d = extra_at_matched_risk(runs, base, m)
            for setting, g in d.groupby("setting"):
                head.append({"method": m, "vs": base, "setting": setting, "worlds": len(g),
                             "extra_patients": g.extra_patients.mean(),
                             "se": g.extra_patients.std(ddof=1) / np.sqrt(len(g))})
    head = pd.DataFrame(head)
    head.to_csv(args.out / "matched_risk.csv", index=False)

    plot_frontier(summary, args.out / "frontier.png")
    plot_calibration(summary, args.out / "calibration.png")

    with open(args.out / "summary.md", "w") as fh:
        t = summary.copy()
        t["method"] = t["method"].map(LABELS)
        t["overrun_rate"] = (100 * t["overrun_rate"]).round(1)
        t["p90_list_risk"] = (100 * t["p90_list_risk"]).round(1)
        t["utilisation"] = (100 * t["utilisation"]).round(1)
        t["touch_utilisation"] = (100 * t["touch_utilisation"]).round(1)
        t["long_case_share"] = (100 * t["long_case_share"]).round(1)
        t = t.round({"patients_booked": 1, "overtime_per_list": 1, "must_book_unbooked": 2})
        cols = ["method", "setting", "patients_booked", "overrun_rate", "p90_list_risk",
                "overtime_per_list", "utilisation", "touch_utilisation", "long_case_share", "must_book_unbooked"]
        fh.write(t[cols].to_markdown(index=False))
        fh.write("\n\nExtra patients vs book-to-average at the same realised overrun rate\n\n")
        fh.write(head.round(2).to_markdown(index=False))
        fh.write("\n")
    print(summary.round(3).to_string())
    print(head.round(2).to_string())


if __name__ == "__main__":
    main()
