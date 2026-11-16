"""Run scenarios and compare them: the engine behind the dashboard.

A ``Scenario`` bundles every setting a user can change. ``run`` simulates it over several
synthetic worlds (seeds) and returns per-seed headline numbers, weekly trajectories and
the waits of everyone treated.

``compare`` lines up two scenarios seed by seed. Seed r builds the same surgeons and
(where the demand settings match) the same patients in both, so a difference reflects
the change between the scenarios rather than chance. Uncertainty is reported as two
standard errors of the paired difference across seeds.

``lists_needed`` sweeps the number of extra lists a week. For each seed and each of the
three 6-week measures it finds where the measure first reaches the target share,
interpolating between the steps of the sweep, so that a difference in lists needed
between two scenarios also comes with a paired uncertainty.

``match_overrun`` retunes scenario B's booking caution (its Monte Carlo overrun limit or
its fill percentage) until B's lists overrun about as often as A's, so the two are
compared like for like.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .rolling import RollingConfig, make_patients, simulate, summarise_run
from .synthetic import SPECIALTIES, WorldConfig, make_world

SPECIALTY_ORDER = ["general_surgery", "orthopaedics", "gynaecology", "ent"]
MEASURES = ["waiting_within_6wk", "treated_within_6wk", "cohort_within_6wk"]
MEASURE_LABELS = {
    "waiting_within_6wk": "Still waiting: within target (national-style)",
    "treated_within_6wk": "Treated: had waited no longer than target",
    "cohort_within_6wk": "Added to list: treated within target",
}


@dataclass(frozen=True)
class Scenario:
    specialty: str = "general_surgery"
    n_surgeons: int = 6
    history_cases: float = 15.0  # typical past cases per surgeon-procedure pair the planner has seen
    lists_per_surgeon: int = 1
    session_minutes: float = 480.0
    working_weeks: int = 42
    load: float = 0.85  # demand as a share of available session time
    extra_lists: float = 0.0
    turnaround: float = 15.0
    policy: str = "in_turn"
    booking_rule: str = "average"
    fill_fraction: float = 1.0
    alpha: float = 0.10
    overrun_tolerance: float = 30.0
    target_weeks: int = 6
    urgent_share: float = 0.10

    def world_config(self) -> WorldConfig:
        return WorldConfig(specialty=self.specialty, n_surgeons=self.n_surgeons,
                           history_median_cases=self.history_cases, session_minutes=self.session_minutes)

    def rolling_config(self) -> RollingConfig:
        return RollingConfig(
            lists_per_surgeon=self.lists_per_surgeon, extra_lists=self.extra_lists,
            session_minutes=self.session_minutes, turnaround=self.turnaround, load=self.load,
            working_weeks=self.working_weeks, alpha=self.alpha, overrun_tolerance=self.overrun_tolerance,
            target_days=7 * self.target_weeks, urgent_target_days=min(28, 7 * self.target_weeks),
            urgent_share=self.urgent_share, policy=self.policy, booking_rule=self.booking_rule,
            fill_fraction=self.fill_fraction,
        )


@dataclass
class ScenarioResult:
    scenario: Scenario
    per_seed: pd.DataFrame  # one row per seed: headline numbers, including annual figures
    weekly: pd.DataFrame  # seed, week, waiting, waiting_within_6wk, ...
    waits: np.ndarray  # days waited by everyone treated in the measured weeks, all seeds


def seed_for(specialty: str, rep: int) -> int:
    """The same world seeds as scripts/run_capacity.py, so default settings reproduce its results."""
    idx = SPECIALTY_ORDER.index(specialty) if specialty in SPECIALTY_ORDER else len(SPECIALTY_ORDER)
    return 1000 * idx + rep


def _annual(row: dict) -> dict:
    tpw = row["treated_per_week"]
    return {
        "treated_per_year": 52 * tpw,
        "lists_per_year": 52 * row["lists_run_per_week"],
        "session_minutes_per_year": 52 * row["session_minutes_per_week"],
        "touch_minutes_per_year": 52 * row["touch_minutes_per_week"],
        "overtime_minutes_per_year": 52 * row["overtime_minutes_per_week"],
        "session_minutes_per_patient": row["session_minutes_per_week"] / tpw if tpw else np.nan,
    }


def _setup(scn: Scenario, rep: int):
    seed = seed_for(scn.specialty, rep)
    world = make_world(scn.world_config(), seed=seed)
    pts = make_patients(world, scn.rolling_config(), seed)
    return seed, world, pts


def run(scn: Scenario, seeds: int = 5) -> ScenarioResult:
    if scn.specialty not in SPECIALTIES:
        raise ValueError(f"unknown specialty: {scn.specialty}")
    rows, weekly, waits = [], [], []
    cfg = scn.rolling_config()
    for rep in range(seeds):
        seed, world, pts = _setup(scn, rep)
        out = simulate(world, cfg, seed, pts)
        r = summarise_run(out, cfg)
        r.update(_annual(r))
        rows.append({"seed": rep, **r})
        weekly.append(out["weekly"].assign(seed=rep))
        p = out["patients"]
        start, end = 7 * cfg.weeks_warmup, 7 * (cfg.weeks_warmup + cfg.weeks_measured)
        t = p[(p.treated_day >= start) & (p.treated_day < end)]
        waits.append((t.treated_day - t.added_day).to_numpy())
    return ScenarioResult(scn, pd.DataFrame(rows), pd.concat(weekly, ignore_index=True), np.concatenate(waits))


# label, key, unit (for display), decimals
COMPARE_METRICS = [
    ("Patients treated per year", "treated_per_year", "", 0),
    ("Still waiting: within target (national-style)", "waiting_within_6wk", "%", 1),
    ("Treated: had waited no longer than target", "treated_within_6wk", "%", 1),
    ("Added to list: treated within target", "cohort_within_6wk", "%", 1),
    ("Median wait (days)", "median_wait_days", "", 1),
    ("95th percentile wait (days)", "p95_wait_days_treated", "", 1),
    ("Patients on waiting list (average)", "waiting_list_mean", "", 0),
    ("Lists run per year", "lists_per_year", "", 0),
    ("Session minutes per year", "session_minutes_per_year", "", 0),
    ("Patient-in-theatre minutes per year", "touch_minutes_per_year", "", 0),
    ("Overtime minutes per year", "overtime_minutes_per_year", "", 0),
    ("Session minutes per patient treated", "session_minutes_per_patient", "", 1),
    ("Touch-time utilisation", "touch_utilisation", "%", 1),
    ("Lists that overrun", "overrun_rate", "%", 1),
    ("Cancelled on the day, per 100 booked", "cancelled_per_100_booked", "", 2),
]


def compare(a: ScenarioResult, b: ScenarioResult) -> pd.DataFrame:
    """Mean of A, mean of B, and the paired difference B - A with +/- two standard errors."""
    n = min(len(a.per_seed), len(b.per_seed))
    pa, pb = a.per_seed.iloc[:n], b.per_seed.iloc[:n]
    rows = []
    for label, key, unit, dec in COMPARE_METRICS:
        scale = 100.0 if unit == "%" else 1.0
        va, vb = scale * pa[key].to_numpy(), scale * pb[key].to_numpy()
        d = vb - va
        se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append({"measure": label, "key": key, "unit": unit, "decimals": dec,
                     "A": float(np.mean(va)), "B": float(np.mean(vb)),
                     "difference": float(np.mean(d)), "plus_minus": 2 * se})
    return pd.DataFrame(rows)


@dataclass
class ListsNeeded:
    curve: pd.DataFrame  # mean of each measure (and utilisation, overruns) per extra-lists level
    per_seed: pd.DataFrame  # one row per seed: extra lists a week needed for each measure (NaN = not reached)
    threshold: float
    max_extra: float

    @property
    def needed(self) -> dict:
        return {m: float(self.per_seed[m].mean()) if self.per_seed[m].notna().all() else np.nan for m in MEASURES}


def _crossing(levels: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """First point where ``values`` reaches ``threshold``, interpolating linearly between levels."""
    hit = np.flatnonzero(values >= threshold)
    if len(hit) == 0:
        return np.nan
    k = hit[0]
    if k == 0:
        return float(levels[0])
    x0, x1, y0, y1 = levels[k - 1], levels[k], values[k - 1], values[k]
    return float(x0 + (threshold - y0) / (y1 - y0) * (x1 - x0))


def lists_needed(scn: Scenario, seeds: int = 5, threshold: float = 0.95,
                 grid: tuple[float, ...] = tuple(np.round(np.arange(0, 3.01, 0.25), 2))) -> ListsNeeded:
    """Sweep extra lists a week and find, per seed and measure, where the target share is first reached."""
    setups = [_setup(scn, rep) for rep in range(seeds)]
    rows = []
    for extra in grid:
        cfg = replace(scn, extra_lists=float(extra)).rolling_config()
        for rep, (seed, world, pts) in enumerate(setups):
            r = summarise_run(simulate(world, cfg, seed, pts), cfg)
            rows.append({"extra_lists": float(extra), "seed": rep, **{m: r[m] for m in MEASURES},
                         "touch_utilisation": r["touch_utilisation"], "overrun_rate": r["overrun_rate"]})
    long = pd.DataFrame(rows)
    curve = long.groupby("extra_lists").mean(numeric_only=True).drop(columns="seed").reset_index()
    per_seed = []
    for rep, g in long.groupby("seed"):
        g = g.sort_values("extra_lists")
        per_seed.append({"seed": rep, **{m: _crossing(g["extra_lists"].to_numpy(), g[m].to_numpy(), threshold)
                                          for m in MEASURES}})
    return ListsNeeded(curve, pd.DataFrame(per_seed), threshold, float(max(grid)))


def compare_lists_needed(a: ListsNeeded, b: ListsNeeded) -> pd.DataFrame:
    """Extra lists a year needed by A and B for each measure, and the paired difference B - A (+/- 2 SE)."""
    rows = []
    n = min(len(a.per_seed), len(b.per_seed))
    for m in MEASURES:
        va, vb = 52 * a.per_seed[m].to_numpy()[:n], 52 * b.per_seed[m].to_numpy()[:n]
        ok = np.isfinite(va) & np.isfinite(vb)
        d = vb[ok] - va[ok]
        rows.append({"measure": MEASURE_LABELS[m], "key": m,
                     "A": float(np.mean(va)) if np.isfinite(va).all() else np.nan,
                     "B": float(np.mean(vb)) if np.isfinite(vb).all() else np.nan,
                     "difference": float(np.mean(d)) if ok.all() else np.nan,
                     "plus_minus": float(2 * np.std(d, ddof=1) / np.sqrt(len(d))) if ok.all() and len(d) > 1 else np.nan})
    return pd.DataFrame(rows)


def match_overrun(target_rate: float, scn: Scenario, seeds: int = 5) -> tuple[Scenario, float]:
    """Retune ``scn``'s booking caution so its lists overrun about as often as ``target_rate``.

    Monte Carlo rule: tries overrun limits from 2% to 50%. Average rule: tries fill percentages
    from 70% to 110%. Returns the closest scenario and its overrun rate.
    """
    if scn.booking_rule == "monte_carlo":
        options = [replace(scn, alpha=a) for a in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)]
    else:
        options = [replace(scn, fill_fraction=f) for f in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10)]
    setups = [_setup(scn, rep) for rep in range(seeds)]
    best, best_rate, best_gap = scn, np.nan, np.inf
    for opt in options:
        cfg = opt.rolling_config()
        rate = float(np.mean([summarise_run(simulate(w, cfg, s, p), cfg)["overrun_rate"] for s, w, p in setups]))
        if abs(rate - target_rate) < best_gap:
            best, best_rate, best_gap = opt, rate, abs(rate - target_rate)
    return best, best_rate


def describe(scn: Scenario) -> str:
    """Short plain-English description of how a scenario books lists."""
    order = {"in_turn": "in turn", "ratio_fill": "filling gaps with short cases",
             "touch_fill": "filling gaps with long cases"}[scn.policy]
    if scn.booking_rule == "average":
        how = f"to average case times, lists filled to {round(100 * scn.fill_fraction)}%"
    else:
        how = f"to a {round(100 * scn.alpha)}% Monte Carlo overrun limit"
    return f"booking {order}, {how}"


def headline(cmp: pd.DataFrame, lists_cmp: pd.DataFrame | None = None, target_weeks: int = 6,
             threshold: float = 0.95, session_minutes: float = 480.0, max_extra: float = 3.0) -> str:
    """One or two sentences summarising B against A, saying "no clear difference" when the
    paired difference is within two standard errors of zero."""
    c = cmp.set_index("key")

    def phrase(d: float, pm: float, dec: int, more: str, fewer: str, noun: str) -> str:
        if not np.isfinite(d):
            return ""
        if not np.isfinite(pm) or abs(d) <= pm:
            return f"no clear difference in {noun} ({d:+,.{dec}f} ± {pm:,.{dec}f})"
        return f"{abs(d):,.{dec}f} {more if d > 0 else fewer} {noun} (± {pm:,.{dec}f})"

    def row(key):
        return c.at[key, "difference"], c.at[key, "plus_minus"], int(c.at[key, "decimals"])

    parts = [
        phrase(*row("treated_per_year"), "more", "fewer", "patients treated a year"),
        phrase(*row("cohort_within_6wk"), "points higher", "points lower",
               f"share of patients treated within {target_weeks} weeks of being added"),
        phrase(*row("overrun_rate"), "points more", "points fewer", "lists overrunning"),
    ]
    text = "Compared with A, B gives " + "; ".join(p for p in parts if p) + "."
    if lists_cmp is not None:
        l = lists_cmp.set_index("key").loc["cohort_within_6wk"]
        if np.isfinite(l["difference"]):
            pm = l["plus_minus"]
            if not np.isfinite(pm) or abs(l["difference"]) <= pm:
                text += (f" To treat {threshold:.0%} within {target_weeks} weeks, there's no clear difference in the "
                         f"extra lists needed ({l['difference']:+.0f} ± {pm:.0f} a year).")
            else:
                word = "fewer" if l["difference"] < 0 else "more"
                text += (f" To treat {threshold:.0%} within {target_weeks} weeks, B needs about "
                         f"{abs(l['difference']):.0f} {word} extra lists a year (± {pm:.0f}), "
                         f"or {abs(l['difference']) * session_minutes:,.0f} {word} session minutes.")
        else:
            text += f" At least one scenario doesn't reach the target within {max_extra:g} extra lists a week."
    return text


def scenario_dict(scn: Scenario) -> dict:
    return asdict(scn)
