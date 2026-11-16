"""First come first served against optimised booking, for each specialty.

A  = first come first served: book in turn, assume average case times, fill lists to 100%.
B1 = Monte Carlo booking in turn with a 10% overrun limit (as cautious as it is set).
B2 = Monte Carlo booking in turn, overrun limit retuned so its lists overrun as often as A's.
B3 = as B2, but filling gaps with short cases (ratio order), also matched to A's overrun rate.

Each comparison is paired run by run on the same surgeons and patients, at 0.5 extra lists a
week. Also compares the extra lists a year each needs to treat 95% of patients within 6 weeks
of being added. Writes results/differentials/.

Usage:  python scripts/run_differentials.py [--seeds 10] [--processes 2]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from theatre_sched.scenarios import (
    Scenario, compare, compare_lists_needed, describe, lists_needed, match_overrun, run,
)

SPECIALTIES = ["general_surgery", "orthopaedics", "gynaecology", "ent"]
NAMES = {"general_surgery": "General surgery", "orthopaedics": "Orthopaedics",
         "gynaecology": "Gynaecology", "ent": "ENT"}
KEYS = ["treated_per_year", "cohort_within_6wk", "waiting_within_6wk", "overrun_rate",
        "cancelled_per_100_booked", "overtime_minutes_per_year", "median_wait_days"]


def one_specialty(args: tuple[str, int]) -> list[dict]:
    specialty, seeds = args
    a = Scenario(specialty=specialty, booking_rule="average", fill_fraction=1.0, policy="in_turn", extra_lists=0.5)
    ra = run(a, seeds)
    target = float(ra.per_seed["overrun_rate"].mean())
    b1 = replace(a, booking_rule="monte_carlo", alpha=0.10)
    b2, _ = match_overrun(target, b1, seeds)
    b3, _ = match_overrun(target, replace(b1, policy="ratio_fill"), seeds)
    la = lists_needed(a, seeds)
    rows = []
    for name, b in (("B1: Monte Carlo, 10% limit", b1), ("B2: Monte Carlo, matched overruns", b2),
                    ("B3: Monte Carlo + short-case fill, matched overruns", b3)):
        cmp = compare(ra, run(b, seeds)).set_index("key")
        lc = compare_lists_needed(la, lists_needed(b, seeds)).set_index("key").loc["cohort_within_6wk"]
        row = {"specialty": NAMES[specialty], "comparison": name, "B": describe(b)}
        for k in KEYS:
            row[f"{k}_A"], row[f"{k}_diff"], row[f"{k}_pm"] = cmp.at[k, "A"], cmp.at[k, "difference"], cmp.at[k, "plus_minus"]
        row.update(lists_year_A=lc["A"], lists_year_B=lc["B"], lists_year_diff=lc["difference"], lists_year_pm=lc["plus_minus"])
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "differentials")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with Pool(args.processes) as pool:
        rows = [r for chunk in pool.map(one_specialty, [(s, args.seeds) for s in SPECIALTIES]) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False)

    def pm(d, p, dec=1):
        if pd.isna(d):
            return "B doesn't reach 95% within 3 extra lists a week"
        return f"{round(d, dec) + 0.0:+.{dec}f} ± {p:.{dec}f}"

    with open(args.out / "summary.md", "w") as fh:
        for comp, g in df.groupby("comparison", sort=False):
            fh.write(f"### A (first come first served, 100% fill) vs {comp}\n\n")
            t = pd.DataFrame({
                "Specialty": g["specialty"],
                "B": g["B"],
                "Patients treated a year": [pm(d, p, 0) for d, p in zip(g.treated_per_year_diff, g.treated_per_year_pm)],
                "Treated within 6 weeks of being added (pts)": [pm(d, p) for d, p in zip(g.cohort_within_6wk_diff, g.cohort_within_6wk_pm)],
                "Lists overrunning (pts)": [pm(d, p) for d, p in zip(g.overrun_rate_diff, g.overrun_rate_pm)],
                "Cancelled per 100 booked": [pm(d, p, 2) for d, p in zip(g.cancelled_per_100_booked_diff, g.cancelled_per_100_booked_pm)],
                "Extra lists a year to reach 95%": [pm(d, p, 0) for d, p in zip(g.lists_year_diff, g.lists_year_pm)],
            })
            fh.write(t.to_markdown(index=False))
            fh.write("\n\n")
    print(open(args.out / "summary.md").read())


if __name__ == "__main__":
    main()
