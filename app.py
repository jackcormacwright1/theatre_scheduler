"""Theatre list booking simulator: a Streamlit dashboard over the rolling simulation.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # works without installing the package

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from theatre_sched.rolling import POLICIES, RollingConfig, implied_load
from theatre_sched.scenarios import (
    MEASURE_LABELS,
    MEASURES,
    Scenario,
    compare,
    compare_lists_needed,
    describe,
    headline,
    lists_needed,
    match_overrun,
    run,
    seed_for,
)
from theatre_sched.synthetic import WorldConfig, make_world

st.set_page_config(page_title="Theatre list booking simulator", page_icon="🏥", layout="wide")

METHODOLOGY_FILE = Path(__file__).resolve().parent / "docs" / "METHODOLOGY.md"
METHODOLOGY = (METHODOLOGY_FILE.read_text(encoding="utf-8") if METHODOLOGY_FILE.exists()
               else "The methodology is in docs/METHODOLOGY.md in the project repository.")

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SPECIALTY_NAMES = {"general_surgery": "General surgery", "orthopaedics": "Orthopaedics",
                   "gynaecology": "Gynaecology", "ent": "ENT"}
POLICY_NAMES = {"in_turn": "In turn (urgent first, then longest wait)",
                "ratio_fill": "Due patients first, then fill gaps with short cases",
                "touch_fill": "Due patients first, then fill gaps with long cases"}
RULE_NAMES = {"average": "To average case times, up to a fill %",
              "monte_carlo": "To a Monte Carlo overrun limit"}
DEFAULTS = {"specialty": "general_surgery", "n_surgeons": 6, "lists_per_surgeon": 1, "session_minutes": 480,
            "working_weeks": 42, "referrals": 24, "extra_lists": 0.5, "turnaround": 15, "policy": "in_turn",
            "booking_rule": "average", "fill_pct": 100, "alpha_pct": 10, "overrun_tolerance": 30, "history_cases": 15}
B_START = {**DEFAULTS, "booking_rule": "monte_carlo"}
FIELDS = list(DEFAULTS)

for prefix, start in (("A", DEFAULTS), ("B", B_START)):
    for f, v in start.items():
        st.session_state.setdefault(f"{prefix}_{f}", v)
for k, v in {"seeds": 5, "target_weeks": 6, "threshold_pct": 95, "match_msg": ""}.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------- cached computation
@st.cache_data(show_spinner=False, max_entries=64)
def cached_run(scn: dict, seeds: int):
    return run(Scenario(**scn), seeds)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_needed(scn: dict, seeds: int, threshold: float):
    return lists_needed(Scenario(**scn), seeds, threshold)


@st.cache_data(show_spinner=False, max_entries=64)
def demand_share(specialty: str, n_surgeons: int, lists_per_surgeon: int, session_minutes: float,
                 working_weeks: int, referrals: float) -> float:
    """Theatre time the referrals need, as a share of the time available after leave (15-minute changeovers)."""
    world = make_world(WorldConfig(specialty=specialty, n_surgeons=n_surgeons), seed=seed_for(specialty, 0))
    cfg = RollingConfig(lists_per_surgeon=lists_per_surgeon, session_minutes=session_minutes,
                        working_weeks=working_weeks, referrals_per_week=referrals)
    return implied_load(world, cfg)


# ---------------------------------------------------------------- settings
def scenario_from_state(p: str) -> Scenario:
    g = lambda f: st.session_state[f"{p}_{f}"]  # noqa: E731
    return Scenario(
        specialty=g("specialty"), n_surgeons=int(g("n_surgeons")), history_cases=float(g("history_cases")),
        lists_per_surgeon=int(g("lists_per_surgeon")), session_minutes=float(g("session_minutes")),
        working_weeks=int(g("working_weeks")), referrals_per_week=float(g("referrals")),
        extra_lists=float(g("extra_lists")),
        turnaround=float(g("turnaround")), policy=g("policy"), booking_rule=g("booking_rule"),
        fill_fraction=g("fill_pct") / 100, alpha=g("alpha_pct") / 100,
        overrun_tolerance=float(g("overrun_tolerance")), target_weeks=int(st.session_state["target_weeks"]),
    )


def controls(p: str, two_columns: bool = False) -> None:
    k = lambda f: f"{p}_{f}"  # noqa: E731
    left, right = st.columns(2) if two_columns else (st.container(), st.container())
    with left, st.expander("Specialty and theatre time", expanded=True):
        st.selectbox("Specialty", list(SPECIALTY_NAMES), format_func=SPECIALTY_NAMES.get, key=k("specialty"))
        st.slider("Surgeons", 2, 12, key=k("n_surgeons"), help="Each surgeon has their own waiting list.")
        st.slider("Lists per surgeon per week", 1, 3, key=k("lists_per_surgeon"))
        st.select_slider("List length (minutes)", options=[240, 360, 480, 600], key=k("session_minutes"))
        st.slider("Operating weeks a year", 32, 52, step=2, key=k("working_weeks"),
                  help="The rest is leave, taken in two-week blocks.")
        st.slider("Extra lists per week", 0.0, 3.0, step=0.25, key=k("extra_lists"),
                  help="Saturday lists, given to the surgeons with most patients about to pass target. "
                       "0.25 = one every four weeks.")
    with left, st.expander("Demand", expanded=True):
        st.slider("Referrals added to the waiting list per week", 4, 100, key=k("referrals"),
                  help="Decisions to operate for the whole specialty, shared equally between the surgeons. "
                       "Adding surgeons or lists adds capacity, not demand.")
        share = demand_share(st.session_state[k("specialty")], st.session_state[k("n_surgeons")],
                             st.session_state[k("lists_per_surgeon")], st.session_state[k("session_minutes")],
                             st.session_state[k("working_weeks")], st.session_state[k("referrals")])
        st.caption(f"These need about {100 * share:.0f}% of the theatre time available after leave "
                   "(before any extra lists, at a 15-minute changeover).")
    with right, st.expander("Booking", expanded=True):
        st.radio("How full lists are booked", list(RULE_NAMES), format_func=RULE_NAMES.get, key=k("booking_rule"))
        avg = st.session_state[k("booking_rule")] == "average"
        st.slider("Fill lists to (% of session)", 60, 120, step=5, key=k("fill_pct"), disabled=not avg)
        st.slider("Overrun limit per list (%)", 1, 50, key=k("alpha_pct"), disabled=avg,
                  help="A list is allowed only if the Monte Carlo estimate of its chance of overrunning "
                       "is at most this.")
        st.selectbox("Booking order", list(POLICIES), format_func=POLICY_NAMES.get, key=k("policy"))
        st.select_slider("Planner's past cases per surgeon and procedure", options=[3, 15, 50, 200],
                         key=k("history_cases"),
                         help="How much history the planner's duration estimates rest on (typical value).")
    with right, st.expander("On the day"):
        st.slider("Changeover between cases (minutes)", 0, 30, step=5, key=k("turnaround"))
        st.slider("Cancel a case if expected to finish more than this many minutes late", 0, 90, step=10,
                  key=k("overrun_tolerance"))


def copy_a_to_b() -> None:
    for f in FIELDS:
        st.session_state[f"B_{f}"] = st.session_state[f"A_{f}"]
    st.session_state["match_msg"] = ""


def match_b_to_a() -> None:
    seeds = st.session_state["seeds"]
    a = cached_run(asdict(scenario_from_state("A")), seeds)
    target = float(a.per_seed["overrun_rate"].mean())
    best, rate = match_overrun(target, scenario_from_state("B"), seeds)
    if best.booking_rule == "monte_carlo":
        st.session_state["B_alpha_pct"] = int(round(100 * best.alpha))
        what = f"overrun limit set to {round(100 * best.alpha)}%"
    else:
        st.session_state["B_fill_pct"] = int(round(100 * best.fill_fraction))
        what = f"fill set to {round(100 * best.fill_fraction)}%"
    st.session_state["match_msg"] = (f"B's {what}: its lists overrun {100 * rate:.1f}% of the time, "
                                     f"against {100 * target:.1f}% for A.")


# ---------------------------------------------------------------- charts
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def weekly_charts(res, target_weeks: int, warmup: int = 26):
    wk = res.weekly.groupby("week")[["waiting", "waiting_within_6wk"]].mean().reset_index()
    wk["week"] += 1
    wk["within"] = 100 * wk["waiting_within_6wk"]
    rule = alt.Chart(pd.DataFrame({"week": [warmup + 1]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="week:Q")
    base = alt.Chart(wk).encode(x=alt.X("week:Q", title="Week (measured from week 27)"))
    size = base.mark_line(color=BLUE, strokeWidth=2).encode(
        y=alt.Y("waiting:Q", title="Patients waiting"),
        tooltip=[alt.Tooltip("week:Q", title="Week"), alt.Tooltip("waiting:Q", title="Waiting", format=".0f")])
    share = base.mark_line(color=BLUE, strokeWidth=2).encode(
        y=alt.Y("within:Q", title=f"Still waiting within {target_weeks} weeks (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("week:Q", title="Week"), alt.Tooltip("within:Q", title="Within target (%)", format=".1f")])
    return (size + rule).properties(height=260), (share + rule).properties(height=260)


def waits_chart(waits: np.ndarray, target_weeks: int):
    weeks = np.floor(waits / 7).astype(int)
    counts = pd.Series(weeks).value_counts().sort_index()
    df = pd.DataFrame({"weeks": counts.index, "share": 100 * counts.values / counts.sum()})
    df["group"] = np.where(df["weeks"] < target_weeks, f"Within {target_weeks} weeks", f"Over {target_weeks} weeks")
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("weeks:O", title="Weeks waited before surgery"),
        y=alt.Y("share:Q", title="Patients treated (%)"),
        color=alt.Color("group:N", scale=alt.Scale(domain=[f"Within {target_weeks} weeks", f"Over {target_weeks} weeks"],
                                                   range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("weeks:O", title="Weeks waited"), alt.Tooltip("share:Q", title="Patients (%)", format=".1f")],
    ).properties(height=260)


def needed_chart(curve: pd.DataFrame, threshold: float):
    long = curve.melt(id_vars="extra_lists", value_vars=MEASURES, var_name="measure", value_name="value")
    long["measure"] = long["measure"].map(MEASURE_LABELS)
    long["value"] *= 100
    lines = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2).encode(
        x=alt.X("extra_lists:Q", title="Extra lists per week"),
        y=alt.Y("value:Q", title="Within target (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("measure:N", scale=alt.Scale(domain=list(MEASURE_LABELS.values()), range=[BLUE, ORANGE, AQUA]),
                        legend=alt.Legend(title=None, orient="bottom", direction="vertical", labelLimit=400)),
        tooltip=["measure:N", alt.Tooltip("extra_lists:Q", title="Extra lists/week"),
                 alt.Tooltip("value:Q", title="Within target (%)", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"y": [100 * threshold]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return (lines + rule).properties(height=320)


def ab_needed_chart(ca: pd.DataFrame, cb: pd.DataFrame, threshold: float, target_weeks: int):
    df = pd.concat([ca.assign(scenario="A"), cb.assign(scenario="B")])
    df["value"] = 100 * df["cohort_within_6wk"]
    lines = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2).encode(
        x=alt.X("extra_lists:Q", title="Extra lists per week"),
        y=alt.Y("value:Q", title=f"Added to list: treated within {target_weeks} weeks (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("scenario:N", scale=alt.Scale(domain=["A", "B"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["scenario:N", alt.Tooltip("extra_lists:Q", title="Extra lists/week"),
                 alt.Tooltip("value:Q", title="Within target (%)", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"y": [100 * threshold]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    return (lines + rule).properties(height=320)


def fmt(v: float, dec: int, unit: str) -> str:
    if not np.isfinite(v):
        return "–"
    v = round(float(v), dec) + 0.0  # avoids "-0.0"
    return f"{v:,.{dec}f}{'%' if unit == '%' else ''}"


def signed(v: float, dec: int, suffix: str = "") -> str:
    if not np.isfinite(v):
        return "–"
    v = round(float(v), dec) + 0.0
    return ("+" if v > 0 else "") + f"{v:,.{dec}f}" + suffix


# ---------------------------------------------------------------- page
st.title("Theatre list booking simulator")
st.caption("Simulates a year of elective theatre booking for one specialty, week by week, on synthetic data. "
           "Every figure comes from invented, illustrative assumptions (see *Methodology*).")

with st.sidebar:
    st.header("Simulation")
    st.slider("Runs (synthetic hospitals)", 3, 20, key="seeds",
              help="More runs: smaller uncertainty, slower. Each run is 26 weeks' warm-up plus 52 measured weeks.")
    st.slider("Target: weeks from being added to surgery", 2, 18, key="target_weeks")
    st.slider("Target share treated within that time (%)", 80, 99, key="threshold_pct")
    st.header("Scenario A")
    controls("A")

seeds = st.session_state["seeds"]
tw = st.session_state["target_weeks"]
thr = st.session_state["threshold_pct"] / 100
scn_a = scenario_from_state("A")

tab_a, tab_cmp, tab_about = st.tabs(["Scenario A", "Compare A and B", "Methodology"])

with tab_a:
    with st.spinner("Simulating…"):
        ra = cached_run(asdict(scn_a), seeds)
    m = ra.per_seed.mean(numeric_only=True)
    st.markdown(f"**{SPECIALTY_NAMES[scn_a.specialty]}**: {describe(scn_a)}.")
    c = st.columns(4)
    c[0].metric("Patients treated a year", f"{m['treated_per_year']:,.0f}")
    c[1].metric("Still waiting: within target", pct(m["waiting_within_6wk"]),
                help="Share of patients on the list at the end of each week who have waited no longer than the "
                     "target. The way the national RTT standard is measured.")
    c[2].metric("Treated: within target", pct(m["treated_within_6wk"]),
                help="Of the patients operated on, the share who had waited no longer than the target.")
    c[3].metric("Added: treated within target", pct(m["cohort_within_6wk"]),
                help="Of everyone added to the list, the share operated on within the target. "
                     "Treating patients out of turn can't flatter it.")
    c = st.columns(4)
    c[0].metric("Touch-time utilisation", pct(m["touch_utilisation"]),
                help="Patient-in-theatre minutes inside the session, as a share of session time. "
                     "Changeovers and overruns don't count.")
    c[1].metric("Lists that overrun", pct(m["overrun_rate"]))
    c[2].metric("Cancelled on the day, per 100 booked", f"{m['cancelled_per_100_booked']:.1f}")
    c[3].metric("Median wait", f"{m['median_wait_days']:.0f} days")

    left, right = st.columns(2)
    size_chart, share_chart = weekly_charts(ra, tw)
    with left:
        st.subheader("Waiting list")
        st.altair_chart(size_chart, width="stretch")
    with right:
        st.subheader("Still waiting within target, by week")
        st.altair_chart(share_chart, width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("How long treated patients waited")
        st.altair_chart(waits_chart(ra.waits, tw), width="stretch")
    with right:
        st.subheader("Extra lists needed")
        if st.checkbox("Work out extra lists needed (about 5–10 seconds)", key="show_needed_a"):
            with st.spinner("Sweeping extra lists…"):
                na = cached_needed(asdict(scn_a), seeds, thr)
            st.altair_chart(needed_chart(na.curve, thr), width="stretch")
            rows = [{"Measure": MEASURE_LABELS[k], "Per week": fmt(v, 2, ""),
                     "Per year": fmt(52 * v, 0, "")} for k, v in na.needed.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(f"Extra lists on top of the surgeons' own weekly lists (this sweep ignores the "
                       f"'Extra lists per week' setting). A dash means not reached within {na.max_extra:g} a week.")
        else:
            st.caption("Sweeps 0 to 3 extra lists a week and finds where each measure reaches the target share.")

with tab_cmp:
    st.markdown("Scenario B starts from the settings below. Change what you want to test. Both scenarios run on "
                "the same synthetic hospitals, and on the same patients where the demand settings match, "
                "so differences reflect the change rather than chance.")
    b1, b2, _ = st.columns([1, 1.4, 2])
    b1.button("Copy A's settings into B", on_click=copy_a_to_b)
    b2.button("Match B's overrun rate to A's", on_click=match_b_to_a,
              help="Adjusts B's overrun limit (or fill %) so its lists overrun about as often as A's. "
                   "Compare like with like: otherwise a cautious scenario looks worse on waits simply "
                   "because it books less.")
    if st.session_state["match_msg"]:
        st.info(st.session_state["match_msg"])
    with st.expander("Scenario B settings", expanded=True):
        controls("B", two_columns=True)
    scn_b = scenario_from_state("B")
    with st.spinner("Simulating both scenarios…"):
        ra = cached_run(asdict(scn_a), seeds)
        rb = cached_run(asdict(scn_b), seeds)
    st.markdown(f"**A**: {SPECIALTY_NAMES[scn_a.specialty]}, {describe(scn_a)}.  \n"
                f"**B**: {SPECIALTY_NAMES[scn_b.specialty]}, {describe(scn_b)}.")
    cmp = compare(ra, rb)
    with_lists = st.checkbox("Include extra lists needed for both (about 10–20 seconds)", key="show_needed_ab")
    lists_cmp, na, nb = None, None, None
    if with_lists:
        with st.spinner("Sweeping extra lists for both scenarios…"):
            na = cached_needed(asdict(scn_a), seeds, thr)
            nb = cached_needed(asdict(scn_b), seeds, thr)
        lists_cmp = compare_lists_needed(na, nb)
    st.subheader("In short")
    st.markdown(headline(cmp, lists_cmp, tw, thr, scn_b.session_minutes, 3.0))

    st.subheader("Differences")
    table = pd.DataFrame({
        "Measure": cmp["measure"],
        "A": [fmt(v, d, u) for v, d, u in zip(cmp["A"], cmp["decimals"], cmp["unit"])],
        "B": [fmt(v, d, u) for v, d, u in zip(cmp["B"], cmp["decimals"], cmp["unit"])],
        "B − A": [signed(v, d, " pts" if u == "%" else "")
                  for v, d, u in zip(cmp["difference"], cmp["decimals"], cmp["unit"])],
        "± (2 standard errors)": [fmt(v, d, "") for v, d in zip(cmp["plus_minus"], cmp["decimals"])],
        "Clear difference?": ["Yes" if np.isfinite(pm) and abs(d) > pm else "No"
                              for d, pm in zip(cmp["difference"], cmp["plus_minus"])],
    })
    if lists_cmp is not None:
        extra = pd.DataFrame({
            "Measure": [f"Extra lists a year to reach {thr:.0%}: " + s.split(":")[0].lower() for s in lists_cmp["measure"]],
            "A": [fmt(v, 0, "") for v in lists_cmp["A"]],
            "B": [fmt(v, 0, "") for v in lists_cmp["B"]],
            "B − A": [signed(v, 0) for v in lists_cmp["difference"]],
            "± (2 standard errors)": [fmt(v, 0, "") for v in lists_cmp["plus_minus"]],
            "Clear difference?": ["Yes" if np.isfinite(pm) and abs(d) > pm else ("–" if not np.isfinite(d) else "No")
                                  for d, pm in zip(lists_cmp["difference"], lists_cmp["plus_minus"])],
        })
        table = pd.concat([table, extra], ignore_index=True)
    st.dataframe(table, hide_index=True, height=38 + 35 * len(table))
    st.caption(f"Means over {seeds} runs. Differences are paired run by run; '±' is two standard errors of the "
               "paired difference, so a difference smaller than it isn't distinguishable from noise.")
    if na is not None:
        st.subheader("Extra lists needed: A and B")
        st.altair_chart(ab_needed_chart(na.curve, nb.curve, thr, tw), width="stretch")

with tab_about:
    st.markdown(METHODOLOGY)
