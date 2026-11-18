"""Theatre list booking simulator: a Streamlit dashboard over the rolling simulation.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # works without installing the package

# Streamlit reruns this script in the same process after a deploy, with the package's old modules still
# imported. If any source file has changed since they were loaded, drop them so they are imported afresh.
_SOURCE_STAMP = max(f.stat().st_mtime for f in (ROOT / "src" / "theatre_sched").glob("*.py"))
_loaded = sys.modules.get("theatre_sched")
if _loaded is not None and getattr(_loaded, "_source_stamp", None) != _SOURCE_STAMP:
    for _name in [m for m in list(sys.modules) if m == "theatre_sched" or m.startswith("theatre_sched.")]:
        del sys.modules[_name]

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import theatre_sched  # noqa: E402
from theatre_sched.durations import fit_bayes, fit_hierarchical  # noqa: E402
from theatre_sched.rolling import POLICIES, RollingConfig, implied_load  # noqa: E402
from theatre_sched.scenarios import (  # noqa: E402
    MEASURE_LABELS,
    MEASURES,
    Scenario,
    compare,
    compare_lists_needed,
    describe,
    describe_start,
    headline,
    lists_needed,
    match_overrun,
    recovery_curve,
    run,
    seed_for,
)
from theatre_sched.starting_list import CountsError, parse_counts, summarise, to_scenario_bands  # noqa: E402
from theatre_sched.synthetic import SUBSPECIALTIES, WorldConfig, make_world, surgeon_subspecialties  # noqa: E402

theatre_sched._source_stamp = _SOURCE_STAMP

st.set_page_config(page_title="Theatre list booking simulator", page_icon="🩺", layout="wide")

METHODOLOGY_FILE = ROOT / "docs" / "METHODOLOGY.md"
METHODOLOGY = (METHODOLOGY_FILE.read_text(encoding="utf-8") if METHODOLOGY_FILE.exists()
               else "The methodology is in docs/METHODOLOGY.md in the project repository.")
SWEEP_FILE = ROOT / "results" / "history_sweep.csv"

# Reference chart palette (categorical slots 1-4), a neutral for reference lines, and status colours.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL, MUTED, GRID, BAND = "#52514e", "#898781", "#e1e0d9", "#f0efea"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
START_NAMES = {"today": "Today's waiting list", "settled": "A settled list (steady state)"}
DESCRIBE_NAMES = {"totals": "Totals", "surgeon": "Surgeon by surgeon", "counts": "Counts by weeks waited"}
POOLING_NAMES = {"own": "Each surgeon's own", "pooled": "Pooled within subspecialty"}
EXAMPLE_COUNTS = "weeks,patients\n0-6,130\n6-12,40\n12-18,18\n18-26,9\n26-40,3"
SPECIALTY_NAMES = {"general_surgery": "General surgery", "orthopaedics": "Orthopaedics",
                   "gynaecology": "Gynaecology", "ent": "ENT"}
POLICY_NAMES = {"in_turn": "In turn (urgent first, then longest wait)",
                "ratio_fill": "Due patients first, then fill gaps with short cases",
                "touch_fill": "Due patients first, then fill gaps with long cases"}
RULE_NAMES = {"average": "To average case times, up to a fill %",
              "monte_carlo": "To a Monte Carlo overrun limit"}
MODEL_NAMES = {"plugin": "Plug-in: each pair's own past cases",
               "hierarchical": "Hierarchical: partial pooling",
               "bayes": "Bayesian: partial pooling, spreads uncertain too",
               "oracle": "True durations (reference only)"}
DEFAULTS = {"specialty": "general_surgery", "n_surgeons": 6, "lists_per_surgeon": 1, "session_minutes": 480,
            "working_weeks": 42, "referrals": 24, "extra_lists": 0.5, "turnaround": 15, "policy": "in_turn",
            "booking_rule": "average", "fill_pct": 100, "alpha_pct": 10, "duration_model": "plugin",
            "overrun_tolerance": 30, "history_cases": 15, "pooling": "own",
            "start_mode": "today", "start_describe": "totals", "start_waiting": 200, "start_over_pct": 35,
            "start_longest": 30, "start_text": EXAMPLE_COUNTS, "horizon": 52}
B_START = {**DEFAULTS, "booking_rule": "monte_carlo"}
FIELDS = list(DEFAULTS)

for prefix, start in (("A", DEFAULTS), ("B", B_START)):
    for f, v in start.items():
        st.session_state.setdefault(f"{prefix}_{f}", v)
for k, v in {"seeds": 5, "target_weeks": 6, "threshold_pct": 95, "match_msg": ""}.items():
    st.session_state.setdefault(k, v)

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }}
[data-testid="stMetric"] {{ background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.75rem 1rem 0.8rem; }}
[data-testid="stMetricLabel"] p {{ color: {NEUTRAL}; font-size: 0.86rem; }}
.eyebrow {{ color: {NEUTRAL}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
            margin: 0.4rem 0 0.35rem; }}
.lede {{ color: #3a3936; font-size: 1.02rem; margin: -0.4rem 0 0.2rem; max-width: 62rem; }}
.badge {{ display: inline-block; font-size: 0.74rem; font-weight: 600; color: {NEUTRAL}; background: {BAND};
          border: 1px solid {GRID}; border-radius: 999px; padding: 0.08rem 0.55rem; margin-left: 0.45rem;
          vertical-align: middle; }}
.scenario-line {{ color: #3a3936; font-size: 0.95rem; margin: 0.2rem 0 0.9rem; }}
.kpi {{ background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.8rem 1rem 0.85rem; height: 100%; }}
.kpi .label {{ color: {NEUTRAL}; font-size: 0.86rem; line-height: 1.3; min-height: 2.2rem; }}
.kpi .value {{ color: #1d1d1b; font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 0.15rem 0 0.35rem;
               font-variant-numeric: tabular-nums; }}
.kpi .status {{ display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.82rem; color: #3a3936;
                background: {BAND}; border-radius: 999px; padding: 0.12rem 0.55rem 0.12rem 0.3rem; }}
.kpi .icon {{ display: inline-flex; width: 1.05rem; height: 1.05rem; border-radius: 50%; color: #fff;
              font-size: 0.72rem; font-weight: 700; align-items: center; justify-content: center; }}
.kpi .note {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.45rem; line-height: 1.35; }}
.callout {{ background: #ffffff; border: 1px solid {GRID}; border-left: 4px solid {BLUE}; border-radius: 8px;
            padding: 0.85rem 1.1rem; margin: 0.3rem 0 1rem; }}
.chart-note {{ color: {MUTED}; font-size: 0.8rem; margin: -0.3rem 0 0.4rem; }}
section[data-testid="stSidebar"][data-collapsed="false"] {{ min-width: 380px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached computation
@st.cache_data(show_spinner=False, max_entries=64)
def cached_run(scn: dict, seeds: int):
    return run(Scenario(**scn), seeds)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_needed(scn: dict, seeds: int, threshold: float):
    return lists_needed(Scenario(**scn), seeds, threshold)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_recovery(scn: dict, seeds: int):
    return recovery_curve(Scenario(**scn), seeds)


@st.cache_data(show_spinner=False, max_entries=64)
def settled_list(scn: dict) -> tuple[float, float]:
    """Average list size and share still waiting within target once the same settings have settled."""
    settled = Scenario(**{**scn, "start_waiting": None})
    m = run(settled, 3).per_seed.mean(numeric_only=True)
    return float(m["waiting_list_mean"]), float(m["waiting_within_6wk"])


@st.cache_data(show_spinner=False, max_entries=64)
def demand_share(specialty: str, n_surgeons: int, lists_per_surgeon: int, session_minutes: float,
                 working_weeks: int, referrals: float) -> float:
    """Theatre time the referrals need, as a share of the time available after leave (15-minute changeovers)."""
    world = make_world(WorldConfig(specialty=specialty, n_surgeons=n_surgeons, subspecialties=True),
                       seed=seed_for(specialty, 0))
    cfg = RollingConfig(lists_per_surgeon=lists_per_surgeon, session_minutes=session_minutes,
                        working_weeks=working_weeks, referrals_per_week=referrals)
    return implied_load(world, cfg)


@st.cache_data(show_spinner=False, max_entries=32)
def tau_posterior(specialty: str, n_surgeons: int, history: float) -> dict:
    """The grid Bayesian model's posterior for tau in the same synthetic hospital as ``duration_estimates``."""
    world = make_world(WorldConfig(specialty=specialty, n_surgeons=n_surgeons, history_median_cases=history,
                                   subspecialties=True), seed=seed_for(specialty, 0))
    b = fit_bayes(world.history, n_draws=1000, seed=0)
    step = float(b.tau_grid[1] - b.tau_grid[0])
    return {"grid": pd.DataFrame({"tau": b.tau_grid, "density": b.tau_post / step}),
            "q": np.quantile(b.tau, [0.05, 0.5, 0.95]), "pairs": len(b.pairs), "cases": int(b.pairs["n"].sum()),
            "eb_tau": fit_hierarchical(world.history).tau}


def tau_chart(post: dict, true_tau: float = 0.08):
    d = post["grid"][post["grid"]["tau"] <= max(0.4, float(post["q"][2]) * 1.3)]
    area = alt.Chart(d).mark_area(color=PINK, opacity=0.35, line={"color": PINK, "strokeWidth": 2}).encode(
        x=alt.X("tau:Q", title="τ: how far pairs differ from their procedure and surgeon averages (log scale)"),
        y=alt.Y("density:Q", title="Posterior density"),
        tooltip=[alt.Tooltip("tau:Q", format=".3f"), alt.Tooltip("density:Q", format=".1f")])
    marks = pd.DataFrame({"tau": [post["eb_tau"], true_tau],
                          "label": [f"Hierarchical estimate {post['eb_tau']:.3f}", f"True value {true_tau}"],
                          "dy": [-8, 8]})
    rule = alt.Chart(marks).mark_rule(strokeDash=[5, 4], strokeWidth=1.5).encode(
        x="tau:Q", color=alt.Color("label:N", scale=alt.Scale(range=[YELLOW, NEUTRAL]),
                                   legend=alt.Legend(title=None, orient="top")))
    return (area + rule).properties(height=260)


@st.cache_data(show_spinner=False, max_entries=32)
def duration_estimates(specialty: str, n_surgeons: int, history: float) -> tuple[pd.DataFrame, float, float, float]:
    """For one synthetic hospital: each surgeon-procedure pair's own average, hierarchical estimate and truth."""
    world = make_world(WorldConfig(specialty=specialty, n_surgeons=n_surgeons, history_median_cases=history,
                                   subspecialties=True), seed=seed_for(specialty, 0))
    model = fit_hierarchical(world.history)
    t = model.table.merge(world.true_params, on=["surgeon", "procedure"])
    t["own_minutes"] = np.exp(t["ybar"])
    t["hier_minutes"] = np.exp(t["mu_hat"])
    t["true_minutes"] = np.exp(t["mu"])
    t["prior_minutes"] = np.exp(t["prior_mean"])
    sigma = float(model.sigma_by_proc.median())
    k = sigma ** 2 / float(t["prior_var"].median())  # cases at which a pair's own data gets half the weight
    return t, model.tau, sigma, k


# ---------------------------------------------------------------- settings
def scenario_from_state(p: str) -> Scenario:
    g = lambda f: st.session_state[f"{p}_{f}"]  # noqa: E731
    return Scenario(
        specialty=g("specialty"), n_surgeons=int(g("n_surgeons")), history_cases=float(g("history_cases")),
        lists_per_surgeon=int(g("lists_per_surgeon")), session_minutes=float(g("session_minutes")),
        working_weeks=int(g("working_weeks")), referrals_per_week=float(g("referrals")),
        extra_lists=float(g("extra_lists")),
        turnaround=float(g("turnaround")), policy=g("policy"), booking_rule=g("booking_rule"),
        fill_fraction=g("fill_pct") / 100, alpha=g("alpha_pct") / 100, duration_model=g("duration_model"),
        overrun_tolerance=float(g("overrun_tolerance")), target_weeks=int(st.session_state["target_weeks"]),
        target_share=st.session_state["threshold_pct"] / 100,
        horizon_weeks=int(g("horizon")), subspecialties=True, pooling=g("pooling"),
        **start_fields(p),
    )


def proc_name(p: str) -> str:
    return p.replace("_", " ").replace("acl ", "ACL ")


def join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def subspecialty_caption(specialty: str, n_surgeons: int) -> str:
    """Who does what: 'S1, S4: rhinology (septoplasty, ...); ... Everyone does grommet insertion and tonsillectomy.'"""
    sub = SUBSPECIALTIES[specialty]
    groups: dict[str, list[str]] = {}
    for k, name in enumerate(surgeon_subspecialties(specialty, n_surgeons)):
        groups.setdefault(name, []).append(f"S{k + 1}")
    parts = [f"{', '.join(who)} {name}" + (f" ({', '.join(proc_name(q) for q in sub[name])})" if sub[name]
                                           else " (only the shared cases: their own operations aren't modelled)")
             for name, who in groups.items()]
    text = "Each surgeon has one subspecialty: " + "; ".join(parts) + "."
    if sub["common"]:
        text += f" Everyone does {join_and([proc_name(q) for q in sub['common']])}."
    return text


def surgeon_rows(p: str) -> pd.DataFrame:
    """The by-surgeon starting list as edited (or, before any edit, the totals split evenly)."""
    n = int(st.session_state[f"{p}_n_surgeons"])
    base = st.session_state.get(f"{p}_surgeon_base")
    if base is None or len(base) != n:
        total, over, longest = (int(st.session_state[f"{p}_{f}"]) for f in ("start_waiting", "start_over_pct", "start_longest"))
        counts = [total // n + (1 if i < total % n else 0) for i in range(n)]
        base = pd.DataFrame({"Surgeon": [f"S{i + 1}" for i in range(n)], "Waiting": counts,
                             "% over": [over] * n, "Longest": [longest] * n})
        st.session_state[f"{p}_surgeon_base"] = base
        st.session_state.pop(f"{p}_surgeon_editor", None)
        st.session_state.pop(f"{p}_surgeon_edited", None)
    return st.session_state.get(f"{p}_surgeon_edited", base)


def reset_surgeon_rows(p: str) -> None:
    for key in ("surgeon_base", "surgeon_edited", "surgeon_editor"):
        st.session_state.pop(f"{p}_{key}", None)


def start_fields(p: str) -> dict:
    """The Scenario fields describing where the run starts, from scenario p's settings."""
    g = lambda f: st.session_state[f"{p}_{f}"]  # noqa: E731
    if g("start_mode") != "today":
        return {"start_waiting": None}
    common = {"start_over_target_pct": float(g("start_over_pct")), "start_longest_weeks": float(g("start_longest"))}
    how = g("start_describe")
    if how == "surgeon":
        rows = surgeon_rows(p)
        by = tuple((int(r["Waiting"]), float(r["% over"]), float(r["Longest"])) for _, r in rows.iterrows())
        if sum(n for n, _, _ in by) > 0:
            return {**common, "start_waiting": sum(n for n, _, _ in by), "start_by_surgeon": by}
    elif how == "counts":
        try:
            bands = to_scenario_bands(parse_counts(g("start_text")), int(g("n_surgeons")))
            return {**common, "start_waiting": sum(b[3] for b in bands), "start_bands": bands}
        except CountsError:
            pass  # the error is shown by the controls; fall back to the totals
    return {**common, "start_waiting": int(g("start_waiting"))}


def controls(p: str, two_columns: bool = False) -> None:
    k = lambda f: f"{p}_{f}"  # noqa: E731
    left, right = st.columns(2) if two_columns else (st.container(), st.container())
    with left, st.expander("Specialty and theatre time", expanded=True):
        st.selectbox("Specialty", list(SPECIALTY_NAMES), format_func=SPECIALTY_NAMES.get, key=k("specialty"))
        st.slider("Surgeons", 2, 12, key=k("n_surgeons"),
                  help="Subspecialty interests are given out in turn: S1 the first, S2 the second, and so on.")
        st.caption(subspecialty_caption(st.session_state[k("specialty")], st.session_state[k("n_surgeons")]))
        st.slider("Lists per surgeon per week", 1, 3, key=k("lists_per_surgeon"))
        st.select_slider("List length (minutes)", options=[240, 360, 480, 600], key=k("session_minutes"))
        st.slider("Operating weeks a year", 32, 52, step=2, key=k("working_weeks"),
                  help="The rest is leave, taken in two-week blocks.")
        st.slider("Extra lists per week", 0.0, 3.0, step=0.25, key=k("extra_lists"),
                  help="Saturday lists, given to the surgeons with most patients about to pass target. "
                       "0.25 = one every four weeks.")
    with left, st.expander("Demand", expanded=True):
        st.slider("Referrals added to the waiting list per week", 4, 100, key=k("referrals"),
                  help="Decisions to operate for the whole specialty, shared between the surgeons so that each "
                       "one's referrals need about the same theatre time. Adding surgeons or lists adds capacity, "
                       "not demand.")
        share = demand_share(st.session_state[k("specialty")], st.session_state[k("n_surgeons")],
                             st.session_state[k("lists_per_surgeon")], st.session_state[k("session_minutes")],
                             st.session_state[k("working_weeks")], st.session_state[k("referrals")])
        st.caption(f"These need about **{100 * share:.0f}%** of the theatre time available after leave "
                   "(before any extra lists, at a 15-minute changeover).")
    with left, st.expander("Starting waiting list", expanded=True):
        st.radio("Start from", list(START_NAMES), format_func=START_NAMES.get, key=k("start_mode"), horizontal=True,
                 help="Today's list: simulate forward from the list you describe, measuring every week. Settled: "
                      "start small and run a 26-week warm-up first, so the results show where these settings end up "
                      "whatever the start.")
        settled = st.session_state[k("start_mode")] == "settled"
        tw_now = int(st.session_state["target_weeks"])
        st.radio("Describe today's list by", list(DESCRIBE_NAMES), format_func=DESCRIBE_NAMES.get,
                 key=k("start_describe"), disabled=settled)
        how = st.session_state[k("start_describe")]
        if st.session_state[k("start_longest")] <= tw_now:
            st.session_state[k("start_longest")] = tw_now + 1
        if how == "totals" or settled:
            st.slider("Patients waiting now", 0, 2000, step=10, key=k("start_waiting"), disabled=settled,
                      help="The whole specialty's list, shared evenly between the surgeons.")
            st.slider(f"Already waited more than {tw_now} weeks (%)", 0, 90, key=k("start_over_pct"), disabled=settled)
            st.slider("Longest wait now (weeks)", tw_now + 1, 104, key=k("start_longest"), disabled=settled,
                      help="Waits beyond the target thin out evenly up to this.")
        elif how == "surgeon":
            base = surgeon_rows(p)
            edited = st.data_editor(
                st.session_state[f"{p}_surgeon_base"], key=f"{p}_surgeon_editor", hide_index=True, width="stretch",
                disabled=["Surgeon"],
                column_config={
                    "Surgeon": st.column_config.TextColumn(width=58),
                    "Waiting": st.column_config.NumberColumn(min_value=0, max_value=2000, step=1, format="%d", width=74),
                    "% over": st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%d", width=74,
                                                            help=f"% already waited more than {tw_now} weeks"),
                    "Longest": st.column_config.NumberColumn(min_value=tw_now + 1, max_value=104, step=1, format="%d",
                                                                 width=80, help="Longest wait now, in weeks"),
                })
            st.session_state[f"{p}_surgeon_edited"] = edited
            pooled_now = st.session_state[k("pooling")] == "pooled"
            st.caption(f"{int(edited['Waiting'].sum()):,} patients in all. Edit any cell. "
                       + ("With pooled lists, other surgeons in the subspecialty can take these patients."
                          if pooled_now else "Each surgeon only treats their own list."))
            st.button("Reset to the totals, split evenly", key=f"{p}_reset_surgeons", on_click=reset_surgeon_rows,
                      args=(p,), help="Uses the totals from the 'Totals' option.")
            del base
        else:
            st.file_uploader("Load a CSV", type=["csv", "txt", "tsv"], key=k("counts_file"),
                             on_change=load_counts_file, args=(p,))
            st.text_area("…or paste counts", key=k("start_text"), height=150,
                         help="One line per band: 'weeks, patients' or 'surgeon, weeks, patients'. Weeks can be a band "
                              "(12-18), a whole number of weeks (7), or open-ended (52+, taken as 52 to 78). A header "
                              "line is optional, and a block copied from a spreadsheet works.")
            try:
                bands = parse_counts(st.session_state[k("start_text")])
                to_scenario_bands(bands, int(st.session_state[k("n_surgeons")]))
                sm = summarise(bands, tw_now)
                names = {b.surgeon for b in bands if b.surgeon}
                st.caption(f"Read **{sm['patients']:,} patients**, about {sm['over_target_pct']:.0f}% past {tw_now} weeks, "
                           f"the longest under {sm['longest_weeks']:.0f} weeks"
                           + (f", for {len(names)} surgeons (matched in order to S1, S2, …)" if names
                              else ", shared evenly between the surgeons.")
                           + (f"; S{len(names) + 1} onwards start with an empty list." if names and
                              len(names) < int(st.session_state[k("n_surgeons")]) else ("." if names else ""))
                           + " Waits are spread evenly within each band.")
            except CountsError as e:
                st.error(f"{e}. Using the totals instead.")
        st.select_slider("Weeks to simulate from today", options=[26, 52, 78, 104], key=k("horizon"),
                         disabled=settled)
    with right, st.expander("Booking", expanded=True):
        st.radio("Waiting lists", list(POOLING_NAMES), format_func=POOLING_NAMES.get, key=k("pooling"),
                 horizontal=True,
                 help="Pooled: a patient can go on the list of any surgeon who does their procedure, so the shared "
                      "cases are pooled across the specialty and the subspecialty cases within the subspecialty. "
                      "Each week, patients are taken in booking order and go on the earliest list with room. A case "
                      "takes as long as that surgeon usually takes, adjusted for how complex the patient is.")
        st.radio("How full lists are booked", list(RULE_NAMES), format_func=RULE_NAMES.get, key=k("booking_rule"))
        avg = st.session_state[k("booking_rule")] == "average"
        st.slider("Fill lists to (% of session)", 60, 120, step=5, key=k("fill_pct"), disabled=not avg)
        st.slider("Overrun limit per list (%)", 1, 50, key=k("alpha_pct"), disabled=avg,
                  help="A list is allowed only if the Monte Carlo estimate of its chance of overrunning "
                       "is at most this.")
        st.selectbox("Planner's duration model", list(MODEL_NAMES), format_func=MODEL_NAMES.get,
                     key=k("duration_model"), disabled=avg,
                     help="How the Monte Carlo planner estimates case times. See the Duration model tab.")
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
    edited = st.session_state.get("A_surgeon_edited", st.session_state.get("A_surgeon_base"))
    if edited is not None:
        st.session_state["B_surgeon_base"] = edited.copy()
        st.session_state["B_surgeon_edited"] = edited.copy()
        st.session_state.pop("B_surgeon_editor", None)
    st.session_state["match_msg"] = ""


def load_counts_file(p: str) -> None:
    f = st.session_state.get(f"{p}_counts_file")
    if f is not None:
        st.session_state[f"{p}_start_text"] = f.getvalue().decode("utf-8-sig", errors="replace")


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


# ---------------------------------------------------------------- formatting
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


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


def target_card(label: str, value: float, threshold: float, note: str) -> str:
    ok = value >= threshold
    icon, colour, word = ("✓", GOOD, "Meets") if ok else ("✕", CRITICAL, "Misses")
    shown = pct(value)
    if not ok and round(100 * value, 1) >= 100 * threshold:  # e.g. 94.96% would display as 95.0%
        shown = f"{100 * value:.2f}%"
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{shown}</div>'
            f'<span class="status"><span class="icon" style="background:{colour}">{icon}</span>'
            f'{word} the {threshold:.0%} target</span><div class="note">{note}</div></div>')


def more_fewer(v: float) -> str:
    return f"{abs(v):.1f} {'more' if v >= 0 else 'fewer'}"


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- charts
def _warmup_band(warmup: int, y_bottom: float, y_scale: alt.Scale):
    band = pd.DataFrame({"start": [0.5], "end": [warmup + 0.5], "y": [y_bottom], "label": ["Warm-up, not measured"]})
    rect = alt.Chart(band).mark_rect(color=BAND).encode(x="start:Q", x2="end:Q")
    text = alt.Chart(band).mark_text(align="left", baseline="bottom", dx=6, dy=-6, color=MUTED, fontSize=11).encode(
        x="start:Q", y=alt.Y("y:Q", scale=y_scale), text="label:N")
    return rect, text


def weekly_charts(res, target_weeks: int, threshold: float, warmup: int = 26):
    wk = res.weekly.groupby("week")[["waiting", "waiting_within_6wk"]].mean().reset_index()
    wk["week"] += 1
    wk["within"] = 100 * wk["waiting_within_6wk"]
    lo = float(np.floor(min(100 * threshold - 10, wk["within"].min() - 2) / 10) * 10)
    lo = max(0.0, lo)
    x = alt.X("week:Q", title="Week", scale=alt.Scale(domain=[1, int(wk["week"].max())], nice=False))
    rect, text = _warmup_band(warmup, 0, alt.Scale(domainMin=0))
    size = alt.Chart(wk).mark_line(color=BLUE, strokeWidth=2).encode(
        x=x, y=alt.Y("waiting:Q", title="Patients waiting", scale=alt.Scale(domainMin=0)),
        tooltip=[alt.Tooltip("week:Q", title="Week"), alt.Tooltip("waiting:Q", title="Waiting", format=".0f")])
    size_layers = rect + text + size
    rect, text = _warmup_band(warmup, lo, alt.Scale(domain=[lo, 100]))
    share = alt.Chart(wk).mark_line(color=BLUE, strokeWidth=2, clip=True).encode(
        x=x, y=alt.Y("within:Q", title=f"Within {target_weeks} weeks (%)", scale=alt.Scale(domain=[lo, 100])),
        tooltip=[alt.Tooltip("week:Q", title="Week"), alt.Tooltip("within:Q", title="Within target (%)", format=".1f")])
    goal = pd.DataFrame({"y": [100 * threshold], "label": [f"Target {threshold:.0%}"]})
    y_goal = alt.Y("y:Q", scale=alt.Scale(domain=[lo, 100]))
    rule = alt.Chart(goal).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(y=y_goal)
    rule_text = alt.Chart(goal).mark_text(align="left", dx=4, dy=12, color=NEUTRAL, fontSize=11).encode(
        x=alt.datum(warmup + 1), y=y_goal, text="label:N")
    return (size_layers.properties(height=250),
            (rect + text + rule + rule_text + share).properties(height=250))


def recovery_charts(res, target_weeks: int, threshold: float):
    """For a run from today's list: the list split into within and over target, and the share within target,
    starting at week 0 (today)."""
    wk = res.weekly.groupby("week")[["waiting", "over_target", "waiting_within_6wk"]].mean().reset_index()
    wk["week"] += 1
    p = res.per_seed
    now = pd.DataFrame({"week": [0], "waiting": [p["waiting_start"].mean()], "over_target": [p["over_target_start"].mean()],
                        "waiting_within_6wk": [float(np.mean(1 - p["over_target_start"] / p["waiting_start"].clip(lower=1)))]})
    wk = pd.concat([now, wk], ignore_index=True)
    wk["within_n"] = wk["waiting"] - wk["over_target"]
    wk["within"] = 100 * wk["waiting_within_6wk"]
    x = alt.X("week:Q", title="Weeks from today", scale=alt.Scale(domain=[0, int(wk["week"].max())], nice=False))

    within_lab, over_lab = f"Waited up to {target_weeks} weeks", f"Waited longer than {target_weeks} weeks"
    long = pd.concat([wk.assign(group=within_lab, n=wk["within_n"], order=0),
                      wk.assign(group=over_lab, n=wk["over_target"], order=1)])
    area = alt.Chart(long).mark_area(opacity=0.9, line={"strokeWidth": 1.5}).encode(
        x=x, y=alt.Y("n:Q", title="Patients waiting", stack=True),
        color=alt.Color("group:N", scale=alt.Scale(domain=[within_lab, over_lab], range=[BLUE, ORANGE]),
                        legend=alt.Legend(title=None, orient="top")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("week:Q", title="Week"), "group:N", alt.Tooltip("n:Q", title="Patients", format=".0f")])

    lo = max(0.0, float(np.floor(min(100 * threshold - 10, wk["within"].min() - 2) / 10) * 10))
    y_scale = alt.Scale(domain=[lo, 100])
    share = alt.Chart(wk).mark_line(color=BLUE, strokeWidth=2, clip=True).encode(
        x=x, y=alt.Y("within:Q", title=f"Within {target_weeks} weeks (%)", scale=y_scale),
        tooltip=[alt.Tooltip("week:Q", title="Week"), alt.Tooltip("within:Q", title="Within target (%)", format=".1f")])
    goal = pd.DataFrame({"y": [100 * threshold], "label": [f"Target {threshold:.0%}"]})
    rule = alt.Chart(goal).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(y=alt.Y("y:Q", scale=y_scale))
    rule_text = alt.Chart(goal).mark_text(align="right", dx=-4, dy=12, color=NEUTRAL, fontSize=11).encode(
        x=alt.datum(int(wk["week"].max())), y=alt.Y("y:Q", scale=y_scale), text="label:N")
    layers = rule + rule_text + share
    met = p["weeks_to_target"]
    if met.notna().all():
        m = pd.DataFrame({"week": [met.mean()], "y": [lo], "label": [f"Met from week {met.mean():.0f}"]})
        layers = (alt.Chart(m).mark_rule(color=GOOD, strokeWidth=1.5).encode(x="week:Q")
                  + alt.Chart(m).mark_text(align="left", dx=5, baseline="bottom", dy=-4, color="#3a3936", fontSize=11)
                  .encode(x="week:Q", y=alt.Y("y:Q", scale=y_scale), text="label:N") + layers)
    return area.properties(height=260), layers.properties(height=260)


def recovery_sweep_chart(curve: pd.DataFrame, deadline: int, horizon: int, seeds: int):
    d = curve.copy()
    d["all_runs"] = np.where(d["share_reaching"] >= 1, "Met in every run", "Not met in every run")
    d["runs_text"] = (d["share_reaching"] * seeds).round().astype(int).astype(str) + f" of {seeds} runs"
    d = d[d["weeks_to_target"].notna()]
    pts = alt.Chart(d).mark_line(strokeWidth=2, color=BLUE, point=False).encode(
        x=alt.X("extra_lists:Q", title="Extra lists per week"),
        y=alt.Y("weeks_to_target:Q", title="Weeks until the target is met", scale=alt.Scale(domain=[0, horizon]))
    ) + alt.Chart(d).mark_point(size=70, strokeWidth=2, color=BLUE).encode(
        x="extra_lists:Q", y="weeks_to_target:Q",
        fill=alt.Fill("all_runs:N", scale=alt.Scale(domain=["Met in every run", "Not met in every run"],
                                                    range=[BLUE, "#ffffff"]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("extra_lists:Q", title="Extra lists/week"),
                 alt.Tooltip("weeks_to_target:Q", title="Weeks to target (runs meeting it)", format=".1f"),
                 alt.Tooltip("runs_text:N", title="Met in")])
    line = pd.DataFrame({"y": [deadline], "label": [f"By week {deadline}"]})
    rule = alt.Chart(line).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(y="y:Q")
    text = alt.Chart(line).mark_text(align="right", dy=-7, color=NEUTRAL, fontSize=11).encode(
        x=alt.datum(float(curve["extra_lists"].max())), y="y:Q", text="label:N")
    return (rule + text + pts).properties(height=280)


def waits_chart(waits: np.ndarray, target_weeks: int):
    weeks = np.floor(waits / 7).astype(int)
    counts = pd.Series(weeks).value_counts().sort_index()
    df = pd.DataFrame({"weeks": counts.index, "share": 100 * counts.values / counts.sum()})
    within, over = f"Within {target_weeks} weeks", f"Over {target_weeks} weeks"
    df["group"] = np.where(df["weeks"] < target_weeks, within, over)
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("weeks:O", title="Weeks waited before surgery", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("share:Q", title="Patients treated (%)"),
        color=alt.Color("group:N", scale=alt.Scale(domain=[within, over], range=[BLUE, ORANGE]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("weeks:O", title="Weeks waited"), alt.Tooltip("share:Q", title="Patients (%)", format=".1f")],
    ).properties(height=250)


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
    rule = alt.Chart(pd.DataFrame({"y": [100 * threshold]})).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(y="y:Q")
    return (lines + rule).properties(height=300)


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
    rule = alt.Chart(pd.DataFrame({"y": [100 * threshold]})).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(y="y:Q")
    return (lines + rule).properties(height=300)


EST_ORDER = ["Plug-in (own cases)", "Hierarchical", "True (hidden)"]
TRUTH_SHAPE = "M-0.12,-1.1L0.12,-1.1L0.12,1.1L-0.12,1.1Z"  # a thin vertical bar


def shrinkage_chart(t: pd.DataFrame, procedure: str):
    d = t[t["procedure"] == procedure].copy()
    d["surgeon_label"] = d["surgeon"] + "  (" + d["n"].astype(str) + " past " + np.where(d["n"] == 1, "case", "cases") + ")"
    order = d.sort_values("true_minutes")["surgeon_label"].tolist()
    long = d.melt(id_vars=["surgeon_label", "n", "weight"], value_vars=["own_minutes", "hier_minutes", "true_minutes"],
                  var_name="estimate", value_name="minutes")
    long["estimate"] = long["estimate"].map(dict(zip(["own_minutes", "hier_minutes", "true_minutes"], EST_ORDER)))
    y = alt.Y("surgeon_label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    pull = alt.Chart(d).mark_rule(color=MUTED, strokeWidth=1.5).encode(
        y=y, x=alt.X("own_minutes:Q", title="Typical (median) case time, minutes", scale=alt.Scale(zero=False)),
        x2="hier_minutes:Q")
    pts = alt.Chart(long).mark_point(filled=True, size=120, opacity=1, strokeWidth=0).encode(
        y=y, x="minutes:Q",
        color=alt.Color("estimate:N", scale=alt.Scale(domain=EST_ORDER, range=[AQUA, YELLOW, NEUTRAL]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=320)),
        shape=alt.Shape("estimate:N", scale=alt.Scale(domain=EST_ORDER, range=["circle", "diamond", TRUTH_SHAPE]),
                        legend=None),
        tooltip=["surgeon_label:N", "estimate:N", alt.Tooltip("minutes:Q", format=".0f"),
                 alt.Tooltip("weight:Q", title="Weight on own data", format=".2f")])
    return (pull + pts).properties(height=max(160, 46 * len(d)))


def weight_chart(t: pd.DataFrame, k: float):
    n = np.unique(np.round(np.geomspace(1, max(250, t["n"].max()), 80)))
    curve = pd.DataFrame({"n": n, "w": n / (n + k)})
    line = alt.Chart(curve).mark_line(color=NEUTRAL, strokeDash=[5, 4], strokeWidth=1.5).encode(
        x=alt.X("n:Q", scale=alt.Scale(type="log"), title="Past cases for the surgeon and procedure (log scale)",
                axis=alt.Axis(values=[1, 3, 10, 30, 100, 300], format="d")),
        y=alt.Y("w:Q", title="Weight on the pair's own average", scale=alt.Scale(domain=[0, 1])))
    pts = alt.Chart(t).mark_circle(size=55, color=BLUE, opacity=0.85).encode(
        x="n:Q", y="weight:Q",
        tooltip=["surgeon:N", "procedure:N", "n:Q", alt.Tooltip("weight:Q", format=".2f")])
    half = pd.DataFrame({"n": [k], "w": [0.5], "label": [f"k ≈ {k:.0f}: half own data, half prior"]})
    mark = alt.Chart(half).mark_point(shape="circle", size=70, filled=False, color=NEUTRAL, strokeWidth=1.5).encode(x="n:Q", y="w:Q")
    # Label below right of the point when it sits towards the left, else above left (empty: the curve rises)
    place = {"align": "left", "dx": 10, "dy": 26} if k < 12 else {"align": "right", "dx": -8, "dy": -56}
    lab = alt.Chart(half).mark_text(**place, color=NEUTRAL, fontSize=11).encode(x="n:Q", y="w:Q", text="label:N")
    return (line + pts + mark + lab).properties(height=300)


SWEEP_NAMES = {"cc-plugin": "Plug-in", "cc-hier": "Hierarchical", "cc-bayes": "Bayesian (grid)",
               "cc-oracle": "True durations"}
PINK = "#e87ba4"
COVERAGE_FILE = ROOT / "results" / "coverage.csv"
PYMC_IMAGE = ROOT / "results" / "pymc_tau.png"


def sweep_chart(sweep: pd.DataFrame):
    d = sweep.assign(model=sweep["method"].map(SWEEP_NAMES), risk=100 * sweep["p90_list_risk"])
    order = list(SWEEP_NAMES.values())
    enc = dict(
        x=alt.X("history_cases:O", title="Typical past cases per surgeon and procedure", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("risk:Q", title="Realised P(overrun), riskiest 10% of lists (%)"),
        color=alt.Color("model:N", scale=alt.Scale(domain=order, range=[AQUA, YELLOW, PINK, NEUTRAL][:len(order)]),
                        legend=alt.Legend(title=None, orient="top", symbolType="circle")),
        tooltip=["model:N", alt.Tooltip("history_cases:O", title="Past cases"),
                 alt.Tooltip("risk:Q", title="Riskiest 10% of lists (%)", format=".1f")])
    point = alt.OverlayMarkDef(size=60, filled=True)
    models = alt.Chart(d[d.method != "cc-oracle"]).mark_line(point=point, strokeWidth=2).encode(**enc)
    oracle = alt.Chart(d[d.method == "cc-oracle"]).mark_line(point=point, strokeWidth=2, strokeDash=[5, 4]).encode(**enc)
    goal = pd.DataFrame({"y": [10.0], "label": ["Limit: 10%"]})
    rule = alt.Chart(goal).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(y="y:Q")
    lab = alt.Chart(goal).mark_text(align="left", dx=4, dy=-7, color=MUTED, fontSize=11).encode(
        x=alt.value(0), y="y:Q", text="label:N")
    return (rule + lab + models + oracle).properties(height=300)


# ---------------------------------------------------------------- page
st.title("Theatre list booking simulator")
st.markdown('<div class="lede">Elective theatre booking for one specialty, simulated week by week from today\'s '
            'waiting list: who gets booked, how full lists are, what happens on the day, how long patients wait, '
            'and when the waiting-time target is met. <span class="badge">Synthetic data</span></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Simulation")
    st.slider("Runs (synthetic hospitals)", 3, 20, key="seeds",
              help="More runs: smaller uncertainty, slower. Each run simulates the weeks from today, or a "
                   "26-week warm-up plus 52 measured weeks for a settled list.")
    st.slider("Target: weeks from being added to surgery", 2, 18, key="target_weeks")
    st.slider("Target share treated within that time (%)", 80, 99, key="threshold_pct")
    st.subheader("Scenario A")
    controls("A")

seeds = st.session_state["seeds"]
tw = st.session_state["target_weeks"]
thr = st.session_state["threshold_pct"] / 100
scn_a = scenario_from_state("A")

tab_a, tab_cmp, tab_dur, tab_about = st.tabs(["Scenario A", "Compare A and B", "Duration model", "Methodology"])

with tab_a:
    with st.spinner("Simulating…"):
        ra = cached_run(asdict(scn_a), seeds)
    m = ra.per_seed.mean(numeric_only=True)
    st.markdown(f'<div class="scenario-line"><b>{SPECIALTY_NAMES[scn_a.specialty]}</b> · {scn_a.n_surgeons} surgeons · '
                f'{scn_a.referrals_per_week:.0f} referrals a week · {describe(scn_a)}. Starting from '
                f'{describe_start(scn_a)}. Means over {seeds} runs.</div>', unsafe_allow_html=True)

    if scn_a.from_today:
        p = ra.per_seed
        H = scn_a.horizon_weeks
        eyebrow(f"From today's list, over the next {H} weeks")
        cols = st.columns(4)
        met = p["weeks_to_target"]
        if met.notna().all():
            value, ok = f"Week {met.mean():.0f}", True
            detail = (f"runs range from week {met.min():.0f} to {met.max():.0f}" if met.min() != met.max()
                      else "in every run")
        else:
            value, ok = "Not met", False
            detail = f"met within {H} weeks in {int(met.notna().sum())} of {len(met)} runs"
        icon, colour = ("✓", GOOD) if ok else ("✕", CRITICAL)
        cols[0].markdown(
            f'<div class="kpi"><div class="label">{thr:.0%} still waiting within {tw} weeks: first met</div>'
            f'<div class="value">{value}</div><span class="status"><span class="icon" style="background:{colour}">'
            f'{icon}</span>{"Met" if ok else "Not met"} within {H} weeks</span>'
            f'<div class="note">The first week from which it holds for 4 weeks running; {detail}.</div></div>',
            unsafe_allow_html=True)
        cols[1].markdown(
            f'<div class="kpi"><div class="label">Patients waiting longer than {tw} weeks</div>'
            f'<div class="value">{m["over_target_start"]:,.0f} → {m["over_target_week_26"]:,.0f}</div>'
            f'<div class="note">Today, then after 26 weeks. After 13 weeks: {m["over_target_week_13"]:,.0f}; '
            f'at week {H}: {m["over_target_end"]:,.0f}.</div></div>', unsafe_allow_html=True)
        settled_n, settled_within = settled_list(asdict(scn_a))
        cols[2].markdown(
            f'<div class="kpi"><div class="label">Patients waiting</div>'
            f'<div class="value">{m["waiting_start"]:,.0f} → {m["waiting_end"]:,.0f}</div>'
            f'<div class="note">Today, then at week {H}. Once settled, these settings hold about '
            f'{settled_n:,.0f} on the list.</div></div>', unsafe_allow_html=True)
        cleared = p["weeks_to_treat_start_list"]
        if cleared.notna().all():
            cv, cn = f"Week {cleared.mean():.0f}", "Everyone on today's list has been operated on by then."
        else:
            cv, cn = f"{100 * m['share_start_list_treated']:.0f}%", f"of today's list operated on by week {H}."
        cols[3].markdown(f'<div class="kpi"><div class="label">Today\'s list: all treated by</div>'
                         f'<div class="value">{cv}</div><div class="note">{cn}</div></div>', unsafe_allow_html=True)

        eyebrow("Where it ends up: the last 13 weeks")
        cols = st.columns(3)
        cards = [
            ("Still waiting: within target", m["waiting_within_6wk_final"],
             "Share of the list at each week's end who have waited no longer than the target (the national measure)."),
            ("Treated: had waited no longer than target", m["treated_within_6wk_final"],
             "Of the patients operated on in the last 13 weeks, the share who had waited no longer than the target."),
            ("Added during the run: treated within target", m["cohort_within_6wk"],
             "Of the new patients added in the run (not today's list), the share operated on within the target."),
        ]
        for c, (label, value, text) in zip(cols, cards):
            c.markdown(target_card(label, value, thr, text), unsafe_allow_html=True)
    else:
        eyebrow(f"Waiting times: the {tw}-week target, three ways")
        cols = st.columns(3)
        cards = [
            ("Still waiting: within target", m["waiting_within_6wk"],
             "Share of the list at each week's end who have waited no longer than the target. How the national RTT "
             "standard is measured."),
            ("Treated: had waited no longer than target", m["treated_within_6wk"],
             "Of the patients operated on, the share who had waited no longer than the target."),
            ("Added to the list: treated within target", m["cohort_within_6wk"],
             "Of everyone added, the share operated on within the target. Treating patients out of turn can't flatter it."),
        ]
        for c, (label, value, text) in zip(cols, cards):
            c.markdown(target_card(label, value, thr, text), unsafe_allow_html=True)

    eyebrow("Theatre use")
    c = st.columns(5)
    c[0].metric("Patients treated a year", f"{m['treated_per_year']:,.0f}")
    c[1].metric("Touch-time utilisation", pct(m["touch_utilisation"]),
                help="Patient-in-theatre minutes inside the session, as a share of session time. "
                     "Changeovers and overruns don't count. The national target is 85%.")
    c[2].metric("Lists that overrun", pct(m["overrun_rate"]))
    c[3].metric("Cancelled on the day", f"{m['cancelled_per_100_booked']:.1f}",
                help="Cases cancelled on the day per 100 booked, because the list was expected to run too late.")
    c[4].metric("Median wait", f"{m['median_wait_days']:.0f} days")

    st.write("")
    left, right = st.columns(2, gap="large")
    if scn_a.from_today:
        size_chart, share_chart = recovery_charts(ra, tw, thr)
        size_note = "Patients waiting at the end of each week, split by how long they've waited. Week 0 is today."
    else:
        size_chart, share_chart = weekly_charts(ra, tw, thr)
        size_note = "Patients waiting at the end of each week, averaged over runs."
    with left:
        st.subheader("Waiting list")
        note(size_note)
        st.altair_chart(size_chart, width="stretch")
    with right:
        st.subheader("Still waiting within target, by week")
        note("The national-style measure, week by week.")
        st.altair_chart(share_chart, width="stretch")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("How long treated patients waited")
        note(f"Patients operated on in the {scn_a.horizon_weeks if scn_a.from_today else 52} measured weeks"
             + (", including today's list." if scn_a.from_today else "."))
        st.altair_chart(waits_chart(ra.waits, tw), width="stretch")
    with right:
        if scn_a.from_today:
            st.subheader("Recovering faster with extra lists")
            if st.toggle("Work out the effect of extra lists (a few seconds)", key="show_needed_a"):
                with st.spinner("Sweeping extra lists…"):
                    rec = cached_recovery(asdict(scn_a), seeds)
                deadline = st.slider("Meet the target by week", 4, scn_a.horizon_weeks,
                                     value=min(26, scn_a.horizon_weeks), step=1, key="deadline_a")
                need = rec.lists_to_meet_by(deadline)
                note(f"Weeks until {thr:.0%} of the list is within {tw} weeks, against extra lists a week "
                     "(hollow points: not met in every run).")
                st.altair_chart(recovery_sweep_chart(rec.curve, deadline, scn_a.horizon_weeks, seeds), width="stretch")
                if np.isfinite(need):
                    st.markdown(f"To meet the target by **week {deadline}**: about **{need:.2f} extra lists a week** "
                                f"on top of the surgeons' own lists, or {need * deadline:.0f} lists in all by then.")
                else:
                    st.markdown(f"Not every run meets the target by week {deadline}, even with "
                                f"{rec.curve['extra_lists'].max():g} extra lists a week.")
                st.caption("This sweep replaces the 'Extra lists per week' setting with each value in turn.")
            else:
                note("Sweeps 0 to 3 extra lists a week and shows how soon the target is met with each.")
        else:
            st.subheader("Extra lists needed")
            if st.toggle("Work out extra lists needed (about 5–10 seconds)", key="show_needed_a"):
                with st.spinner("Sweeping extra lists…"):
                    na = cached_needed(asdict(scn_a), seeds, thr)
                note(f"Each measure against extra lists a week; the dashed line is the {thr:.0%} target.")
                st.altair_chart(needed_chart(na.curve, thr), width="stretch")
                rows = [{"Measure": MEASURE_LABELS[k], "Per week": fmt(v, 2, ""),
                         "Per year": fmt(52 * v, 0, "")} for k, v in na.needed.items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(f"Extra lists on top of the surgeons' own weekly lists (this sweep ignores the "
                           f"'Extra lists per week' setting). A dash means not reached within {na.max_extra:g} a week.")
            else:
                note("Sweeps 0 to 3 extra lists a week and finds where each measure reaches the target share.")

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
    if scn_a.from_today != scn_b.from_today:
        st.warning("A and B start from different places (one from today's list, one from a settled list), so they "
                   "can't be compared run by run. Set both to the same starting point under Starting waiting list.")
    else:
        with st.spinner("Simulating both scenarios…"):
            ra = cached_run(asdict(scn_a), seeds)
            rb = cached_run(asdict(scn_b), seeds)
        cmp = compare(ra, rb)
        today = scn_a.from_today
        with_lists = st.toggle("Include the effect of extra lists for both (a few seconds)" if today else
                               "Include extra lists needed for both (about 10–20 seconds)", key="show_needed_ab")
        lists_cmp, na, nb, rec_a, rec_b = None, None, None, None, None
        if with_lists and today:
            with st.spinner("Sweeping extra lists for both scenarios…"):
                rec_a = cached_recovery(asdict(scn_a), seeds)
                rec_b = cached_recovery(asdict(scn_b), seeds)
        elif with_lists:
            with st.spinner("Sweeping extra lists for both scenarios…"):
                na = cached_needed(asdict(scn_a), seeds, thr)
                nb = cached_needed(asdict(scn_b), seeds, thr)
            lists_cmp = compare_lists_needed(na, nb)
        starts = (f"Both start from {describe_start(scn_a)}." if today and describe_start(scn_a) == describe_start(scn_b)
                  else (f"A starts from {describe_start(scn_a)}; B from {describe_start(scn_b)}." if today else ""))
        st.markdown(
            f'<div class="callout"><div class="eyebrow" style="margin-top:0">In short</div>'
            f'{headline(cmp, lists_cmp, tw, thr, scn_b.session_minutes, 3.0)}'
            f'<div class="chart-note" style="margin:0.5rem 0 0">A: {SPECIALTY_NAMES[scn_a.specialty]}, {describe(scn_a)}.<br>'
            f'B: {SPECIALTY_NAMES[scn_b.specialty]}, {describe(scn_b)}.' + (f"<br>{starts}" if starts else "")
            + '</div></div>', unsafe_allow_html=True)

        st.subheader("Differences")
        table = pd.DataFrame({
            "Measure": cmp["measure"],
            "A": [fmt(v, d, u) for v, d, u in zip(cmp["A"], cmp["decimals"], cmp["unit"])],
            "B": [fmt(v, d, u) for v, d, u in zip(cmp["B"], cmp["decimals"], cmp["unit"])],
            "B − A": [signed(v, d, " pts" if u == "%" else "")
                      for v, d, u in zip(cmp["difference"], cmp["decimals"], cmp["unit"])],
            "± (2 standard errors)": [fmt(v, d, "") for v, d in zip(cmp["plus_minus"], cmp["decimals"])],
            "Clear difference?": ["Yes" if np.isfinite(pm) and abs(d) > pm else ("–" if not np.isfinite(d) else "No")
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
        st.dataframe(table, hide_index=True, height=38 + 35 * len(table), width="stretch")
        st.caption(f"Means over {seeds} runs. Differences are paired run by run; '±' is two standard errors of the "
                   "paired difference, so a difference smaller than it isn't distinguishable from noise. A dash means "
                   "the measure isn't defined in every run (for example, the target is never met).")
        if na is not None:
            st.subheader("Extra lists needed: A and B")
            st.altair_chart(ab_needed_chart(na.curve, nb.curve, thr, tw), width="stretch")
        if rec_a is not None:
            st.subheader("Recovering faster with extra lists: A and B")
            H = min(scn_a.horizon_weeks, scn_b.horizon_weeks)
            deadline = st.slider("Meet the target by week", 4, H, value=min(26, H), key="deadline_ab")
            need_a, need_b = rec_a.lists_to_meet_by(deadline), rec_b.lists_to_meet_by(deadline)
            say = lambda v: f"{v:.2f} extra lists a week" if np.isfinite(v) else "more than 3 extra lists a week"  # noqa: E731
            st.markdown(f"To meet the target by week {deadline}, A needs about **{say(need_a)}** and B about "
                        f"**{say(need_b)}**.")
            both = pd.concat([rec_a.curve.assign(scenario="A"), rec_b.curve.assign(scenario="B")])
            both = both[both["weeks_to_target"].notna()]
            chart = alt.Chart(both).mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2).encode(
                x=alt.X("extra_lists:Q", title="Extra lists per week"),
                y=alt.Y("weeks_to_target:Q", title="Weeks until the target is met", scale=alt.Scale(domain=[0, H])),
                color=alt.Color("scenario:N", scale=alt.Scale(domain=["A", "B"], range=[BLUE, ORANGE]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["scenario:N", alt.Tooltip("extra_lists:Q", title="Extra lists/week"),
                         alt.Tooltip("weeks_to_target:Q", title="Weeks to target", format=".1f"),
                         alt.Tooltip("share_reaching:Q", title="Share of runs meeting it", format=".0%")])
            rule = alt.Chart(pd.DataFrame({"y": [deadline]})).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart((rule + chart).properties(height=300), width="stretch")
            note("Each point is the mean over the runs that meet the target; hover for the share of runs that do.")

with tab_dur:
    st.markdown(
        "The Monte Carlo planner needs each surgeon's typical time for each procedure. The **plug-in** model uses "
        "only that surgeon's own past cases of it, and treats the result as exact, even when it rests on one or two "
        "cases. The **hierarchical** model blends the pair's own average with what you'd expect from the procedure "
        "and the surgeon's speed on their other procedures, giving the pair's own data more weight the more of it "
        "there is. It also carries the remaining uncertainty into the Monte Carlo draws, shared by all of that "
        "surgeon's cases of the procedure on a list.")
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        spec = c1.selectbox("Specialty", list(SPECIALTY_NAMES), format_func=SPECIALTY_NAMES.get,
                            index=list(SPECIALTY_NAMES).index(scn_a.specialty), key="dur_specialty")
        hist = c2.select_slider("Typical past cases per pair", options=[3, 15, 50, 200], value=15, key="dur_history")
        est, tau, sigma, k = duration_estimates(spec, 6, float(hist))
        procs = list(dict.fromkeys(est.sort_values("true_minutes")["procedure"]))
        proc = c3.selectbox("Procedure", procs, format_func=lambda p: p.replace("_", " ").capitalize(),
                            index=min(2, len(procs) - 1), key="dur_procedure")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("One procedure, every surgeon")
        note("Each grey line runs from the surgeon's own average to the hierarchical estimate: the pull towards "
             "the prior. Pairs with few past cases are pulled furthest.")
        st.altair_chart(shrinkage_chart(est, proc), width="stretch")
    with right:
        st.subheader("How much weight the pair's own data gets")
        note(f"w = n / (n + k), with k = σ² / prior variance. σ ≈ {sigma:.2f} is how much single cases vary; "
             f"the prior variance is τ² (τ ≈ {tau:.2f}, how far pairs differ from the prior) plus the uncertainty in the "
             f"prior itself. All on the log scale. Each dot is one pair; the line uses typical values.")
        st.altair_chart(weight_chart(est, k), width="stretch")

    own_err = np.sqrt(np.mean(np.log(est["own_minutes"] / est["true_minutes"]) ** 2))
    hier_err = np.sqrt(np.mean(np.log(est["hier_minutes"] / est["true_minutes"]) ** 2))
    st.caption(f"In this synthetic hospital, the typical error in a pair's estimated time is {100 * own_err:.0f}% from "
               f"its own cases alone and {100 * hier_err:.0f}% from the hierarchical model (root mean square, log scale).")

    st.subheader("When τ itself is uncertain: the Bayesian model")
    post = tau_posterior(spec, 6, float(hist))
    lo, med, hi = post["q"]
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.altair_chart(tau_chart(post), width="stretch")
        note(f"This hospital: {post['cases']:,} past cases in {post['pairs']} surgeon–procedure pairs. The posterior "
             f"has its median at {med:.3f} and a 90% interval from {lo:.3f} to {hi:.3f}.")
    with right:
        st.markdown(
            "The hierarchical model estimates τ once and then treats that estimate as exact, as it does each "
            "procedure's spread σ. With few past cases, both are uncertain, so its intervals for each pair are too "
            "narrow. The **Bayesian** model keeps the same structure but averages over every plausible τ and σ. "
            "Without MCMC: σ is drawn from a standard formula, τ from its posterior worked out on a grid of 200 values "
            "(the curve on the left), and everything else follows exactly. Choose it as the planner's duration model "
            "under Booking in the sidebar.")
        if COVERAGE_FILE.exists():
            cov = pd.read_csv(COVERAGE_FILE)
            ct = (cov.groupby(["history_cases", "model"])["covered"].mean().unstack() * 100).round(1)
            ct = ct.rename(columns={"Hierarchical (empirical Bayes)": "Hierarchical", "Bayesian (grid)": "Bayesian"})
            ct = ct[[c for c in ("Hierarchical", "Bayesian") if c in ct.columns]]
            ct = ct.reset_index().rename(columns={"history_cases": "Typical past cases"})
            st.caption("90% intervals for each pair's typical time that contain the true value (%), 30 synthetic "
                       "hospitals per row (scripts/run_coverage.py)")
            st.dataframe(ct, hide_index=True, width="stretch")
    if PYMC_IMAGE.exists():
        with st.expander("Checked against MCMC (PyMC)"):
            st.image(str(PYMC_IMAGE), width="stretch")
            st.caption("The same model fitted by MCMC with PyMC (scripts/check_pymc.py) gives the same posterior for τ "
                       "as the grid. Written in the 'centred' form, the sampler struggles when τ is small (190 "
                       "divergent transitions and a biased posterior near zero); in the 'non-centred' form it doesn't.")

    st.subheader("Does it help the planner hit its limit?")
    if SWEEP_FILE.exists():
        sweep = pd.read_csv(SWEEP_FILE)
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            note("Part 1 of the project: four weeks of lists for 5 surgeons, chance-constrained booking with a 10% "
                 "overrun limit, scored on the true durations. Mean of 20 synthetic hospitals per point.")
            st.altair_chart(sweep_chart(sweep), width="stretch")
        with right:
            tbl = sweep.assign(model=sweep["method"].map(SWEEP_NAMES)).pivot(
                index="history_cases", columns="model", values="p90_list_risk")
            tbl = tbl[[c for c in SWEEP_NAMES.values() if c in tbl.columns]]
            tbl = (100 * tbl).map(lambda v: f"{v:.1f}%").reset_index().rename(columns={"history_cases": "Past cases"})
            st.caption("Riskiest 10% of lists: realised chance of overrunning, against a 10% limit")
            st.dataframe(tbl, hide_index=True, width="stretch")
            h15 = sweep[sweep.history_cases == 15].set_index("method")
            h3 = sweep[sweep.history_cases == 3].set_index("method")
            st.markdown(
                f"With a typical 15 past cases per pair, the plug-in planner's riskiest tenth of lists overrun "
                f"{100 * h15.at['cc-plugin', 'p90_list_risk']:.0f}% of the time against a 10% limit. The hierarchical "
                f"planner brings that to {100 * h15.at['cc-hier', 'p90_list_risk']:.0f}%"
                + (f" and the Bayesian planner to {100 * h15.at['cc-bayes', 'p90_list_risk']:.0f}%"
                   if "cc-bayes" in h15.index else "")
                + f", and a planner that knew "
                f"the true durations would manage {100 * h15.at['cc-oracle', 'p90_list_risk']:.0f}%. Compared with "
                f"booking to average at the same overrun rate, the plug-in planner books "
                f"{more_fewer(h15.at['cc-plugin', 'extra_vs_avg_ratio'])} patients in four weeks and the hierarchical "
                f"one {more_fewer(h15.at['cc-hier', 'extra_vs_avg_ratio'])}."
                + (f" The Bayesian model matters most when history is thin: with a typical 3 past cases per pair, the "
                   f"riskiest tenth of lists overrun {100 * h3.at['cc-plugin', 'p90_list_risk']:.0f}% of the time with the "
                   f"plug-in planner, {100 * h3.at['cc-hier', 'p90_list_risk']:.0f}% with the hierarchical and "
                   f"{100 * h3.at['cc-bayes', 'p90_list_risk']:.0f}% with the Bayesian." if "cc-bayes" in h3.index else ""))
            st.caption("Try it in the year-long simulation: choose the Monte Carlo booking rule and a duration model "
                       "under Booking in the sidebar.")
    else:
        st.caption("Run scripts/run_history_sweep.py to produce these results.")

with tab_about:
    st.markdown(METHODOLOGY)
