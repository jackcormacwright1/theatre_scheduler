"""Fit the one-spread hierarchical duration model with PyMC (MCMC) and check it against the grid.

The grid model (``theatre_sched.durations.fit_bayes``) and this script fit the same model with the
same priors, so their posteriors should agree. The grid is the answer key; this checks the MCMC
machinery against it and shows its diagnostics:

    log minutes of case k:   y_k ~ Normal(theta_i, sigma_p)        (i its pair, p its procedure)
    pair averages:           theta_i = alpha_p + beta_s + tau * z_i,  z_i ~ Normal(0, 1)
    priors:                  tau ~ Uniform(0, 1);  log sigma_p ~ Uniform(log 0.01, log 2)  (so p(sigma) ~ 1/sigma);
                             alpha_p ~ Normal(4, 10) and beta_s ~ Normal(0, 10), wide enough to be flat

The surgeon speeds beta_s sum to zero, as in the grid model. Writing theta_i = mean + tau * z_i
(the "non-centred" form) rather than theta_i ~ Normal(mean, tau) (the "centred" form) is what lets
the sampler cope when tau is small; ``--centred`` fits the other form too, to show the difference.

Writes results/pymc_check.md and results/pymc_tau.png. PyMC is not needed for anything else in
the project (pip install pymc).

Usage:  python scripts/check_pymc.py [--history 3] [--seed 1] [--centred]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theatre_sched import WorldConfig, make_world  # noqa: E402
from theatre_sched.durations import fit_bayes  # noqa: E402


def split_rhat(x: np.ndarray) -> float:
    """Split R-hat for draws of shape (chains, draws): each chain is cut in half, and the spread
    between the halves' means is compared with the spread within them. Near 1 when they agree;
    above about 1.01 suggests the chains haven't mixed."""
    c, n = x.shape
    h = n // 2
    halves = np.concatenate([x[:, :h], x[:, h:2 * h]])
    m, n2 = halves.shape
    within = halves.var(axis=1, ddof=1).mean()
    between = n2 * halves.mean(axis=1).var(ddof=1)
    var_hat = (n2 - 1) / n2 * within + between / n2
    return float(np.sqrt(var_hat / within))


def ess(x: np.ndarray) -> float:
    """Effective sample size for draws of shape (chains, draws): how many independent draws the
    correlated chains are worth, from their autocorrelations (Geyer's initial positive sequence)."""
    c, n = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    var = x.var(axis=1).mean()
    if var == 0:
        return float(c * n)
    acf = np.mean([np.correlate(ch, ch, "full")[n - 1:] / (n * ch.var()) for ch in x], axis=0)
    s = 0.0
    for t in range(1, n - 1, 2):
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        s += pair
    return float(c * n / (1 + 2 * s))


def build(history: pd.DataFrame, centred: bool):
    import pymc as pm

    h = history.assign(y=np.log(history["minutes"]))
    pairs = h.groupby(["surgeon", "procedure"]).size().reset_index()[["surgeon", "procedure"]]
    procs = sorted(pairs["procedure"].unique())
    surgeons = sorted(pairs["surgeon"].unique(), key=lambda s: (len(s), s))
    pair_idx = pd.MultiIndex.from_frame(h[["surgeon", "procedure"]]).map(
        {k: i for i, k in enumerate(zip(pairs["surgeon"], pairs["procedure"]))}).to_numpy()
    p_of_pair = pairs["procedure"].map({p: i for i, p in enumerate(procs)}).to_numpy()
    s_of_pair = pairs["surgeon"].map({s: i for i, s in enumerate(surgeons)}).to_numpy()
    p_of_case = p_of_pair[pair_idx]
    with pm.Model() as model:
        tau = pm.Uniform("tau", 0, 1)
        log_sigma = pm.Uniform("log_sigma", np.log(0.01), np.log(2), shape=len(procs))
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        alpha = pm.Normal("alpha", 4, 10, shape=len(procs))
        b = pm.Normal("b", 0, 10, shape=len(surgeons) - 1)
        beta = pm.Deterministic("beta", pm.math.concatenate([b, -pm.math.sum(b, keepdims=True)]))
        mean = alpha[p_of_pair] + beta[s_of_pair]
        if centred:
            theta = pm.Normal("theta", mean, tau, shape=len(pairs))
        else:
            z = pm.Normal("z", 0, 1, shape=len(pairs))
            theta = pm.Deterministic("theta", mean + tau * z)
        pm.Normal("y", theta[pair_idx], sigma[p_of_case], observed=h["y"].to_numpy())
    return model, pairs


def fit(history: pd.DataFrame, centred: bool, seed: int) -> dict:
    import pymc as pm
    model, pairs = build(history, centred)
    t0 = time.time()
    with model:
        idata = pm.sample(1000, tune=1000, chains=4, cores=1, target_accept=0.9, random_seed=seed,
                          progressbar=False, compute_convergence_checks=False)
    post = idata["posterior"]
    tau = post["tau"].values  # (chains, draws)
    theta = post["theta"].values  # (chains, draws, pairs)
    div = int(np.asarray(idata["sample_stats"]["diverging"].values).sum())
    rhats = [split_rhat(tau)] + [split_rhat(theta[:, :, i]) for i in range(theta.shape[2])]
    esss = [ess(tau)] + [ess(theta[:, :, i]) for i in range(theta.shape[2])]
    return {"tau": tau.ravel(), "theta": theta.reshape(-1, theta.shape[2]), "pairs": pairs,
            "divergences": div, "max_rhat": max(rhats), "min_ess": min(esss), "ess_tau": esss[0],
            "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=float, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--centred", action="store_true", help="also fit the centred form")
    args = ap.parse_args()
    w = make_world(WorldConfig(history_median_cases=args.history), seed=args.seed)
    grid = fit_bayes(w.history, n_draws=4000, seed=args.seed)
    nc = fit(w.history, centred=False, seed=args.seed)
    runs = {"Non-centred": nc}
    if args.centred:
        runs["Centred"] = fit(w.history, centred=True, seed=args.seed)

    # Compare pair averages: grid vs MCMC, matched by surgeon and procedure
    g_iv = grid.interval(0.90).set_index(["surgeon", "procedure"])
    m_theta = pd.DataFrame(nc["theta"], columns=pd.MultiIndex.from_frame(nc["pairs"]))
    m_med = m_theta.median().reindex(g_iv.index)
    m_lo, m_hi = m_theta.quantile(0.05).reindex(g_iv.index), m_theta.quantile(0.95).reindex(g_iv.index)
    true_mu = w.true_params.set_index(["surgeon", "procedure"])["mu"].reindex(g_iv.index)

    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    lines = ["# One-spread model: grid against PyMC\n",
             f"One synthetic world (seed {args.seed}), a typical {args.history:g} past cases per surgeon-procedure pair, "
             f"{len(w.history)} past cases in {len(g_iv)} pairs. Grid: 4,000 draws. PyMC: NUTS, 4 chains of 1,000 draws "
             "after 1,000 tuning steps each.\n",
             "## tau (how far pairs differ from their procedure and surgeon averages, log scale; true value 0.08)\n",
             "| | 5% | 25% | median | 75% | 95% |", "|---|---|---|---|---|---|",
             "| Grid | " + " | ".join(f"{v:.3f}" for v in np.quantile(grid.tau, q)) + " |"]
    for name, r in runs.items():
        lines.append(f"| PyMC, {name.lower()} | " + " | ".join(f"{v:.3f}" for v in np.quantile(r["tau"], q)) + " |")
    lines += ["", "## Pair averages (log scale)\n",
              f"- Largest difference between the grid's and PyMC's posterior medians: {np.max(np.abs(m_med - g_iv['median'])):.3f} "
              f"(the pair averages themselves range from {g_iv['median'].min():.2f} to {g_iv['median'].max():.2f}).",
              f"- Largest difference in the ends of the 90% intervals: "
              f"{max(np.max(np.abs(m_lo - g_iv['low'])), np.max(np.abs(m_hi - g_iv['high']))):.3f}.",
              f"- 90% intervals containing the true value: grid {np.mean((g_iv['low'] <= true_mu) & (true_mu <= g_iv['high'])):.0%}, "
              f"PyMC {np.mean((m_lo <= true_mu) & (true_mu <= m_hi)):.0%}.", "",
              "## Diagnostics\n", "| Form | Divergences | Largest R-hat | Smallest effective sample size | ESS for tau | Seconds |",
              "|---|---|---|---|---|---|"]
    for name, r in runs.items():
        lines.append(f"| {name} | {r['divergences']} | {r['max_rhat']:.3f} | {r['min_ess']:.0f} | {r['ess_tau']:.0f} | {r['seconds']:.0f} |")
    lines.append("\nThe grid fit takes about a second.")
    out = ROOT / "results"
    suffix = "" if args.history == 3 else f"_{args.history:g}"
    (out / f"pymc_check{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    step = grid.tau_grid[1] - grid.tau_grid[0]
    for name, r in runs.items():
        ax.hist(r["tau"], bins=np.arange(0, max(0.5, r["tau"].max()) + 0.01, 0.01), density=True, alpha=0.45,
                color="#2a78d6" if name == "Non-centred" else "#eb6834", label=f"PyMC ({name.lower()})")
    ax.plot(grid.tau_grid, grid.tau_post / step, color="#0b0b0b", lw=2, label="Grid")
    ax.axvline(0.08, color="#898781", ls="--", lw=1)
    ax.annotate("true 0.08", (0.08, ax.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points", fontsize=8,
                color="#898781")
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("tau (log scale)", fontsize=9)
    ax.set_ylabel("Posterior density", fontsize=9)
    ax.set_title(f"Posterior of tau, typical {args.history:g} past cases per pair", loc="left", fontsize=10.5,
                 fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"pymc_tau{suffix}.png", dpi=160, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
