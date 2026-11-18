import numpy as np
import pandas as pd
import pytest

from theatre_sched import fit_plugin


def _history(pairs, rng):
    frames = []
    for surgeon, procedure, mu, sigma, n in pairs:
        frames.append(pd.DataFrame({"surgeon": surgeon, "procedure": procedure,
                                    "minutes": np.exp(rng.normal(mu, sigma, n))}))
    return pd.concat(frames, ignore_index=True)


def test_plugin_recovers_parameters_with_plenty_of_history():
    rng = np.random.default_rng(1)
    hist = _history([("A", "x", np.log(60), 0.3, 5000), ("B", "x", np.log(80), 0.3, 5000)], rng)
    t = fit_plugin(hist).table.set_index("surgeon")
    assert t.loc["A", "mu"] == pytest.approx(np.log(60), abs=0.02)
    assert t.loc["B", "sigma"] == pytest.approx(0.3, abs=0.02)


def test_sparse_pairs_borrow_the_procedure_spread():
    rng = np.random.default_rng(2)
    hist = _history([("A", "x", np.log(60), 0.3, 2000), ("B", "x", np.log(80), 0.3, 2)], rng)
    t = fit_plugin(hist, min_cases_for_own_sd=3).table.set_index("surgeon")
    assert t.loc["B", "sigma_source"] == "pooled"
    assert t.loc["B", "sigma"] == pytest.approx(0.3, abs=0.03)
    assert t.loc["A", "sigma_source"] == "own"


def test_samples_have_right_shape_and_are_positive():
    rng = np.random.default_rng(3)
    hist = _history([("A", "x", np.log(60), 0.3, 50)], rng)
    fit = fit_plugin(hist)
    patients = pd.DataFrame({"surgeon": ["A", "A", "A"], "procedure": ["x", "x", "x"]})
    s = fit.model.sample(patients, 1000, rng)
    assert s.shape == (3, 1000) and (s > 0).all()


# ---------------------------------------------------------------- hierarchical model
from theatre_sched import WorldConfig, fit_hierarchical, make_world  # noqa: E402
from theatre_sched.rolling import RollingConfig, make_patients  # noqa: E402


def test_hierarchical_matches_plugin_with_plenty_of_history():
    rng = np.random.default_rng(4)
    # B is slower than A at x but faster at y, so a surgeon effect can't explain the pairs: tau is large
    hist = _history([("A", "x", np.log(60), 0.3, 3000), ("B", "x", np.log(80), 0.3, 3000),
                     ("A", "y", np.log(40), 0.3, 3000), ("B", "y", np.log(30), 0.3, 3000)], rng)
    t = fit_hierarchical(hist).table.set_index(["surgeon", "procedure"])
    assert t.loc[("A", "x"), "mu_hat"] == pytest.approx(np.log(60), abs=0.02)
    assert t.loc[("B", "y"), "mu_hat"] == pytest.approx(np.log(30), abs=0.02)
    assert (t["weight"] > 0.95).all()  # the pairs' own data dominate
    assert t.loc[("B", "y"), "sigma"] == pytest.approx(0.3, abs=0.02)


def test_sparse_pairs_are_pulled_towards_the_prior_and_stay_uncertain():
    rng = np.random.default_rng(5)
    pairs = [(s, p, np.log(m), 0.3, 200) for s, p, m in
             [("A", "x", 60), ("B", "x", 66), ("C", "x", 54), ("A", "y", 30), ("B", "y", 33), ("C", "y", 27)]]
    hist = _history(pairs, rng)
    # D has one very slow case of x: on its own it would look like a 120-minute surgeon
    hist = pd.concat([hist, pd.DataFrame({"surgeon": ["D", "D"], "procedure": ["x", "y"], "minutes": [120.0, 31.0]})])
    t = fit_hierarchical(hist).table.set_index(["surgeon", "procedure"])
    d = t.loc[("D", "x")]
    assert d["weight"] < 0.6 and t.loc[("A", "x"), "weight"] > d["weight"]
    assert d["prior_mean"] < d["mu_hat"] < d["ybar"]  # between its own average and the prior
    assert d["post_var"] > 5 * t.loc[("A", "x"), "post_var"]


def test_draws_share_each_pairs_uncertainty():
    rng = np.random.default_rng(6)
    hist = _history([("A", "x", np.log(60), 0.3, 2), ("B", "x", np.log(60), 0.3, 200),
                     ("A", "y", np.log(30), 0.3, 200), ("B", "y", np.log(30), 0.3, 200)], rng)
    model = fit_hierarchical(hist)
    patients = pd.DataFrame({"surgeon": ["A", "A", "B"], "procedure": ["x", "x", "y"]})
    s = np.log(model.sample(patients, 20000, np.random.default_rng(7)))
    same = np.corrcoef(s[0], s[1])[0, 1]
    other = np.corrcoef(s[0], s[2])[0, 1]
    assert same > 0.05 and abs(other) < 0.03
    assert s.shape == (3, 20000)


def test_pairs_with_no_history_use_the_prior():
    rng = np.random.default_rng(8)
    hist = _history([("A", "x", np.log(60), 0.3, 100), ("B", "x", np.log(70), 0.3, 100),
                     ("A", "y", np.log(30), 0.3, 100)], rng)
    model = fit_hierarchical(hist)
    new = pd.DataFrame({"surgeon": ["B"], "procedure": ["y"]})  # B has never done y
    assert np.isfinite(model.mean_minutes(new)).all()
    assert model._posterior(new)["post_var"].iloc[0] >= model.tau ** 2
    with pytest.raises(KeyError):
        model.sample(pd.DataFrame({"surgeon": ["A"], "procedure": ["z"]}), 10, rng)


def test_tau_is_recovered_in_a_large_world():
    taus = []
    for seed in range(5):
        w = make_world(WorldConfig(n_surgeons=12, history_median_cases=40), seed=seed)
        with_s, without = fit_hierarchical(w.history).tau, fit_hierarchical(w.history, surgeon_effect=False).tau
        assert without > with_s  # leaving out surgeon speed leaves more unexplained spread
        taus.append((with_s, without))
    with_s, without = np.mean(taus, axis=0)
    assert 0.06 < with_s < 0.10  # true surgeon-procedure deviation 0.08
    assert 0.12 < without < 0.22  # plus surgeon speed (0.15): sqrt(0.15^2 + 0.08^2) = 0.17


def test_rolling_planner_choice_keeps_the_same_patients():
    w = make_world(WorldConfig(n_surgeons=4), seed=2)
    a = make_patients(w, RollingConfig(duration_model="plugin", weeks_measured=10, plan_samples=50), seed=2)
    b = make_patients(w, RollingConfig(duration_model="hierarchical", weeks_measured=10, plan_samples=50), seed=2)
    assert np.array_equal(a.true_minutes, b.true_minutes) and np.array_equal(a.added_day, b.added_day)
    assert not np.allclose(a.plan, b.plan)


# ---------------------------------------------------------------- the grid Bayesian model
from theatre_sched import fit_bayes  # noqa: E402
from theatre_sched.durations import tau_log_posterior  # noqa: E402


def test_tau_posterior_on_the_grid_finds_the_true_spread():
    # 400 pair averages with a known spread of 0.1 around a common mean, each measured with variance 0.01
    rng = np.random.default_rng(10)
    v = np.full(400, 0.01)
    ybar = 4.0 + rng.normal(0, 0.1, 400) + rng.normal(0, np.sqrt(v))
    grid = np.linspace(0.005, 0.5, 100)
    lp = tau_log_posterior(grid, np.ones((400, 1)), ybar, v)
    assert grid[np.argmax(lp)] == pytest.approx(0.1, abs=0.02)


def test_bayes_agrees_with_the_hierarchical_model_when_data_are_plentiful():
    w = make_world(WorldConfig(history_median_cases=200), seed=3)
    b = fit_bayes(w.history, n_draws=400, seed=3).interval(0.9).set_index(["surgeon", "procedure"])
    h = fit_hierarchical(w.history).table.set_index(["surgeon", "procedure"])
    assert np.allclose(b["median"], h.loc[b.index, "mu_hat"], atol=0.02)


def test_bayes_intervals_are_wider_than_empirical_bayes_when_data_are_sparse():
    widths = []
    for seed in range(4):
        w = make_world(WorldConfig(history_median_cases=3), seed=seed)
        b = fit_bayes(w.history, n_draws=400, seed=seed).interval(0.9)
        h = fit_hierarchical(w.history).table
        widths.append(((b["high"] - b["low"]).mean(), (2 * 1.6449 * np.sqrt(h["post_var"])).mean()))
    bayes, eb = np.mean(widths, axis=0)
    assert bayes > eb  # tau and sigma are uncertain too


def test_bayes_draws_share_each_pair_and_handle_unseen_pairs():
    rng = np.random.default_rng(11)
    hist = _history([("A", "x", np.log(60), 0.3, 2), ("B", "x", np.log(60), 0.3, 200),
                     ("A", "y", np.log(30), 0.3, 200), ("B", "y", np.log(30), 0.3, 200),
                     ("C", "y", np.log(35), 0.3, 50)], rng)
    model = fit_bayes(hist, n_draws=2000, seed=1)
    patients = pd.DataFrame({"surgeon": ["A", "A", "B", "C"], "procedure": ["x", "x", "y", "x"]})
    s = np.log(model.sample(patients, 2000, np.random.default_rng(12)))
    assert s.shape == (4, 2000) and np.isfinite(s).all()
    assert np.corrcoef(s[0], s[1])[0, 1] > 0.05  # same pair: shared uncertainty
    assert np.isfinite(model.mean_minutes(patients)).all()  # C has never done x
    # sigma draws for procedure x: S / chi^2(nu) has mean S / (nu - 2), with nu = (2 - 1) + (200 - 1)
    y = np.log(hist.loc[hist.procedure == "x", "minutes"])
    S = float(sum(((g - g.mean()) ** 2).sum() for _, g in y.groupby(hist.loc[hist.procedure == "x", "surgeon"])))
    assert model.sigma2[:, model.procs.index("x")].mean() == pytest.approx(S / 198, rel=0.03)
    with pytest.raises(KeyError):
        model.sample(pd.DataFrame({"surgeon": ["A"], "procedure": ["z"]}), 10, rng)


def test_rolling_simulation_runs_with_the_bayesian_planner():
    w = make_world(WorldConfig(n_surgeons=4), seed=2)
    a = make_patients(w, RollingConfig(duration_model="plugin", weeks_measured=10, plan_samples=50), seed=2)
    b = make_patients(w, RollingConfig(duration_model="bayes", weeks_measured=10, plan_samples=50), seed=2)
    assert np.array_equal(a.true_minutes, b.true_minutes)
    assert b.plan.shape == a.plan.shape and np.isfinite(b.plan).all()
