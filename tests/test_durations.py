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
