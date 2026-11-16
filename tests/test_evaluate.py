import numpy as np
import pytest
from scipy import stats

from theatre_sched import list_total_minutes, session_metrics


def test_turnaround_counted_between_cases_only():
    samples = np.full((3, 5), 60.0)  # three cases, always exactly 60 minutes
    total = list_total_minutes(samples, [0, 1, 2], turnaround=15)
    assert np.allclose(total, 3 * 60 + 2 * 15)
    assert np.allclose(list_total_minutes(samples, [0], turnaround=15), 60)
    assert np.allclose(list_total_minutes(samples, [], turnaround=15), 0)


def test_fixed_durations_give_certain_outcomes():
    samples = np.full((2, 100), 100.0)
    under = session_metrics(list_total_minutes(samples, [0, 1], 10), session_minutes=240)
    over = session_metrics(list_total_minutes(samples, [0, 1], 10), session_minutes=200)
    assert under["p_overrun"] == 0.0 and under["expected_overtime"] == 0.0
    assert over["p_overrun"] == 1.0 and over["expected_overtime"] == pytest.approx(10.0)
    assert under["utilisation"] == pytest.approx(210 / 240)


def test_monte_carlo_matches_lognormal_tail():
    # One case: P(duration > L) has a closed form, so the Monte Carlo estimate can be checked.
    mu, sigma, L, n = np.log(90), 0.35, 120.0, 200_000
    rng = np.random.default_rng(0)
    samples = np.exp(mu + sigma * rng.standard_normal((1, n)))
    m = session_metrics(list_total_minutes(samples, [0], 0), L)
    exact = stats.lognorm.sf(L, s=sigma, scale=np.exp(mu))
    assert abs(m["p_overrun"] - exact) < 4 * m["p_overrun_se"]
