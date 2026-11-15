import numpy as np

from delta_hedging import realised_vol, simulate_gbm

from .conftest import S0, SIGMA, T


def test_shape_and_start(rng):
    S = simulate_gbm(S0, 0.05, SIGMA, T, 63, 100, rng)
    assert S.shape == (100, 64)
    assert np.all(S[:, 0] == S0)
    assert np.all(S > 0)


def test_terminal_moments_match_theory(rng):
    mu, n = 0.08, 200_000
    S_T = simulate_gbm(S0, mu, SIGMA, T, 4, n, rng)[:, -1]
    # E[S_T] = S0 e^{mu T}: this is the check that catches a missing -sigma^2/2.
    se = S_T.std(ddof=1) / np.sqrt(n)
    assert abs(S_T.mean() - S0 * np.exp(mu * T)) < 4 * se
    log_ret = np.log(S_T / S0)
    assert abs(log_ret.mean() - (mu - 0.5 * SIGMA**2) * T) < 4 * SIGMA * np.sqrt(T / n)
    assert abs(log_ret.std(ddof=1) - SIGMA * np.sqrt(T)) < 0.001


def test_realised_vol_recovers_sigma(rng):
    S = simulate_gbm(S0, 0.0, 0.25, T, 2000, 200, rng)
    assert abs(realised_vol(S, T).mean() - 0.25) < 0.002
