"""Monte Carlo checks from PDF sections 2.6, 2.7, 3.4 and 3.5."""

import numpy as np
import pytest

from delta_hedging import (
    bs_greeks,
    bs_price,
    derman_kamal_std,
    gamma_weighted_pnl,
    hedge_ledger,
    hedge_short_option,
    loglog_slope,
    simulate_gbm,
    summarize,
    unhedged_short_option,
)

from .conftest import K, R, S0, SIGMA, T

N_PATHS = 20_000
VEGA0 = bs_greeks(S0, K, T, R, SIGMA)["vega"]


def test_mean_zero_and_derman_kamal_scaling(rng):
    ns, stds = (21, 63, 252), []
    for n in ns:
        S = simulate_gbm(S0, R, SIGMA, T, n, N_PATHS, rng)
        s = summarize(hedge_short_option(S, K, T, R, SIGMA).pnl)
        assert abs(s["mean"]) < 3 * s["se"]  # discrete hedging adds noise, not bias
        assert s["std"] == pytest.approx(derman_kamal_std(VEGA0, SIGMA, n), rel=0.15)
        stds.append(s["std"])
    scaled = np.array(stds) * np.sqrt(ns)
    assert scaled.max() / scaled.min() < 1.10  # std * sqrt(N) roughly constant
    assert loglog_slope(ns, stds) == pytest.approx(-0.5, abs=0.05)


def test_hedging_beats_doing_nothing(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 63, N_PATHS, rng)
    hedged = hedge_short_option(S, K, T, R, SIGMA).pnl
    raw = unhedged_short_option(S, K, T, R, SIGMA)
    assert raw.std() > 10 * hedged.std()


def test_error_vanishes_with_frequent_hedging(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 4000, 2000, rng)
    assert hedge_short_option(S, K, T, R, SIGMA).pnl.std() < 0.07


def test_drift_barely_matters_when_hedged(rng):
    """mu drops out of the PDE, so the hedged P&L should not care about it."""
    out = {}
    for mu in (0.04, 0.15):
        S = simulate_gbm(S0, mu, SIGMA, T, 63, N_PATHS, np.random.default_rng(7))
        out[mu] = (summarize(hedge_short_option(S, K, T, R, SIGMA).pnl),
                   unhedged_short_option(S, K, T, R, SIGMA).mean())
    assert abs(out[0.15][0]["mean"] - out[0.04][0]["mean"]) < 0.02
    assert out[0.15][0]["std"] == pytest.approx(out[0.04][0]["std"], rel=0.03)
    assert abs(out[0.15][1] - out[0.04][1]) > 1.0  # the unhedged seller cares a lot


def test_realised_above_implied_costs_the_seller(rng):
    sig_r = 0.25
    S = simulate_gbm(S0, R, sig_r, T, 63, N_PATHS, rng)
    res = hedge_short_option(S, K, T, R, SIGMA)
    s = summarize(res.pnl)
    assert s["mean"] == pytest.approx(VEGA0 * (SIGMA - sig_r), abs=0.05)  # about -1.0
    exact = (bs_price(S0, K, T, R, SIGMA) - bs_price(S0, K, T, R, sig_r)) * np.exp(R * T)
    assert abs(s["mean"] - exact) < 4 * s["se"]
    # The gamma-weighted variance formula explains the P&L path by path.
    predicted = gamma_weighted_pnl(S, K, T, R, SIGMA)
    assert np.corrcoef(predicted, res.pnl)[0, 1] > 0.95


def test_hedging_at_realised_vol_removes_path_dependence(rng):
    """Ahmad & Wilmott: hedge at actual vol and the total is (nearly) locked in."""
    sig_r = 0.25
    S = simulate_gbm(S0, R, sig_r, T, 2000, 4000, rng)
    at_implied = hedge_short_option(S, K, T, R, SIGMA).pnl
    at_realised = hedge_short_option(S, K, T, R, SIGMA, sigma_hedge=sig_r).pnl
    assert at_realised.std() < 0.35 * at_implied.std()
    locked_in = (bs_price(S0, K, T, R, SIGMA) - bs_price(S0, K, T, R, sig_r)) * np.exp(R * T)
    assert at_realised.mean() == pytest.approx(locked_in, abs=0.02)


def test_put_hedge_also_replicates(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 63, N_PATHS, rng)
    s = summarize(hedge_short_option(S, K, T, R, SIGMA, kind="put").pnl)
    assert abs(s["mean"]) < 3 * s["se"]
    # Same gamma as the call, so the same hedging error path by path.
    call = hedge_short_option(S, K, T, R, SIGMA, kind="call").pnl
    assert np.allclose(hedge_short_option(S, K, T, R, SIGMA, kind="put").pnl, call, atol=1e-9)


def test_dividend_yield_is_handled_consistently(rng):
    q = 0.03
    S = simulate_gbm(S0, R - q, SIGMA, T, 252, N_PATHS, rng)
    s = summarize(hedge_short_option(S, K, T, R, SIGMA, q=q).pnl)
    assert abs(s["mean"]) < 3 * s["se"] + 0.002


def test_costs_reduce_pnl_by_what_was_paid(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 63, 5000, rng)
    free = hedge_short_option(S, K, T, R, SIGMA)
    paid = hedge_short_option(S, K, T, R, SIGMA, cost=0.001)
    assert np.all(paid.costs > 0)
    assert np.allclose(free.pnl - paid.pnl, paid.costs, atol=1e-9)


def test_band_zero_is_time_based_and_wide_band_never_rebalances(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 63, 2000, rng)
    base = hedge_short_option(S, K, T, R, SIGMA, cost=0.001)
    for mode in ("target", "edge"):
        zero = hedge_short_option(S, K, T, R, SIGMA, cost=0.001, band=0.0, band_mode=mode)
        assert np.allclose(zero.pnl, base.pnl, atol=1e-12)
    wide = hedge_short_option(S, K, T, R, SIGMA, cost=0.001, band=2.0)
    assert np.all(wide.n_trades == 1)  # only the initial hedge
    mid = hedge_short_option(S, K, T, R, SIGMA, cost=0.001, band=0.05)
    assert mid.n_trades.mean() < base.n_trades.mean()
    assert mid.costs.mean() < base.costs.mean()


def test_per_path_parameters_broadcast(rng):
    """Per-path K, r and vols give the same answer as running each path alone."""
    S = simulate_gbm(S0, R, SIGMA, T, 21, 3, rng) * np.array([[1.0], [2.0], [0.5]])
    Ks, rs, vols = S[:, 0] * 1.02, np.array([0.01, 0.04, 0.06]), np.array([0.15, 0.2, 0.3])
    together = hedge_short_option(S, Ks, T, rs, vols, cost=0.0005, q=0.01).pnl
    for j in range(3):
        alone = hedge_short_option(S[j], Ks[j], T, rs[j], vols[j], cost=0.0005, q=0.01).pnl
        assert alone[0] == pytest.approx(together[j], abs=1e-10)


def test_ledger_is_self_consistent(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 63, 1, rng)
    led = hedge_ledger(S[0], K, T, R, SIGMA)
    res = hedge_short_option(S, K, T, R, SIGMA)
    assert len(led) == 64
    assert led["hedging_error"].iloc[0] == pytest.approx(0.0, abs=1e-12)  # premium = fair value
    assert led["hedging_error"].iloc[-1] == pytest.approx(res.pnl[0])
    assert led["trade"].iloc[1:-1].to_numpy() == pytest.approx(np.diff(led["delta"])[:-1])
