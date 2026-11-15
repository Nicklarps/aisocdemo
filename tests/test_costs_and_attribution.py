import numpy as np
import pytest

from delta_hedging import (
    attribute_pnl,
    hedge_short_option,
    leland_number,
    leland_vol,
    simulate_gbm,
    summarize,
    whalley_wilmott_band,
)

from .conftest import K, R, S0, SIGMA, T


def test_leland_example_from_the_pdf():
    """One-way 10 bps -> k = 0.002 round trip, daily hedging over 3 months."""
    dt = T / 63
    assert leland_number(SIGMA, 0.002, dt) == pytest.approx(0.127, abs=0.001)
    assert leland_vol(SIGMA, 0.002, dt) == pytest.approx(0.212, abs=0.001)
    assert leland_vol(SIGMA, 0.002, dt, short=False) < SIGMA
    assert leland_vol(SIGMA, 1.0, dt, short=False) == 0.0  # ill-posed region, clipped


def test_leland_vol_pays_for_the_rebalancing_costs(rng):
    cost, n = 0.001, 63
    S = simulate_gbm(S0, R, SIGMA, T, n, 20_000, rng)
    plain = hedge_short_option(S, K, T, R, SIGMA, cost=cost, return_history=True)
    sig_hat = leland_vol(SIGMA, 2 * cost, T / n)
    leland = hedge_short_option(S, K, T, R, sig_hat, cost=cost, return_history=True)
    # Leland's volatility covers the cost of REBALANCING. Setting up and
    # unwinding the hedge are separate one-off costs, so add those back.
    growth = np.exp(R * T)
    one_off = leland.history["fee"][:, 0] * growth + leland.history["fee"][:, -1]
    assert abs((leland.pnl + one_off).mean()) < 0.03
    assert plain.pnl.mean() < -0.3


def test_whalley_wilmott_band_shape():
    band = whalley_wilmott_band(cost=0.001, risk_aversion=1.0, r=R)
    assert band(100.0, 0.1, 0.08) > band(100.0, 0.1, 0.02)  # wider where gamma is big
    tight = whalley_wilmott_band(cost=0.001, risk_aversion=8.0, r=R)
    assert tight(100.0, 0.1, 0.04) == pytest.approx(band(100.0, 0.1, 0.04) / 2.0)
    assert whalley_wilmott_band(0.0, 1.0)(100.0, 0.1, 0.04) == 0.0


def test_gamma_scaled_band_runs_in_the_loop(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 252, 2000, rng)
    base = hedge_short_option(S, K, T, R, SIGMA, cost=0.001)
    band = whalley_wilmott_band(cost=0.001, risk_aversion=1.0, r=R)
    ww = hedge_short_option(S, K, T, R, SIGMA, cost=0.001, band=band, band_mode="edge")
    assert ww.costs.mean() < base.costs.mean()
    assert np.all(np.isfinite(ww.pnl))


@pytest.mark.parametrize("cost", [0.0, 0.001])
def test_attribution_adds_up(rng, cost):
    S = simulate_gbm(S0, R, 0.25, T, 63, 1, rng)[0]
    df = attribute_pnl(S, K, T, R, SIGMA, cost=cost)
    parts = ["theta", "gamma", "financing", "dividend", "vega", "delta_mismatch", "cost", "residual"]
    assert np.allclose(df[parts].sum(axis=1), df["total"], atol=1e-12)
    assert df.attrs["initial"] + df["total"].sum() == pytest.approx(df.attrs["terminal_pnl"])
    assert (df["theta"] > 0).all() and (df["gamma"] <= 0).all()  # short option: earn theta, pay gamma
    assert df["vega"].abs().max() == 0.0  # constant marking vol
    assert df["delta_mismatch"].abs().max() < 1e-12


def test_attribution_residual_is_small_on_average(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 63, 200, rng)
    explained, unexplained = 0.0, 0.0
    for path in S:
        df = attribute_pnl(path, K, T, R, SIGMA)
        explained += df[["theta", "gamma"]].abs().to_numpy().sum()
        unexplained += df["residual"].abs().sum()
    assert unexplained / explained < 0.10


def test_attribution_vega_term_with_a_moving_mark(rng):
    S = simulate_gbm(S0, R, SIGMA, T, 63, 1, rng)
    marks = np.full((1, 64), SIGMA)
    marks[0, 10:] = 0.24  # implied vol jumps 4 points at step 10
    df = attribute_pnl(S[0], K, T, R, SIGMA, sigma_mark=marks)
    assert df["vega"].iloc[9] < -0.5  # short vega: a vol spike hurts
    assert df["vega"].drop(index=9).abs().max() == 0.0
    flat = attribute_pnl(S[0], K, T, R, SIGMA)
    # Re-marking moves P&L between days, never the total at expiry.
    assert df.attrs["terminal_pnl"] == pytest.approx(flat.attrs["terminal_pnl"])
