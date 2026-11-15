"""Sanity checks from PDF section 3.5: run these before trusting any chart."""

import numpy as np
import pytest

from delta_hedging import bs_delta, bs_greeks, bs_price

from .conftest import K, R, S0, SIGMA, T


def test_default_values_match_the_pdf():
    assert bs_price(S0, K, T, R, SIGMA, "call") == pytest.approx(4.49, abs=0.01)
    assert bs_price(S0, K, T, R, SIGMA, "put") == pytest.approx(3.49, abs=0.01)
    g = bs_greeks(S0, K, T, R, SIGMA)
    assert g["delta"] == pytest.approx(0.560, abs=0.001)
    assert g["gamma"] == pytest.approx(0.0394, abs=0.0001)
    assert g["vega"] == pytest.approx(19.72, abs=0.01)
    assert g["theta"] == pytest.approx(-9.95, abs=0.01)


@pytest.mark.parametrize("q", [0.0, 0.015])
def test_put_call_parity(q):
    S = np.array([80.0, 100.0, 125.0])
    c = bs_price(S, K, T, R, SIGMA, "call", q)
    p = bs_price(S, K, T, R, SIGMA, "put", q)
    assert np.allclose(c - p, S * np.exp(-q * T) - K * np.exp(-R * T), atol=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.015])
def test_greeks_match_finite_differences(kind, q):
    S, h = 103.0, 1e-4 * 103.0
    g = bs_greeks(S, K, T, R, SIGMA, kind, q)
    up, mid, dn = (bs_price(s, K, T, R, SIGMA, kind, q) for s in (S + h, S, S - h))
    assert (up - dn) / (2 * h) == pytest.approx(g["delta"], abs=1e-6)
    assert (up - 2 * mid + dn) / h**2 == pytest.approx(g["gamma"], abs=1e-4)
    dv = 1e-5
    fd_vega = (bs_price(S, K, T, R, SIGMA + dv, kind, q) - bs_price(S, K, T, R, SIGMA - dv, kind, q)) / (2 * dv)
    assert fd_vega == pytest.approx(g["vega"], abs=1e-5)
    # theta is dV/dt = -dV/dtau
    fd_theta = -(bs_price(S, K, T + dv, R, SIGMA, kind, q) - bs_price(S, K, T - dv, R, SIGMA, kind, q)) / (2 * dv)
    assert fd_theta == pytest.approx(g["theta"], abs=1e-5)
    assert bs_delta(S, K, T, R, SIGMA, kind, q) == pytest.approx(g["delta"])


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.015])
def test_theta_gamma_pde_identity(kind, q):
    """Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0."""
    S = 97.0
    g = bs_greeks(S, K, T, R, SIGMA, kind, q)
    v = bs_price(S, K, T, R, SIGMA, kind, q)
    pde = g["theta"] + 0.5 * SIGMA**2 * S**2 * g["gamma"] + (R - q) * S * g["delta"] - R * v
    assert abs(pde) < 1e-8


def test_vega_gamma_identity():
    """vega = sigma * tau * S^2 * gamma (PDF section 2.5)."""
    g = bs_greeks(S0, K, T, R, SIGMA)
    assert g["vega"] == pytest.approx(SIGMA * T * S0**2 * g["gamma"])


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        bs_price(S0, K, T, R, SIGMA, "straddle")
