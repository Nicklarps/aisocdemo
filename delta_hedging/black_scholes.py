"""Black-Scholes price and Greeks for European calls and puts.

Maths (PDF sections 2.4-2.5), with tau = T - t the time left to expiry:

    d1 = [ln(S/K) + (r - q + sigma^2/2) tau] / (sigma sqrt(tau))
    d2 = d1 - sigma sqrt(tau)

    call  C = S e^{-q tau} N(d1) - K e^{-r tau} N(d2)
    put   P = K e^{-r tau} N(-d2) - S e^{-q tau} N(-d1)

N is the standard normal CDF and phi its density. ``q`` is a continuous
dividend yield (Merton's extension). It defaults to 0, which gives exactly
the formulas in the PDF; the real-data replay sets it to a small positive
number because an equity index pays dividends.

Every function broadcasts over numpy arrays, so the same code prices one
option or 20,000 simulated paths in a single call.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["bs_d1_d2", "bs_price", "bs_delta", "bs_greeks"]


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_d1_d2(S, K, tau, r, sigma, q=0.0):
    """Return (d1, d2). Requires tau > 0 and sigma > 0."""
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    return d1, d1 - sigma * sqrt_tau


def bs_price(S, K, tau, r, sigma, kind="call", q=0.0):
    """Black-Scholes value of a European call or put."""
    _check_kind(kind)
    d1, d2 = bs_d1_d2(S, K, tau, r, sigma, q)
    df_r = np.exp(-r * tau)  # discount factor on the strike
    df_q = np.exp(-q * tau)  # dividend discount on the stock
    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def bs_delta(S, K, tau, r, sigma, kind="call", q=0.0):
    """Delta = dV/dS: N(d1) for a call, N(d1) - 1 for a put (times e^{-q tau})."""
    _check_kind(kind)
    d1, _ = bs_d1_d2(S, K, tau, r, sigma, q)
    df_q = np.exp(-q * tau)
    if kind == "call":
        return df_q * norm.cdf(d1)
    return df_q * (norm.cdf(d1) - 1.0)


def bs_greeks(S, K, tau, r, sigma, kind="call", q=0.0):
    """Return a dict with delta, gamma, vega and theta.

    Units: vega is per 1.00 of volatility (divide by 100 for "per vol
    point"); theta is per year (divide by 252 for "per trading day").
    Gamma and vega are the same for a call and a put.
    """
    _check_kind(kind)
    d1, d2 = bs_d1_d2(S, K, tau, r, sigma, q)
    pdf = norm.pdf(d1)
    df_r = np.exp(-r * tau)
    df_q = np.exp(-q * tau)
    sqrt_tau = np.sqrt(tau)

    gamma = df_q * pdf / (S * sigma * sqrt_tau)
    vega = S * df_q * pdf * sqrt_tau
    decay = -S * df_q * pdf * sigma / (2.0 * sqrt_tau)  # the part of theta shared by both
    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = decay - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    else:
        delta = df_q * (norm.cdf(d1) - 1.0)
        theta = decay + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}
