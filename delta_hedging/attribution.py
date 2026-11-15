"""P&L attribution: where did each step's profit or loss come from?

For the short, hedged position the book value is

    Pi = cash + delta * S - V

Over one interval, a Taylor expansion of the option value V gives
(PDF section 2.6)

    dPi  ~  -Theta dt            theta income: the option you are short decays
            -1/2 Gamma dS^2      gamma loss: the hedge was a step behind the move
            + r * cash * dt      interest on the cash account
            - Vega * d(sigma)    change in the vol the option is marked at
            + (delta_held - delta_mark) dS    only if you hedge at a different vol
            - costs
            + residual           higher-order terms; small if steps are small

Black-Scholes sets the price so that theta income plus financing equals the
EXPECTED gamma loss at the implied volatility. What is left over is the bet
on realised versus implied variance.
"""

from __future__ import annotations

import numpy as np

from .black_scholes import bs_greeks
from .hedging import hedge_short_option

__all__ = ["attribute_pnl", "gamma_weighted_pnl"]


def attribute_pnl(
    S_path, K, T, r, sigma_price, sigma_hedge=None, sigma_mark=None,
    kind="call", cost=0.0, q=0.0, **kwargs,
):
    """Split each step's P&L on ONE path into theta, gamma, financing, vega and costs.

    Returns a DataFrame with one row per interval (t_i -> t_{i+1}). The
    columns add up: ``theta + gamma + financing + dividend + vega +
    delta_mismatch + cost + residual == total``, where ``total`` is the actual
    change in the book's mark-to-market value. ``df.attrs["initial"]`` holds
    the value at t0 (minus the first trade's cost), so
    ``initial + total.sum()`` is the terminal P&L.
    """
    import pandas as pd

    S = np.asarray(S_path, dtype=float).reshape(1, -1)
    if sigma_mark is None:
        sigma_mark = sigma_price
    res = hedge_short_option(
        S, K, T, r, sigma_price, sigma_hedge=sigma_hedge, sigma_mark=sigma_mark,
        kind=kind, cost=cost, q=q, return_history=True, **kwargs,
    )
    h = res.history
    n_steps = S.shape[1] - 1
    dt = T / n_steps

    # The marking vol at each step: a constant, or one value per step.
    sm = np.asarray(sigma_mark, dtype=float)
    sig = sm[0] if sm.ndim == 2 else np.full(n_steps + 1, float(sm.ravel()[0]))

    rows = []
    for i in range(n_steps):
        tau = T - i * dt
        sig_now = sig[i]
        # At expiry the option is worth its payoff, so there is no vol to re-mark.
        sig_next = sig[i + 1] if i + 1 < n_steps else sig_now
        g = bs_greeks(S[0, i], K, tau, r, sig_now, kind, q)
        dS = S[0, i + 1] - S[0, i]
        parts = {
            "theta": -g["theta"] * dt,
            "gamma": -0.5 * g["gamma"] * dS**2,
            "financing": h["interest"][0, i + 1],
            "dividend": h["dividend"][0, i + 1],
            "vega": -g["vega"] * (sig_next - sig_now),
            "delta_mismatch": (h["delta"][0, i] - g["delta"]) * dS,
            "cost": -h["fee"][0, i + 1],
        }
        total = h["value"][0, i + 1] - h["value"][0, i]
        parts["residual"] = total - sum(parts.values())
        parts["total"] = total
        rows.append({"t": h["t"][i + 1], "S": S[0, i + 1], **parts})

    df = pd.DataFrame(rows).rename_axis("step")
    df.attrs["initial"] = float(h["value"][0, 0])
    df.attrs["terminal_pnl"] = float(res.pnl[0])
    return df


def gamma_weighted_pnl(S, K, T, r, sigma_implied, kind="call", q=0.0):
    """Predicted terminal P&L of the SHORT option hedged at implied vol, path by path.

    Discrete version of the El Karoui, Jeanblanc-Picque & Shreve formula
    (PDF section 2.6), with the sign for the option seller:

        P&L_T ~ sum_i  e^{r (T - t_{i+1})} * 1/2 Gamma_i S_i^2
                       * (sigma_i^2 dt - (dS_i / S_i)^2)

    Read it as: dollar gamma x (implied variance - realised variance), summed
    along the path. It needs no cash account at all, yet it should track the
    simulated hedging P&L closely.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    n_steps = S.shape[1] - 1
    dt = T / n_steps
    total = np.zeros(S.shape[0])
    for i in range(n_steps):
        tau = T - i * dt
        gamma = bs_greeks(S[:, i], K, tau, r, sigma_implied, kind, q)["gamma"]
        ret2 = (S[:, i + 1] / S[:, i] - 1.0) ** 2
        step = 0.5 * gamma * S[:, i] ** 2 * (sigma_implied**2 * dt - ret2)
        total += step * np.exp(r * (T - (i + 1) * dt))
    return total
