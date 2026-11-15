"""The self-financing delta-hedging loop (PDF sections 1.3 and 3.3).

The desk SELLS one European option, receives the premium, and uses it to
fund a portfolio of stock and cash that tries to replicate the payoff it
owes. At expiry

    hedging error = cash + shares * S_T - payoff

How the code maps to the concepts:

    premium funds the hedge          cash = premium - delta * S0
    risk-free account                cash *= growth          (growth = e^{r dt})
    self-financing condition         cash -= trade * S       (stock bought from cash)
    proportional transaction cost    cost * |trade| * S
    replication error                cash + delta * S_T - payoff

Convention: with ``n_steps`` intervals there are N = n_steps hedge decisions,
at t_0, ..., t_{N-1} (the initial hedge plus N - 1 rebalances). The position
is unwound at T. So n_steps = 63 over T = 0.25 means "hedge daily for three
months".

Three volatilities are kept separate on purpose:

    the path's own volatility   how much the stock actually moves (realised)
    sigma_price                 the vol the option was sold at (implied)
    sigma_hedge                 the vol used to compute the hedge delta

Mixing them up is bug number 4 in the PDF's list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_delta, bs_greeks, bs_price

__all__ = [
    "HedgeResult",
    "payoff",
    "hedge_short_option",
    "unhedged_short_option",
    "hedge_ledger",
]


@dataclass
class HedgeResult:
    """Output of :func:`hedge_short_option`. Every array has one entry per path."""

    pnl: np.ndarray  # terminal hedging error: cash + stock - payoff
    premium: np.ndarray  # premium received at t0
    costs: np.ndarray  # transaction costs paid, grown at r to expiry
    n_trades: np.ndarray  # how many times the hedge was traded (unwind excluded)
    history: dict | None = None  # per-step arrays, only if return_history=True


def payoff(S_T, K, kind="call"):
    """Option payoff at expiry: max(S - K, 0) for a call, max(K - S, 0) for a put."""
    if kind == "call":
        return np.maximum(S_T - K, 0.0)
    if kind == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _at_step(x, i):
    """Pick column i if x is a per-step (n_paths, n_steps + 1) array, else return x."""
    return x[:, i] if np.ndim(x) == 2 else x


def hedge_short_option(
    S,
    K,
    T,
    r,
    sigma_price,
    sigma_hedge=None,
    kind="call",
    cost=0.0,
    unwind_cost=True,
    band=None,
    band_mode="target",
    q=0.0,
    sigma_mark=None,
    return_history=False,
):
    """Short one option, delta-hedge it along each path, return the terminal P&L.

    Parameters
    ----------
    S : array, shape (n_paths, n_steps + 1) or (n_steps + 1,)
        Price paths on an even time grid from 0 to T.
    K, r : float or array of shape (n_paths,)
        Strike and continuously compounded risk-free rate.
    T : float
        Time to expiry in years.
    sigma_price : float or array (n_paths,)
        Volatility the option is SOLD at (implied vol). Sets the premium.
    sigma_hedge : float or array (n_paths,), optional
        Volatility used to compute the hedge delta. Defaults to sigma_price.
    kind : "call" or "put"
    cost : float
        One-way proportional transaction cost, as a fraction of traded value
        (0.001 = 10 basis points). The round-trip cost in Leland's formula is
        k = 2 * cost.
    unwind_cost : bool
        Whether selling the hedge at expiry also pays the cost.
    band : None, float, or callable(S, tau, gamma) -> array
        No-trade band on delta. None rehedges at every step. A number h
        trades only when |target - held| > h.
    band_mode : "target" or "edge"
        When the band is breached, trade all the way to the target delta
        ("target", as in the PDF) or only back to the edge of the band
        ("edge", what the transaction-cost theory recommends).
    q : float
        Continuous dividend yield on the stock (0 for the synthetic demo).
    sigma_mark : float, array (n_paths,) or array (n_paths, n_steps + 1), optional
        Volatility used to mark the option to market in the history.
        Defaults to sigma_price. It never affects the terminal P&L.
    return_history : bool
        If True, also return per-step arrays of shape (n_paths, n_steps + 1).

    Returns
    -------
    HedgeResult
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    if n_steps < 1:
        raise ValueError("S needs at least two time points")
    if band_mode not in ("target", "edge"):
        raise ValueError("band_mode must be 'target' or 'edge'")
    dt = T / n_steps
    growth = np.exp(r * dt)  # one step of risk-free growth
    div_rate = np.expm1(q * dt)  # dividends per unit of stock value per step
    if sigma_hedge is None:
        sigma_hedge = sigma_price
    if sigma_mark is None:
        sigma_mark = sigma_price

    # --- t0: sell the option, buy the first hedge -------------------------
    S0 = S[:, 0]
    premium = np.broadcast_to(bs_price(S0, K, T, r, sigma_price, kind, q), (n_paths,)).copy()
    delta = np.broadcast_to(bs_delta(S0, K, T, r, sigma_hedge, kind, q), (n_paths,)).copy()
    fee = cost * np.abs(delta) * S0
    cash = premium - delta * S0 - fee
    costs = fee.copy()
    n_trades = (delta != 0).astype(int)

    if return_history:
        shape = (n_paths, n_steps + 1)
        hist = {name: np.zeros(shape) for name in
                ("delta", "cash", "option", "fee", "interest", "dividend", "trade")}
        hist["delta"][:, 0] = delta
        hist["cash"][:, 0] = cash
        hist["option"][:, 0] = bs_price(S0, K, T, r, _at_step(sigma_mark, 0), kind, q)
        hist["fee"][:, 0] = fee
        hist["trade"][:, 0] = delta

    # --- t1 .. t_{N-1}: rebalance ----------------------------------------
    for i in range(1, n_steps):
        S_i = S[:, i]
        interest = cash * (growth - 1.0)
        dividend = delta * 0.5 * (S[:, i - 1] + S_i) * div_rate
        cash = cash + interest + dividend  # earnings over (t_{i-1}, t_i]
        costs = costs * growth

        tau = T - i * dt  # strictly > 0 here
        target = bs_delta(S_i, K, tau, r, sigma_hedge, kind, q)  # uses S at t_i only
        gap = target - delta
        if band is None:
            trade = gap
        else:
            if callable(band):
                gamma = bs_greeks(S_i, K, tau, r, sigma_hedge, kind, q)["gamma"]
                h = band(S_i, tau, gamma)
            else:
                h = band
            if band_mode == "target":
                trade = np.where(np.abs(gap) > h, gap, 0.0)
            else:
                trade = np.sign(gap) * np.maximum(np.abs(gap) - h, 0.0)

        fee = cost * np.abs(trade) * S_i
        cash = cash - trade * S_i - fee  # self-financing: stock is paid for from cash
        costs = costs + fee
        n_trades = n_trades + (trade != 0)
        delta = delta + trade

        if return_history:
            hist["delta"][:, i] = delta
            hist["cash"][:, i] = cash
            hist["option"][:, i] = bs_price(S_i, K, tau, r, _at_step(sigma_mark, i), kind, q)
            hist["fee"][:, i] = fee
            hist["interest"][:, i] = interest
            hist["dividend"][:, i] = dividend
            hist["trade"][:, i] = trade

    # --- T: last interval, then unwind and settle -------------------------
    S_T = S[:, -1]
    interest = cash * (growth - 1.0)
    dividend = delta * 0.5 * (S[:, -2] + S_T) * div_rate
    cash = cash + interest + dividend
    costs = costs * growth
    fee = cost * np.abs(delta) * S_T if unwind_cost else np.zeros(n_paths)
    costs = costs + fee
    owed = payoff(S_T, K, kind)
    pnl = cash + delta * S_T - fee - owed  # hedging error (terminal value)

    history = None
    if return_history:
        hist["delta"][:, -1] = delta  # shares carried into expiry
        hist["cash"][:, -1] = cash  # before selling the hedge
        hist["option"][:, -1] = owed
        hist["fee"][:, -1] = fee
        hist["interest"][:, -1] = interest
        hist["dividend"][:, -1] = dividend
        hist["trade"][:, -1] = -delta
        hist["S"] = S
        hist["t"] = np.linspace(0.0, T, n_steps + 1)
        # Mark-to-market P&L: what the hedged book is worth at each step.
        value = hist["cash"] + hist["delta"] * S - hist["option"]
        value[:, -1] -= fee
        hist["value"] = value
        history = hist

    return HedgeResult(pnl=pnl, premium=premium, costs=costs, n_trades=n_trades, history=history)


def unhedged_short_option(S, K, T, r, sigma_price, kind="call", q=0.0):
    """P&L of selling the option and doing nothing: premium e^{rT} - payoff."""
    S = np.atleast_2d(np.asarray(S, dtype=float))
    premium = bs_price(S[:, 0], K, T, r, sigma_price, kind, q)
    return premium * np.exp(r * T) - payoff(S[:, -1], K, kind)


def hedge_ledger(S_path, K, T, r, sigma_price, **kwargs):
    """Step-by-step ledger for ONE path, as a pandas DataFrame.

    Columns: time, stock price, delta held, value of the shares, cash,
    option value, and the running hedging error (portfolio minus option).
    Extra keyword arguments go to :func:`hedge_short_option`.
    """
    import pandas as pd

    S_path = np.asarray(S_path, dtype=float).reshape(1, -1)
    res = hedge_short_option(S_path, K, T, r, sigma_price, return_history=True, **kwargs)
    h = res.history
    return pd.DataFrame(
        {
            "t": h["t"],
            "S": h["S"][0],
            "delta": h["delta"][0],
            "trade": h["trade"][0],
            "stock_value": h["delta"][0] * h["S"][0],
            "cash": h["cash"][0],
            "option_value": h["option"][0],
            "hedging_error": h["value"][0],
        }
    ).rename_axis("step")
