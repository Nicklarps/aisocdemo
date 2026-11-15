"""The experiments behind every table and chart, as plain functions.

The notebook and ``python -m delta_hedging`` both call these, so the numbers
you read in one place are the numbers in the other. Each function takes a
seed and returns pandas objects; nothing here draws a chart.

Unless stated otherwise the set-up is the PDF's default: the desk is short
one 3-month at-the-money call, S0 = K = 100, r = 4%, implied vol 20%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import derman_kamal_std, loglog_slope, summarize
from .attribution import gamma_weighted_pnl
from .black_scholes import bs_greeks, bs_price
from .costs import leland_number, leland_vol, whalley_wilmott_band
from .hedging import hedge_short_option, unhedged_short_option
from .paths import simulate_gbm

__all__ = [
    "frequency_study",
    "step_error_study",
    "vol_bet_study",
    "drift_study",
    "cost_study",
    "band_frontier",
    "hedge_vol_study",
]

S0, K, T, R, SIGMA = 100.0, 100.0, 0.25, 0.04, 0.20


def frequency_study(n_values=(21, 63, 252), n_paths=20_000, seed=42,
                    sigma_implied=SIGMA, sigma_realised=SIGMA, mu=R):
    """Hedging error against the number of rebalances (PDF sections 2.7 and 3.4).

    Returns (table, pnl) where ``pnl[N]`` holds the hedged P&L per path and
    ``pnl["unhedged"]`` the P&L of the naked short call on the N = 63 paths.
    """
    rng = np.random.default_rng(seed)
    premium = bs_price(S0, K, T, R, sigma_implied)
    vega0 = bs_greeks(S0, K, T, R, sigma_implied)["vega"]
    rows, pnl = [], {}
    for n in n_values:
        S = simulate_gbm(S0, mu, sigma_realised, T, n, n_paths, rng)
        hedged = hedge_short_option(S, K, T, R, sigma_implied).pnl
        raw = unhedged_short_option(S, K, T, R, sigma_implied)
        s = summarize(hedged)
        dk = derman_kamal_std(vega0, sigma_implied, n)
        rows.append({
            "N": n, "mean": s["mean"], "se": s["se"], "std": s["std"],
            "derman_kamal_std": dk, "std_x_sqrtN": s["std"] * np.sqrt(n),
            "std_pct_premium": 100 * s["std"] / premium, "skew": s["skew"],
            "unhedged_std": raw.std(ddof=1),
        })
        pnl[n] = hedged
        if n == 63 or "unhedged" not in pnl:
            pnl["unhedged"] = raw
    table = pd.DataFrame(rows).set_index("N")
    table.attrs["loglog_slope"] = loglog_slope(table.index, table["std"])
    table.attrs["premium"] = premium
    return table, pnl


def step_error_study(n_steps=63, n_paths=50_000, seed=1):
    """One step's hedging error versus the total (PDF section 2.7, Boyle & Emanuel).

    Over a single interval the short hedger's P&L is a scaled, shifted
    chi-squared variable: many small gains, a few large losses. Summed over
    many steps it becomes close to normal.
    """
    rng = np.random.default_rng(seed)
    S = simulate_gbm(S0, R, SIGMA, T, n_steps, n_paths, rng)
    res = hedge_short_option(S, K, T, R, SIGMA, return_history=True)
    first = res.history["value"][:, 1] - res.history["value"][:, 0]
    g = bs_greeks(S0, K, T, R, SIGMA)
    dt = T / n_steps
    theory_std = np.sqrt(0.5) * g["gamma"] * S0**2 * SIGMA**2 * dt
    rows = []
    for name, x in (("first step", first), ("whole life", res.pnl)):
        s = summarize(x)
        rows.append({"error": name, "mean": s["mean"], "std": s["std"], "skew": s["skew"],
                     "excess_kurtosis": s["excess_kurtosis"], "share_gains": float((x > 0).mean())})
    table = pd.DataFrame(rows).set_index("error")
    table.attrs["first_step_theory_std"] = theory_std
    return table, {"first step": first, "whole life": res.pnl}


def vol_bet_study(sigma_realised_values=(0.10, 0.15, 0.20, 0.25, 0.30), n_steps=63,
                  n_paths=20_000, seed=7, sigma_implied=SIGMA):
    """Sell at implied vol, let the world move at a different vol (PDF section 2.6).

    Returns (table, detail). ``detail`` holds the paths, hedged P&L and the
    gamma-weighted prediction for the 25% case, for the path-dependence charts.
    """
    premium = bs_price(S0, K, T, R, sigma_implied)
    vega0 = bs_greeks(S0, K, T, R, sigma_implied)["vega"]
    rows, detail = [], {}
    for sig_r in sigma_realised_values:
        rng = np.random.default_rng(seed)  # same random numbers for every vol
        S = simulate_gbm(S0, R, sig_r, T, n_steps, n_paths, rng)
        pnl = hedge_short_option(S, K, T, R, sigma_implied).pnl
        s = summarize(pnl)
        rows.append({
            "sigma_realised": sig_r, "mean": s["mean"], "se": s["se"], "std": s["std"],
            "vega_rule": vega0 * (sigma_implied - sig_r),
            "exact_value_gap": (premium - bs_price(S0, K, T, R, sig_r)) * np.exp(R * T),
            "share_losing": float((pnl < 0).mean()),
        })
        if np.isclose(sig_r, 0.25):
            detail = {"S": S, "pnl": pnl,
                      "predicted": gamma_weighted_pnl(S, K, T, R, sigma_implied)}
    return pd.DataFrame(rows).set_index("sigma_realised"), detail


def drift_study(mu_values=(-0.10, 0.04, 0.15, 0.30), n_steps=63, n_paths=20_000, seed=7):
    """Change the stock's expected return and see who cares (PDF section 3.5)."""
    rows = []
    for mu in mu_values:
        rng = np.random.default_rng(seed)
        S = simulate_gbm(S0, mu, SIGMA, T, n_steps, n_paths, rng)
        h = summarize(hedge_short_option(S, K, T, R, SIGMA).pnl)
        u = summarize(unhedged_short_option(S, K, T, R, SIGMA))
        rows.append({"mu": mu, "hedged_mean": h["mean"], "hedged_std": h["std"],
                     "unhedged_mean": u["mean"], "unhedged_std": u["std"]})
    return pd.DataFrame(rows).set_index("mu")


def cost_study(n_values=(5, 10, 21, 42, 63, 126, 252, 504), cost=0.001,
               n_paths=20_000, seed=11):
    """Proportional costs against rebalancing frequency, and Leland's fix (PDF 2.8).

    Columns:
        mean, std            hedged P&L when priced and hedged at sigma
        mean_cost            average cost paid, of which ``one_off_cost`` is
                             setting up and unwinding the hedge
        rmse                 sqrt(mean^2 + std^2): one number for "how far
                             from zero does the seller typically end up"
        leland_vol           sigma_hat for this N
        leland_uplift        extra premium from pricing at sigma_hat
        leland_mean          mean P&L when priced AND hedged at sigma_hat
        leland_mean_ex_one_off   the same, adding back the one-off costs
    """
    rng = np.random.default_rng(seed)
    premium = bs_price(S0, K, T, R, SIGMA)
    growth = np.exp(R * T)
    rows = []
    for n in n_values:
        S = simulate_gbm(S0, R, SIGMA, T, n, n_paths, rng)
        plain = hedge_short_option(S, K, T, R, SIGMA, cost=cost, return_history=True)
        s = summarize(plain.pnl)
        one_off = plain.history["fee"][:, 0] * growth + plain.history["fee"][:, -1]
        dt = T / n
        sig_hat = leland_vol(SIGMA, 2 * cost, dt)
        lel = hedge_short_option(S, K, T, R, sig_hat, cost=cost, return_history=True)
        lel_one_off = lel.history["fee"][:, 0] * growth + lel.history["fee"][:, -1]
        rows.append({
            "N": n, "mean": s["mean"], "se": s["se"], "std": s["std"],
            "rmse": float(np.sqrt(np.mean(plain.pnl**2))),
            "mean_cost": plain.costs.mean(), "one_off_cost": one_off.mean(),
            "leland_number": leland_number(SIGMA, 2 * cost, dt), "leland_vol": sig_hat,
            "leland_uplift": (bs_price(S0, K, T, R, sig_hat) - premium) * growth,
            "leland_mean": lel.pnl.mean(),
            "leland_mean_ex_one_off": (lel.pnl + lel_one_off).mean(),
        })
    return pd.DataFrame(rows).set_index("N")


def band_frontier(cost=0.001, n_fine=504, n_paths=10_000, seed=21,
                  time_steps=(504, 252, 126, 63, 42, 21, 12, 7),
                  fixed_bands=(0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18),
                  risk_aversions=(30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03)):
    """Cost against hedging error for three rebalancing rules (PDF 2.8, recommendation 2).

    All rules see the same paths. The price is checked ``n_fine`` times over
    the option's life (8 times a day by default):

        time-based     rehedge every m-th check, whatever the market did
        fixed band     rehedge only when delta has drifted more than h
        gamma band     Whalley-Wilmott band, wider where gamma is larger

    Both band rules trade back to the edge of the band. One row per setting;
    a rule is better if it sits lower (less error) at the same cost.
    """
    rng = np.random.default_rng(seed)
    S = simulate_gbm(S0, R, SIGMA, T, n_fine, n_paths, rng)
    rows = []

    def record(rule, setting, res):
        rows.append({"rule": rule, "setting": setting, "mean_cost": res.costs.mean(),
                     "std": res.pnl.std(ddof=1), "mean": res.pnl.mean(),
                     "trades": res.n_trades.mean()})

    for n in time_steps:
        if n_fine % n:
            raise ValueError(f"{n} does not divide n_fine={n_fine}")
        record("time-based", f"N = {n}",
               hedge_short_option(S[:, :: n_fine // n], K, T, R, SIGMA, cost=cost))
    for h in fixed_bands:
        record("fixed band", f"h = {h}",
               hedge_short_option(S, K, T, R, SIGMA, cost=cost, band=h, band_mode="edge"))
        record("fixed band, trade to target", f"h = {h}",
               hedge_short_option(S, K, T, R, SIGMA, cost=cost, band=h, band_mode="target"))
    for lam in risk_aversions:
        band = whalley_wilmott_band(cost, lam, R)
        record("gamma band", f"risk aversion = {lam}",
               hedge_short_option(S, K, T, R, SIGMA, cost=cost, band=band, band_mode="edge"))
    return pd.DataFrame(rows)


def hedge_vol_study(sigma_implied=SIGMA, sigma_realised=0.30, n_values=(63, 2016),
                    n_paths=5_000, seed=5, n_show=6):
    """Hedge at implied or at realised vol? (Ahmad & Wilmott, PDF section 2.6)

    The option is sold and marked at implied vol; the world moves at a
    higher realised vol. Returns (table, paths): the table has one row per
    (N, hedging vol); ``paths`` has the mark-to-market P&L history of the
    first ``n_show`` paths at the finest N, for both choices.
    """
    locked_in = (bs_price(S0, K, T, R, sigma_implied)
                 - bs_price(S0, K, T, R, sigma_realised)) * np.exp(R * T)
    rows, paths = [], {}
    for n in n_values:
        rng = np.random.default_rng(seed)
        S = simulate_gbm(S0, R, sigma_realised, T, n, n_paths, rng)
        for label, vol in (("implied", sigma_implied), ("realised", sigma_realised)):
            res = hedge_short_option(S, K, T, R, sigma_implied, sigma_hedge=vol,
                                     return_history=True)
            daily = np.diff(res.history["value"], axis=1)
            s = summarize(res.pnl)
            rows.append({
                "N": n, "hedge_vol": label, "mean": s["mean"], "std": s["std"],
                "p05": s["p05"], "p95": s["p95"],
                # how bumpy the ride is: std of step-to-step P&L, scaled to one day
                "daily_pnl_std": float(daily.std() * np.sqrt(n / 63)),
            })
            if n == max(n_values):
                paths[label] = {"t": res.history["t"], "value": res.history["value"][:n_show],
                                "pnl": res.pnl}
    table = pd.DataFrame(rows).set_index(["N", "hedge_vol"])
    table.attrs["locked_in"] = locked_in
    return table, paths
