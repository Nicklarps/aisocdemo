"""Run the whole demo:  python -m delta_hedging

Prints each result table and saves the charts to ``outputs/figures`` and the
tables to ``outputs/results``. Takes about a minute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import experiments as ex  # noqa: E402
from . import plots  # noqa: E402
from .attribution import attribute_pnl  # noqa: E402
from .black_scholes import bs_greeks, bs_price  # noqa: E402
from .hedging import hedge_ledger  # noqa: E402
from .paths import simulate_gbm  # noqa: E402

S0, K, T, R, SIGMA = ex.S0, ex.K, ex.T, ex.R, ex.SIGMA


def _show(title, table, path=None):
    print(f"\n=== {title} ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    if path is not None:
        table.to_csv(path, float_format="%.6f")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m delta_hedging", description=__doc__)
    parser.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    parser.add_argument("--no-real-data", action="store_true",
                        help="skip the FRED download and the historical replay")
    parser.add_argument("--refresh-data", action="store_true",
                        help="download the FRED series again even if cached")
    args = parser.parse_args(argv)

    figs, res = Path(args.out) / "figures", Path(args.out) / "results"
    figs.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    plots.use_style()
    pd.set_option("display.width", 200)

    # 1. Pricing --------------------------------------------------------------
    g = bs_greeks(S0, K, T, R, SIGMA)
    pricing = pd.Series({
        "call": bs_price(S0, K, T, R, SIGMA), "put": bs_price(S0, K, T, R, SIGMA, "put"),
        "delta": g["delta"], "gamma": g["gamma"], "vega_per_1.00": g["vega"],
        "theta_per_year": g["theta"], "theta_per_day": g["theta"] / 252,
    }, name="value").to_frame()
    _show("Black-Scholes, default case (S0=K=100, T=0.25, r=4%, sigma=20%)", pricing,
          res / "01_pricing.csv")
    plots.save(plots.fig_delta_tangent(S0, K, T, R, SIGMA), figs / "01_delta_tangent.png")

    # 2. Paths and one hedge, step by step -----------------------------------
    S = simulate_gbm(S0, R, SIGMA, T, 63, 2000, np.random.default_rng(3))
    plots.save(plots.fig_sample_paths(S, T, K=K), figs / "02_sample_paths.png")
    ledger = hedge_ledger(S[0], K, T, R, SIGMA)
    _show("One path: first and last rows of the hedging ledger",
          pd.concat([ledger.head(4), ledger.tail(3)]))
    ledger.to_csv(res / "02_single_path_ledger.csv", float_format="%.6f")
    plots.save(plots.fig_single_path(ledger, K), figs / "03_single_path.png")

    # 3. Monte Carlo ----------------------------------------------------------
    freq, pnl = ex.frequency_study()
    _show(f"Hedging error by rebalance count (log-log slope "
          f"{freq.attrs['loglog_slope']:.3f})", freq, res / "03_frequency.csv")
    plots.save(plots.fig_hedged_vs_unhedged(pnl), figs / "04_hedged_vs_unhedged.png")
    plots.save(plots.fig_error_by_frequency(freq, pnl), figs / "05_error_by_frequency.png")

    step, samples = ex.step_error_study()
    _show("One step versus the whole life", step, res / "04_step_vs_total.csv")
    plots.save(plots.fig_step_vs_total(samples, step), figs / "06_step_vs_total.png")

    # 4. Realised versus implied ---------------------------------------------
    vol, detail = ex.vol_bet_study()
    _show("Sold at 20% implied, stock moves at a different vol", vol, res / "05_vol_bet.csv")
    plots.save(plots.fig_vol_bet(vol), figs / "07_vol_bet.png")
    plots.save(plots.fig_path_dependence(detail, K), figs / "08_path_dependence.png")
    _show("Changing the stock's drift", ex.drift_study(), res / "06_drift.csv")

    # 5. Costs ----------------------------------------------------------------
    costs = ex.cost_study()
    _show("Proportional cost of 10 bp one-way, and Leland's adjustment", costs,
          res / "07_costs.csv")
    plots.save(plots.fig_cost_tradeoff(costs), figs / "09_cost_tradeoff.png")
    frontier = ex.band_frontier()
    _show("Cost against error: time-based versus bands", frontier.set_index(["rule", "setting"]),
          res / "08_band_frontier.csv")
    plots.save(plots.fig_frontier(frontier), figs / "10_band_frontier.png")

    # 6. Which vol to hedge at ------------------------------------------------
    hv, hv_paths = ex.hedge_vol_study()
    _show(f"Hedge at implied or realised vol (locked-in value "
          f"{hv.attrs['locked_in']:.3f})", hv, res / "09_hedge_vol.csv")
    plots.save(plots.fig_hedge_vol(hv, hv_paths), figs / "11_hedge_vol.png")

    # 7. Attribution ----------------------------------------------------------
    S_hot = simulate_gbm(S0, R, 0.25, T, 63, 1, np.random.default_rng(12))[0]
    att = attribute_pnl(S_hot, K, T, R, SIGMA)
    _show("P&L attribution on one path (realised 25%, implied 20%): column totals",
          att.drop(columns=["t", "S"]).sum().to_frame("sum"))
    att.to_csv(res / "10_attribution.csv", float_format="%.6f")
    plots.save(plots.fig_attribution(att), figs / "12_attribution.png")

    # 8. Real data ------------------------------------------------------------
    if not args.no_real_data:
        from . import realdata as rd

        try:
            data = rd.load_market_data(refresh=args.refresh_data)
        except OSError as err:  # no network, or FRED refused
            print(f"\nSkipping the real-data replay: could not download from FRED ({err}).")
        else:
            rd.write_manifest()
            roll = rd.rolling_replay(data)
            summary = roll[["implied", "realised", "premium_pct", "pnl_pct", "vega_pred_pct",
                            "gamma_pred_pct", "unhedged_pct"]].describe().T
            _show(f"S&P 500 replay, {len(roll)} overlapping 3-month windows "
                  f"({roll.index[0]:%Y-%m-%d} to {roll.index[-1]:%Y-%m-%d})",
                  summary, res / "11_rolling_replay_summary.csv")
            print(f"share of windows with positive hedged P&L: {(roll.pnl_pct > 0).mean():.3f}")
            plots.save(plots.fig_rolling_vols(roll), figs / "13_rolling_vols.png")
            plots.save(plots.fig_rolling_pnl(roll), figs / "14_rolling_pnl.png")
            for tag, start in (("calm", "2017-06-01"), ("crash", "2020-02-03")):
                led, s = rd.replay_window(data, start)
                _show(f"Single window starting {s['start']:%Y-%m-%d}",
                      pd.Series(s).to_frame("value"))
                fig = plots.fig_single_path(
                    led, s["K"], x=led.index, xlabel="Date", value_col="pnl",
                    value_label="Mark-to-market P&L", price_label="S&P 500",
                    title=f"S&P 500: 3-month ATM call sold on {s['start']:%d %b %Y}",
                    note=f"implied {100 * s['implied']:.1f}%, "
                         f"realised {100 * s['realised']:.1f}%")
                plots.save(fig, figs / f"15_replay_{tag}.png")

    print(f"\nCharts saved to {figs}/, tables to {res}/")


if __name__ == "__main__":
    main()
