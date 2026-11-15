"""Delta hedging simulator: Black-Scholes pricing, GBM paths and a self-financing hedge.

Built from "Delta Hedging, From First Principles to a Working Simulator".
Start with ``notebooks/delta_hedging_walkthrough.ipynb`` or run
``python -m delta_hedging``.
"""

from .analytics import derman_kamal_std, loglog_slope, realised_vol, summarize
from .attribution import attribute_pnl, gamma_weighted_pnl
from .black_scholes import bs_d1_d2, bs_delta, bs_greeks, bs_price
from .costs import leland_number, leland_vol, whalley_wilmott_band
from .hedging import (
    HedgeResult,
    hedge_ledger,
    hedge_short_option,
    payoff,
    unhedged_short_option,
)
from .paths import simulate_gbm

__all__ = [
    "HedgeResult",
    "attribute_pnl",
    "bs_d1_d2",
    "bs_delta",
    "bs_greeks",
    "bs_price",
    "derman_kamal_std",
    "gamma_weighted_pnl",
    "hedge_ledger",
    "hedge_short_option",
    "leland_number",
    "leland_vol",
    "loglog_slope",
    "payoff",
    "realised_vol",
    "simulate_gbm",
    "summarize",
    "unhedged_short_option",
    "whalley_wilmott_band",
]

# The PDF's default set-up: short a 3-month at-the-money call.
DEFAULTS = {"S0": 100.0, "K": 100.0, "T": 0.25, "r": 0.04, "sigma": 0.20}
