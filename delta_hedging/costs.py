"""Transaction costs: Leland's adjusted volatility and no-trade bands (PDF section 2.8).

With proportional costs, hedging more often costs more (roughly like
sqrt(N)) while the hedging error shrinks (like 1 / sqrt(N)), so neither
"hedge as often as possible" nor "hedge rarely" is best.
"""

from __future__ import annotations

import numpy as np

__all__ = ["leland_number", "leland_vol", "whalley_wilmott_band"]


def leland_number(sigma, k_round_trip, dt):
    """Leland number  Le = sqrt(2 / pi) * k / (sigma * sqrt(dt)).

    ``k_round_trip`` is the ROUND-TRIP proportional cost as a fraction of
    traded value. The hedging loop takes a ONE-WAY cost, so pass
    ``k_round_trip = 2 * cost``. ``dt`` is the time between rebalances in years.
    """
    return np.sqrt(2.0 / np.pi) * k_round_trip / (sigma * np.sqrt(dt))


def leland_vol(sigma, k_round_trip, dt, short=True):
    """Leland's (1985) cost-adjusted volatility.

        option writer (short):  sigma_hat = sigma * sqrt(1 + Le)
        option buyer  (long):   sigma_hat = sigma * sqrt(1 - Le)   (ill-posed if Le > 1)

    The seller prices AND hedges at the higher sigma_hat, so the extra
    premium covers the expected trading costs.
    """
    le = leland_number(sigma, k_round_trip, dt)
    if short:
        return sigma * np.sqrt(1.0 + le)
    return sigma * np.sqrt(np.maximum(1.0 - le, 0.0))


def whalley_wilmott_band(cost, risk_aversion, r=0.0):
    """Gamma-scaled no-trade band in the Whalley & Wilmott (1997) asymptotic form.

        half-width = (3 * cost * S * e^{-r tau} * Gamma^2 / (2 * lambda)) ^ (1/3)

    ``cost`` is the one-way proportional cost and ``risk_aversion`` (lambda)
    is the hedger's absolute risk aversion: larger lambda means a tighter
    band. The band is widest where gamma is largest, which is exactly where
    a fixed band trades most.

    The PDF states only the cube-root scaling (cost * S * Gamma^2 /
    risk-aversion)^(1/3); the constant 3/2 and the discount factor are the
    form usually quoted from the paper and were not re-derived here.

    Returns a function ``band(S, tau, gamma)`` to pass to
    ``hedge_short_option(..., band=..., band_mode="edge")``.
    """

    def band(S, tau, gamma):
        return (1.5 * cost * S * np.exp(-r * tau) * gamma**2 / risk_aversion) ** (1.0 / 3.0)

    return band
