"""Summary statistics and the rules of thumb the Monte Carlo is checked against."""

from __future__ import annotations

import numpy as np

__all__ = ["derman_kamal_std", "summarize", "loglog_slope", "realised_vol"]


def derman_kamal_std(vega, sigma, n_rebalances):
    """Kamal & Derman's rule of thumb for the hedging-error standard deviation.

        sigma_PnL ~ sqrt(pi / 4) * vega * sigma / sqrt(N)        (PDF section 2.7)

    ``vega`` is the option's initial vega per 1.00 of vol. Hedging N times is
    like estimating volatility from N samples, so the error falls like
    1 / sqrt(N): hedge four times as often, halve the error.
    """
    return np.sqrt(np.pi / 4.0) * vega * sigma / np.sqrt(n_rebalances)


def summarize(pnl):
    """Mean, its Monte Carlo standard error, std, skew, excess kurtosis and tails.

    Always read the mean together with ``se``: a mean within about two
    standard errors of zero is indistinguishable from zero (PDF bug 8).
    """
    x = np.asarray(pnl, dtype=float)
    n = x.size
    mean = x.mean()
    std = x.std(ddof=1)
    z = (x - mean) / std
    return {
        "n": n,
        "mean": mean,
        "se": std / np.sqrt(n),
        "std": std,
        "skew": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def loglog_slope(n_values, std_values):
    """Slope of log(std) against log(N). Theory says about -0.5."""
    slope, _ = np.polyfit(np.log(n_values), np.log(std_values), 1)
    return float(slope)


def realised_vol(S, T):
    """Annualised realised volatility of each path from its log returns.

    Uses the zero-mean estimator sqrt(sum(log returns^2) / T), the usual
    convention for realised variance.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    log_ret = np.diff(np.log(S), axis=1)
    return np.sqrt(np.sum(log_ret**2, axis=1) / T)
