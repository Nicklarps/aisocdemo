"""Replay the hedge on a real underlying, using free data from FRED (PDF section 4.2).

Inputs, all daily, all from the Federal Reserve Bank of St. Louis (FRED):

    SP500    S&P 500 index close (price index, dividends not included)
    DGS3MO   3-month Treasury constant-maturity yield, per cent, quoted on an
             investment (bond-equivalent) basis
    VXVCLS   Cboe S&P 500 3-Month Volatility Index, per cent
    VIXCLS   Cboe VIX (30-day), per cent, kept for reference

What the replay does: on a start date, sell a 3-month at-the-money call on
the index at that day's 3-month implied volatility, delta-hedge it once a
day at that same volatility, and settle at expiry. It then repeats this for
every start date in the history.

Read the results with the PDF's caveats in mind:

* The VIX family measures variance-swap-style volatility across many
  strikes. It is NOT the Black-Scholes implied vol of one ATM option.
* Daily closes hide intraday moves; overnight gaps count as realised vol.
* The dividend yield ``q`` is an assumed constant, not downloaded data.
* Overlapping 3-month windows are far from independent, and one history is
  one sample. Nothing here is evidence of a tradable edge.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import realised_vol
from .attribution import gamma_weighted_pnl
from .black_scholes import bs_greeks
from .hedging import hedge_short_option

__all__ = [
    "FRED_SERIES",
    "parse_fred_csv",
    "fetch_fred",
    "bond_equivalent_to_continuous",
    "load_market_data",
    "make_windows",
    "replay_window",
    "rolling_replay",
    "write_manifest",
]

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_PAGE_URL = "https://fred.stlouisfed.org/series/{series_id}"
TRADING_DAYS = 252

FRED_SERIES = {
    "SP500": "S&P 500 index, daily close (S&P Dow Jones Indices LLC)",
    "DGS3MO": "Market yield on U.S. Treasury securities at 3-month constant maturity, "
              "investment basis (Board of Governors, H.15)",
    "VXVCLS": "Cboe S&P 500 3-Month Volatility Index, daily close (Cboe)",
    "VIXCLS": "Cboe Volatility Index: VIX, daily close (Cboe)",
}


def parse_fred_csv(text):
    """Parse a FRED ``fredgraph.csv`` download into a float Series indexed by date.

    FRED marks missing days (holidays) with an empty field or a single dot;
    both become NaN and are dropped.
    """
    df = pd.read_csv(io.StringIO(text), na_values=[".", ""])
    date_col, value_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name=value_col,
    )
    s.index.name = "date"
    return s.dropna()


def fetch_fred(series_id, cache_dir="data/raw", refresh=False, timeout=60):
    """Return one FRED series, downloading it once and caching the raw CSV.

    The raw file is saved untouched as ``<cache_dir>/<series_id>.csv`` so the
    manifest can record its checksum.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{series_id}.csv"
    if refresh or not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Plain urllib with its default headers: FRED drops requests that
        # carry an unfamiliar or browser-like User-Agent.
        url = FRED_CSV_URL.format(series_id=series_id)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read()
        path.write_bytes(raw)
        stamp = cache_dir / f"{series_id}.retrieved"
        stamp.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return parse_fred_csv(path.read_text())


def bond_equivalent_to_continuous(yield_pct, days=91):
    """Convert a bond-equivalent (investment) yield in per cent to a continuous rate.

    A bill yielding y on an investment basis grows 1 to 1 + y * days / 365
    with simple interest. The continuously compounded rate r that does the
    same satisfies e^{r * days / 365} = 1 + y * days / 365.
    """
    frac = days / 365.0
    return np.log1p(np.asarray(yield_pct, dtype=float) / 100.0 * frac) / frac


def load_market_data(cache_dir="data/raw", refresh=False):
    """Download (or read from cache) the four series and align them on index trading days.

    Returns a DataFrame indexed by date with columns
    ``spx`` (index level), ``r`` (continuous 3-month rate), ``iv_3m`` and
    ``iv_1m`` (implied vols as decimals). Rates and vols are carried forward
    over days when their own market was closed but equities traded.
    """
    data = pd.DataFrame({"spx": fetch_fred("SP500", cache_dir, refresh)})
    bill = fetch_fred("DGS3MO", cache_dir, refresh)
    columns = {
        "r": pd.Series(bond_equivalent_to_continuous(bill.to_numpy()), index=bill.index),
        "iv_3m": fetch_fred("VXVCLS", cache_dir, refresh) / 100.0,
        "iv_1m": fetch_fred("VIXCLS", cache_dir, refresh) / 100.0,
    }
    for name, series in columns.items():
        data[name] = series.reindex(data.index, method="ffill", limit=5)
    return data.dropna()


def make_windows(data, n_days=63):
    """Cut the history into every possible ``n_days``-long hedging window.

    Returns a dict: ``S`` with shape (n_windows, n_days + 1), and per-window
    arrays ``start``, ``end``, ``S0``, ``sigma_implied``, ``r``.
    """
    spx = data["spx"].to_numpy()
    if len(spx) <= n_days:
        raise ValueError("not enough history for one window")
    S = np.lib.stride_tricks.sliding_window_view(spx, n_days + 1).copy()
    n = S.shape[0]
    return {
        "S": S,
        "start": data.index[:n],
        "end": data.index[n_days:],
        "S0": S[:, 0],
        "sigma_implied": data["iv_3m"].to_numpy()[:n],
        "r": data["r"].to_numpy()[:n],
    }


def rolling_replay(data, n_days=63, q=0.013, cost=0.0):
    """Sell and delta-hedge a 3-month ATM call starting on every date in the history.

    Returns one row per start date. Money amounts are in per cent of the
    starting index level so that windows years apart are comparable:

        implied, realised    vols over the window (decimals)
        premium_pct          option premium received
        pnl_pct              terminal hedged P&L of the option seller
        vega_pred_pct        rule of thumb: vega x (implied - realised)
        gamma_pred_pct       gamma-weighted variance formula, path by path
        unhedged_pct         P&L from selling the call and not hedging
    """
    w = make_windows(data, n_days)
    S, S0, vol, r = w["S"], w["S0"], w["sigma_implied"], w["r"]
    T = n_days / TRADING_DAYS
    res = hedge_short_option(S, S0, T, r, vol, cost=cost, q=q)
    rv = realised_vol(S, T)
    vega0 = bs_greeks(S0, S0, T, r, vol, q=q)["vega"]
    unhedged = res.premium * np.exp(r * T) - np.maximum(S[:, -1] - S0, 0.0)
    pct = 100.0 / S0
    return pd.DataFrame(
        {
            "end": w["end"],
            "S0": S0,
            "r": r,
            "implied": vol,
            "realised": rv,
            "premium_pct": res.premium * pct,
            "pnl_pct": res.pnl * pct,
            "vega_pred_pct": vega0 * (vol - rv) * pct,
            "gamma_pred_pct": gamma_weighted_pnl(S, S0, T, r, vol, q=q) * pct,
            "unhedged_pct": unhedged * pct,
        },
        index=pd.Index(w["start"], name="start"),
    )


def replay_window(data, start, n_days=63, q=0.013, cost=0.0, remark=True):
    """Hedge one historical window and return its day-by-day ledger.

    ``start`` is a date (the first trading day on or after it is used). The
    hedge delta always uses the implied vol of the start date. If ``remark``
    is True the option is marked each day at that day's 3-month implied vol,
    so the daily P&L includes vega P&L, as it would on a desk. The terminal
    P&L is the same either way.

    Returns (ledger DataFrame indexed by date, summary dict).
    """
    pos = int(data.index.searchsorted(pd.Timestamp(start)))
    if pos + n_days >= len(data):
        raise ValueError("window runs past the end of the data")
    win = data.iloc[pos : pos + n_days + 1]
    S = win["spx"].to_numpy()[None, :]
    S0, vol, r = S[0, 0], float(win["iv_3m"].iloc[0]), float(win["r"].iloc[0])
    T = n_days / TRADING_DAYS
    marks = win["iv_3m"].to_numpy()[None, :] if remark else vol
    res = hedge_short_option(S, S0, T, r, vol, cost=cost, q=q, sigma_mark=marks,
                             return_history=True)
    h = res.history
    ledger = pd.DataFrame(
        {
            "S": S[0],
            "implied": win["iv_3m"].to_numpy(),
            "delta": h["delta"][0],
            "cash": h["cash"][0],
            "option_value": h["option"][0],
            "pnl": h["value"][0],
        },
        index=win.index,
    )
    rv = float(realised_vol(S, T)[0])
    vega0 = float(bs_greeks(S0, S0, T, r, vol, q=q)["vega"])
    summary = {
        "start": win.index[0], "end": win.index[-1], "S0": S0, "K": S0, "T": T, "r": r, "q": q,
        "implied": vol, "realised": rv, "premium": float(res.premium[0]),
        "pnl": float(res.pnl[0]), "vega": vega0, "vega_pred": vega0 * (vol - rv),
        "gamma_pred": float(gamma_weighted_pnl(S, S0, T, r, vol, q=q)[0]),
        "costs": float(res.costs[0]),
    }
    return ledger, summary


def write_manifest(cache_dir="data/raw", path="data/manifest.json"):
    """Record exactly which data files were used: URL, dates, row count, SHA-256."""
    cache_dir = Path(cache_dir)
    entries = []
    for series_id, description in FRED_SERIES.items():
        raw_path = cache_dir / f"{series_id}.csv"
        if not raw_path.exists():
            continue
        raw = raw_path.read_bytes()
        s = parse_fred_csv(raw.decode())
        stamp = cache_dir / f"{series_id}.retrieved"
        entries.append(
            {
                "series_id": series_id,
                "description": description,
                "provider": "FRED, Federal Reserve Bank of St. Louis",
                "download_url": FRED_CSV_URL.format(series_id=series_id),
                "series_page": FRED_PAGE_URL.format(series_id=series_id),
                "retrieved_utc": stamp.read_text().strip() if stamp.exists() else None,
                "first_observation": s.index[0].strftime("%Y-%m-%d"),
                "last_observation": s.index[-1].strftime("%Y-%m-%d"),
                "observations": int(s.size),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"series": entries}, indent=2) + "\n")
    return entries
