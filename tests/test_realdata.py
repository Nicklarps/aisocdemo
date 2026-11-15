"""Real-data plumbing, tested on a small fixture so no network is needed."""

import json

import numpy as np
import pandas as pd
import pytest

from delta_hedging import hedge_short_option, realdata as rd

FIXTURE = """observation_date,DGS3MO
2026-11-09,4.31
2026-11-10,4.30
2026-11-11,
2026-11-12,4.32
2026-11-13,.
"""


def test_parse_fred_csv_drops_missing_days():
    s = rd.parse_fred_csv(FIXTURE)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-11-09", "2026-11-10", "2026-11-12"]
    assert s.iloc[-1] == pytest.approx(4.32)
    assert s.name == "DGS3MO"


def test_bond_equivalent_to_continuous():
    r = rd.bond_equivalent_to_continuous(4.0)
    frac = 91 / 365
    assert np.exp(r * frac) == pytest.approx(1 + 0.04 * frac)  # same growth over 91 days
    assert 0.0395 < r < 0.04  # continuous rate sits just below the simple rate
    assert rd.bond_equivalent_to_continuous(0.0) == 0.0


def _write_cache(tmp_path, n=140, seed=0):
    """Four fake FRED files: a GBM 'index', flat rates and vols, with one holiday gap."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    spx = 5000 * np.exp(np.cumsum(0.15 / np.sqrt(252) * rng.standard_normal(n)))
    series = {"SP500": spx, "DGS3MO": np.full(n, 4.0), "VXVCLS": np.full(n, 20.0),
              "VIXCLS": np.full(n, 18.0)}
    for name, values in series.items():
        lines = [f"observation_date,{name}"]
        for i, (d, v) in enumerate(zip(dates, values)):
            blank = name == "DGS3MO" and i == 5  # bond market closed, equities open
            lines.append(f"{d:%Y-%m-%d},{'' if blank else f'{v:.2f}'}")
        (tmp_path / f"{name}.csv").write_text("\n".join(lines) + "\n")
    return dates


def test_load_align_and_replay(tmp_path):
    dates = _write_cache(tmp_path)
    data = rd.load_market_data(cache_dir=tmp_path)  # reads the cache, never the network
    assert len(data) == len(dates) and not data.isna().any().any()
    assert data["iv_3m"].iloc[0] == pytest.approx(0.20)
    assert data["r"].iloc[5] == pytest.approx(data["r"].iloc[4])  # holiday carried forward

    windows = rd.make_windows(data, n_days=63)
    assert windows["S"].shape == (len(dates) - 63, 64)
    assert np.all(windows["S"][:, 0] == data["spx"].to_numpy()[: len(dates) - 63])

    roll = rd.rolling_replay(data, n_days=63, q=0.013)
    assert len(roll) == len(dates) - 63
    # Realised vol (about 15%) is below the 20% the option was sold at: the seller gains.
    assert roll["pnl_pct"].mean() > 0
    assert roll["pnl_pct"].corr(roll["gamma_pred_pct"]) > 0.9

    # One window through replay_window matches the same window in the rolling run.
    ledger, s = rd.replay_window(data, dates[10])
    assert s["pnl"] * 100 / s["S0"] == pytest.approx(roll["pnl_pct"].iloc[10])
    assert ledger["pnl"].iloc[-1] == pytest.approx(s["pnl"])
    direct = hedge_short_option(ledger["S"].to_numpy(), s["K"], s["T"], s["r"], s["implied"],
                                q=s["q"]).pnl[0]
    assert direct == pytest.approx(s["pnl"])  # re-marking never changes the terminal P&L

    with pytest.raises(ValueError):
        rd.replay_window(data, dates[-5])


def test_manifest(tmp_path):
    _write_cache(tmp_path)
    out = tmp_path / "manifest.json"
    entries = rd.write_manifest(cache_dir=tmp_path, path=out)
    assert [e["series_id"] for e in entries] == list(rd.FRED_SERIES)
    saved = json.loads(out.read_text())["series"]
    assert saved[0]["observations"] == 140 and len(saved[0]["sha256"]) == 64
    assert saved[1]["observations"] == 139  # the blank holiday row is not an observation
