"""Chart helpers. Each ``fig_*`` function takes experiment output and returns a Figure.

Conventions: one y-scale per panel (never a dual axis), a legend whenever
there are two or more series, thin lines, solid hairline grid, and three
series colours at most, in a fixed order (blue, orange, aqua) checked for
colour-blind separation. Reference lines (theory, zero, strike) are grey.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .black_scholes import bs_delta, bs_price

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 9.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
    "axes.prop_cycle": mpl.cycler(color=SERIES),
}


def use_style():
    """Apply the chart style to every figure made after this call."""
    mpl.rcParams.update(STYLE)


def _hist(ax, x, color, label, bins, density=True):
    """Outline histogram with a light wash, so overlapping series stay readable."""
    ax.hist(x, bins=bins, density=density, histtype="stepfilled", color=color, alpha=0.12, lw=0)
    ax.hist(x, bins=bins, density=density, histtype="step", color=color, lw=1.8, label=label)


def _zero_line(ax, axis="y", value=0.0):
    (ax.axhline if axis == "y" else ax.axvline)(value, color=AXIS, lw=1.2, zorder=1)


def _edge_label(ax, y, text):
    """Label a horizontal reference line just outside the right edge of the axes."""
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK_2,
                annotation_clip=False)


def _note(ax, text, xy=(0.02, 0.96), ha="left", va="top"):
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9, color=INK_2,
            linespacing=1.4)


# --- concept ---------------------------------------------------------------

def fig_delta_tangent(S0=100.0, K=100.0, T=0.25, r=0.04, sigma=0.20):
    """Option value against the stock price, with the tangent whose slope is delta."""
    S = np.linspace(75, 125, 300)
    value = bs_price(S, K, T, r, sigma)
    c0, d0 = bs_price(S0, K, T, r, sigma), bs_delta(S0, K, T, r, sigma)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(S, np.maximum(S - K, 0), color=MUTED, lw=1.5, label="Payoff at expiry")
    ax.plot(S, value, color=BLUE, label="Call value today (3 months left)")
    near = (S > S0 - 14) & (S < S0 + 14)
    ax.plot(S[near], c0 + d0 * (S[near] - S0), color=ORANGE, lw=1.8,
            label=f"Tangent at S = {S0:.0f}: slope = delta = {d0:.2f}")
    ax.plot([S0], [c0], "o", color=BLUE, ms=8, mec=SURFACE, mew=2, zorder=5)
    ax.annotate("The gap between curve and tangent\nis gamma: the hedge is exact only\nfor small moves.",
                xy=(113.5, 0.5 * (bs_price(113.5, K, T, r, sigma) + c0 + d0 * (113.5 - S0))),
                xytext=(106.5, -1.5), fontsize=9, color=INK_2, va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set(xlabel="Stock price S", ylabel="Value of the call",
           title="Delta is the slope of the option value")
    ax.legend(loc="upper left")
    return fig


def fig_sample_paths(S, T, n_show=40, K=None):
    """A handful of simulated price paths plus the terminal distribution."""
    t = np.linspace(0, T, S.shape[1]) * 252
    fig, (ax, ah) = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
    ax.plot(t, S[:n_show].T, color=BLUE, lw=0.9, alpha=0.35)
    ax.plot(t, S[0], color=BLUE, lw=2.0)
    if K is not None:
        ax.axhline(K, color=MUTED, lw=1.0)
        _note(ax, "grey line: strike")
    ax.set(xlabel="Trading days", ylabel="Stock price",
           title=f"{n_show} simulated GBM paths, one highlighted")
    ah.hist(S[:, -1], bins=60, orientation="horizontal", color=BLUE, alpha=0.75)
    ah.set(xlabel="Paths", title="Price at expiry")
    ah.set_xticks([])
    return fig


def fig_single_path(ledger, K, title="One path, step by step", x=None, xlabel="Trading days",
                    value_col="hedging_error", value_label="Hedging error so far",
                    price_label="Stock price", note=None):
    """Stock price, hedge ratio and running hedging error on one path."""
    x = np.arange(len(ledger)) if x is None else x
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1, 1], "hspace": 0.22})
    axes[0].plot(x, ledger["S"], color=BLUE)
    axes[0].axhline(K, color=MUTED, lw=1.0)
    _edge_label(axes[0], K, "strike")
    axes[0].set(ylabel=price_label, title=title)
    if note:  # right-aligned on the title line
        axes[0].text(1.0, 1.035, note, transform=axes[0].transAxes, ha="right", va="bottom",
                     fontsize=9, color=INK_2)
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        locator = mpl.dates.AutoDateLocator(minticks=4, maxticks=7)
        axes[2].xaxis.set_major_locator(locator)
        axes[2].xaxis.set_major_formatter(
            mpl.dates.ConciseDateFormatter(locator, show_offset=False))
    axes[1].step(x, ledger["delta"], where="post", color=ORANGE)
    axes[1].set(ylabel="Shares held (delta)", ylim=(-0.03, 1.03))
    axes[2].plot(x, ledger[value_col], color=AQUA)
    _zero_line(axes[2])
    axes[2].set(ylabel=value_label, xlabel=xlabel)
    end = ledger[value_col].iloc[-1]
    axes[2].annotate(f"{end:+.2f}", xy=(x[-1], end), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=9, color=INK_2, annotation_clip=False)
    fig.align_ylabels(axes)
    return fig


# --- Monte Carlo -----------------------------------------------------------

def fig_hedged_vs_unhedged(pnl, n=63):
    """Distribution of P&L for the naked short call and the delta-hedged one."""
    hedged, raw = pnl[n], pnl["unhedged"]
    bins = np.linspace(np.percentile(raw, 0.3), raw.max() + 0.3, 110)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    _hist(ax, raw, ORANGE, f"Unhedged short call (std {raw.std():.2f})", bins)
    _hist(ax, hedged, BLUE, f"Delta-hedged daily (std {hedged.std():.2f})", bins)
    _zero_line(ax, "x")
    kept = float(np.mean(np.isclose(raw, raw.max())))
    ax.annotate(f"Call expires worthless on {100 * kept:.0f}% of paths:\n"
                "the unhedged seller keeps the premium",
                xy=(raw.max() - 0.25, 0.82), xycoords=("data", "axes fraction"),
                xytext=(raw.max() - 2.2, 0.82), textcoords=("data", "axes fraction"),
                ha="right", va="center", fontsize=9, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set(xlabel="P&L at expiry", ylabel="Density", yticks=[],
           title="Hedging turns a wide directional bet into a narrow one")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55))
    return fig


def fig_error_by_frequency(table, pnl):
    """Left: hedging-error distributions for each N. Right: std against N on log axes."""
    fig, (ax, ar) = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"wspace": 0.24})
    ns = list(table.index)
    lim = 3.2 * table["std"].max()
    bins = np.linspace(-lim, lim, 90)
    for n, color in zip(ns, SERIES):
        _hist(ax, pnl[n], color, f"N = {n}  (std {table.loc[n, 'std']:.2f})", bins)
    _zero_line(ax, "x")
    ax.set(xlabel="Hedging error at expiry", ylabel="Density", yticks=[],
           title="More rebalances, tighter error, same mean")
    ax.legend(loc="upper left")

    grid = np.array([ns[0] * 0.8, ns[-1] * 1.25])
    c = table["derman_kamal_std"].iloc[0] * np.sqrt(ns[0])
    ar.plot(grid, c / np.sqrt(grid), color=MUTED, lw=1.5, label="Derman-Kamal rule of thumb")
    ar.plot(ns, table["std"], "o", color=BLUE, ms=8, mec=SURFACE, mew=2, label="Monte Carlo",
            zorder=5)
    for n in ns:
        ar.annotate(f"{table.loc[n, 'std']:.2f}", (n, table.loc[n, "std"]), xytext=(8, 6),
                    textcoords="offset points", fontsize=9, color=INK_2)
    ar.set(xscale="log", yscale="log", xlabel="Rebalances N (log scale)",
           ylabel="Std of hedging error (log scale)",
           title=f"Std falls like 1/sqrt(N): slope {table.attrs['loglog_slope']:.2f}")
    ar.set_xticks(ns, [str(n) for n in ns])
    ar.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    yt = [0.2, 0.3, 0.5, 0.8]
    ar.set_yticks(yt, [f"{v:.1f}" for v in yt])
    ar.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ar.legend(loc="upper right")
    return fig


def fig_step_vs_total(samples, table):
    """One step's error is skewed; the sum over the option's life is close to normal."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"wspace": 0.12})
    titles = {"first step": "One rebalancing interval: skewed",
              "whole life": "Summed over 63 intervals: far closer to normal"}
    for ax, (name, x) in zip(axes, samples.items()):
        z = (x - x.mean()) / x.std()
        bins = np.linspace(-6, 3.5, 100)
        _hist(ax, z, BLUE, "Simulated", bins)
        grid = np.linspace(-6, 3.5, 300)
        ax.plot(grid, norm.pdf(grid), color=MUTED, lw=1.5, label="Normal curve")
        ax.set(xlabel="Hedging error, in standard deviations", yticks=[], title=titles[name])
        _note(ax, f"skew {table.loc[name, 'skew']:+.2f}\n"
                  f"gains on {100 * table.loc[name, 'share_gains']:.0f}% of paths")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.78))
    axes[0].set_ylabel("Density")
    return fig


def fig_vol_bet(table, sigma_implied=0.20):
    """Mean hedged P&L of the option seller against the vol the stock actually delivered."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = table.index.to_numpy() * 100
    ax.plot(x, table["vega_rule"], color=MUTED, lw=1.5,
            label="Rule of thumb: vega x (implied - realised)")
    ax.errorbar(x, table["mean"], yerr=table["std"], fmt="none", ecolor=BLUE, elinewidth=1.2,
                alpha=0.45, capsize=3)
    ax.plot(x, table["mean"], "o", color=BLUE, ms=8, mec=SURFACE, mew=2, zorder=5,
            label="Monte Carlo mean (bar = 1 std across paths)")
    _zero_line(ax)
    ax.axvline(sigma_implied * 100, color=AXIS, lw=1.2)
    ax.text(sigma_implied * 100, ax.get_ylim()[1], " sold at 20% implied", va="top", fontsize=8.5,
            color=INK_2)
    for xv, m in zip(x, table["mean"]):
        ax.annotate(f"{round(m, 2) + 0.0:+.2f}", (xv, m), xytext=(9, 7),
                    textcoords="offset points", fontsize=9,
                    color=INK_2)
    ax.set(xlabel="Realised volatility of the stock (%)", ylabel="Seller's hedged P&L",
           title="A hedged option is a bet on realised versus implied volatility")
    ax.set_xticks(x)
    ax.legend(loc="lower left")
    return fig


def fig_path_dependence(detail, K=100.0, n_show=4000, seed=0):
    """Same vol gap, different outcomes: the P&L depends on where the path spent its time."""
    idx = np.random.default_rng(seed).choice(len(detail["pnl"]), n_show, replace=False)
    S_T, pnl, pred = detail["S"][idx, -1], detail["pnl"][idx], detail["predicted"][idx]
    fig, (ax, ar) = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"wspace": 0.24})
    ax.scatter(S_T, pnl, s=7, color=BLUE, alpha=0.30, lw=0)
    ax.axvline(K, color=MUTED, lw=1.0)
    ax.text(K, ax.get_ylim()[0], " strike", va="bottom", fontsize=8.5, color=INK_2)
    _zero_line(ax)
    ax.set(xlabel="Stock price at expiry", ylabel="Seller's hedged P&L",
           title="Worst losses: the stock ends near the strike")
    lo, hi = np.percentile(pnl, [0.2, 99.9])
    ar.plot([lo, hi], [lo, hi], color=MUTED, lw=1.5, label="Perfect agreement")
    ar.scatter(pred, pnl, s=7, color=BLUE, alpha=0.30, lw=0, label="One simulated path")
    corr = np.corrcoef(detail["predicted"], detail["pnl"])[0, 1]
    ar.set(xlabel="Predicted: sum of 1/2 Gamma S^2 x (implied var - realised var)",
           ylabel="Simulated hedged P&L",
           title=f"Formula against simulation, path by path (corr {corr:.2f})")
    ar.legend(loc="upper left")
    return fig


# --- costs -----------------------------------------------------------------

def fig_cost_tradeoff(table, cost_bps=10):
    """Costs rise with N while the error falls, so the combined miss has a minimum."""
    n = table.index.to_numpy()
    fig, (ax, ar) = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"wspace": 0.22})
    ax.plot(n, table["mean_cost"], "-o", color=ORANGE, ms=6, mec=SURFACE, mew=1.5,
            label="Average cost paid")
    ax.plot(n, table["leland_uplift"] + table["one_off_cost"], color=MUTED, lw=1.5,
            label="Leland's estimate (plus set-up and unwind)")
    ax.set(xscale="log", xlabel="Rebalances N (log scale)", ylabel="Cost per option sold",
           title=f"Cost rises with frequency ({cost_bps} bp one-way)")
    ax.legend(loc="upper left")

    ar.plot(n, table["std"], "-o", color=BLUE, ms=6, mec=SURFACE, mew=1.5,
            label="Std of hedging error")
    ar.plot(n, table["rmse"], "-o", color=AQUA, ms=6, mec=SURFACE, mew=1.5,
            label="Typical miss: sqrt(mean^2 + std^2)")
    best = table["rmse"].idxmin()
    ar.annotate(f"smallest at N = {best}", (best, table.loc[best, "rmse"]), xytext=(0, 22),
                textcoords="offset points", ha="center", fontsize=9, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ar.set(xscale="log", xlabel="Rebalances N (log scale)", ylabel="P&L units",
           title="So there is a best frequency", ylim=(0, None))
    ar.legend(loc="upper right")
    for a in (ax, ar):
        a.set_xticks(n, [str(v) for v in n])
        a.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    return fig


def fig_frontier(frontier):
    """Cost against hedging error for time-based rebalancing and two band rules."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = {"time-based": "Time-based (every m-th check)",
              "fixed band": "Fixed delta band",
              "gamma band": "Gamma-scaled band (Whalley-Wilmott)"}
    for (rule, label), color in zip(labels.items(), SERIES):
        d = frontier[frontier["rule"] == rule].sort_values("mean_cost")
        ax.plot(d["mean_cost"], d["std"], "-o", color=color, ms=6, mec=SURFACE, mew=1.5,
                label=label)
    tb = frontier[frontier["rule"] == "time-based"].set_index("setting")
    for setting in ("N = 21", "N = 63", "N = 252"):
        if setting in tb.index:
            ax.annotate(setting, (tb.loc[setting, "mean_cost"], tb.loc[setting, "std"]),
                        xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set(xlabel="Average transaction cost paid", ylabel="Std of hedging error",
           title="Bands buy more risk reduction per unit of cost", ylim=(0, None))
    _note(ax, "lower-left is better", xy=(0.03, 0.06), va="bottom")
    ax.legend(loc="upper right")
    return fig


# --- which vol to hedge at -------------------------------------------------

def fig_hedge_vol(table, paths):
    """Hedge at implied: smooth ride, uncertain total. Hedge at realised: the reverse."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), gridspec_kw={"wspace": 0.26})
    n = max(table.index.get_level_values("N"))
    both = np.concatenate([paths["implied"]["pnl"], paths["realised"]["pnl"]])
    bins = np.linspace(np.percentile(both, 0.2), np.percentile(both, 99.9), 80)
    for label, color in (("implied", BLUE), ("realised", ORANGE)):
        std = table.loc[(n, label), "std"]
        _hist(axes[0], paths[label]["pnl"], color, f"Hedge at {label} vol (std {std:.2f})", bins)
    axes[0].axvline(table.attrs["locked_in"], color=MUTED, lw=1.2)
    axes[0].set(xlabel="P&L at expiry", ylabel="Density", yticks=[],
                title="Total P&L at expiry")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.32)
    axes[0].legend(loc="upper left", fontsize=8.5)
    lo = min(paths[k]["value"].min() for k in paths)
    hi = max(paths[k]["value"].max() for k in paths)
    pad = 0.06 * (hi - lo)
    for ax, (label, color) in zip(axes[1:], (("implied", BLUE), ("realised", ORANGE))):
        t = paths[label]["t"] * 252
        ax.plot(t, paths[label]["value"].T, color=color, lw=1.0, alpha=0.8)
        _zero_line(ax)
        ax.axhline(table.attrs["locked_in"], color=MUTED, lw=1.2)
        ax.set(xlabel="Trading days", ylim=(lo - pad, hi + pad),
               title=f"Hedged at {label} vol")
    axes[1].set_ylabel("Mark-to-market P&L (6 paths)")
    _edge_label(axes[2], table.attrs["locked_in"], "locked-in\nvalue")
    return fig


# --- attribution -----------------------------------------------------------

def fig_attribution(df, title="Where the P&L came from, cumulatively", x=None,
                    xlabel="Trading days"):
    """Cumulative theta-plus-financing income, gamma loss and the total, on one path."""
    x = np.arange(1, len(df) + 1) if x is None else x
    carry = (df["theta"] + df["financing"] + df["dividend"]).cumsum()
    gamma = df["gamma"].cumsum()
    other = (df["vega"] + df["delta_mismatch"] + df["cost"] + df["residual"]).cumsum()
    total = df.attrs["initial"] + df["total"].cumsum()
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.plot(x, carry, color=BLUE, label="Theta income, net of financing")
    ax.plot(x, gamma, color=ORANGE, label="Gamma losses:  -1/2 Gamma dS^2")
    ax.plot(x, total, color=AQUA, label="Total hedged P&L")
    ax.plot(x, other, color=MUTED, lw=1.3, label="Everything else (vega, costs, residual)")
    _zero_line(ax)
    for series in (carry, gamma, total):
        ax.annotate(f"{series.iloc[-1]:+.2f}", (x[-1], series.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2,
                    annotation_clip=False)
    ax.set(xlabel=xlabel, ylabel="Cumulative P&L", title=title)
    ax.legend(loc="upper left")
    return fig


# --- real data -------------------------------------------------------------

def fig_rolling_vols(roll):
    """3-month implied vol on each date against the vol realised over the next 3 months."""
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.plot(roll.index, roll["implied"] * 100, color=BLUE, lw=1.6,
            label="Implied on the start date (Cboe 3-month index)")
    ax.plot(roll.index, roll["realised"] * 100, color=ORANGE, lw=1.6,
            label="Realised over the following 63 trading days")
    ax.set(ylabel="Volatility (%)", xlabel="Start date of the 3-month window",
           title="S&P 500: implied volatility usually sat above what was then realised",
           ylim=(0, None))
    ax.legend(loc="upper right")
    return fig


def fig_rolling_pnl(roll):
    """Left: hedged seller's P&L by start date. Right: P&L against the vol gap."""
    fig, (ax, ar) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                 gridspec_kw={"wspace": 0.2, "width_ratios": [1.25, 1]})
    ax.fill_between(roll.index, roll["pnl_pct"], 0, color=BLUE, alpha=0.12, lw=0)
    ax.plot(roll.index, roll["pnl_pct"], color=BLUE, lw=1.4)
    _zero_line(ax)
    ax.set(xlabel="Start date of the 3-month window", ylabel="Hedged P&L, % of index level",
           title="Seller's delta-hedged P&L for each start date")
    gap = (roll["implied"] - roll["realised"]) * 100
    ar.scatter(gap, roll["pnl_pct"], s=7, color=BLUE, alpha=0.30, lw=0,
               label="One 3-month window")
    grid = np.array([gap.min(), gap.max()])
    slope = (roll["vega_pred_pct"] / gap).replace([np.inf, -np.inf], np.nan).median()
    ar.plot(grid, slope * grid, color=MUTED, lw=1.5, label="Rule of thumb: vega x gap")
    _zero_line(ar)
    _zero_line(ar, "x")
    ar.set(xlabel="Implied minus realised volatility (vol points)",
           ylabel="Hedged P&L, % of index level",
           title="The P&L tracks the volatility gap")
    ar.legend(loc="upper left")
    return fig


def save(fig, path):
    """Save a figure as PNG and close it."""
    fig.savefig(path)
    plt.close(fig)
    return path
