# Sources and data

Everything this demo relies on, and how far each item was checked.

## 1. The reference document

**"Delta Hedging, From First Principles to a Working Simulator: Concept, Mathematics, Python and Sources"** (14-page PDF supplied by the repository owner).

The package structure, the default parameters, the formulas, the hedging loop, the sanity checks, the list of common bugs and the interview questions all come from it. The PDF itself is not stored in this repository.

## 2. Data

### 2.1 Synthetic data (Parts 1 to 8 and 10 of the notebook)

No market data is used. Price paths are exact geometric Brownian motion drawn with `numpy.random.default_rng(seed)` (PCG64). Every experiment fixes its seed, so the numbers reproduce exactly on the same library versions.

| Parameter | Value | Note |
|---|---|---|
| S0, K | 100, 100 | at-the-money call |
| T | 0.25 years (63 trading days) | |
| r | 4% continuous | stylised |
| implied vol | 20% | stylised |
| realised vol | 20% unless the experiment varies it | |
| paths per experiment | 20,000 (5,000 to 50,000 where stated) | |
| transaction cost | 0 or 10 bp one-way | |

These are stylised inputs, not calibrated to any instrument. The saved result tables are in `outputs/results/` and the charts in `outputs/figures/`.

### 2.2 Market data (Part 9 of the notebook)

All four series come from FRED, the Federal Reserve Bank of St. Louis database, through its public CSV endpoint `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`. No API key is needed.

| Series | Description | Observations used from | Rows | Retrieved (UTC) | SHA-256 (first 16) |
|---|---|---|---|---|---|
| [`SP500`](https://fred.stlouisfed.org/series/SP500) | S&P 500 index, daily close (S&P Dow Jones Indices LLC) | 2016-11-14 to 2026-11-13 | 2,514 | 2026-11-15 20:59 | `6e092f266c10be5d` |
| [`DGS3MO`](https://fred.stlouisfed.org/series/DGS3MO) | 3-month Treasury constant-maturity yield, investment basis (Board of Governors, H.15) | 1981-09-01 to 2026-11-12 | 11,299 | 2026-11-15 20:59 | `89bb7be934f8149c` |
| [`VXVCLS`](https://fred.stlouisfed.org/series/VXVCLS) | Cboe S&P 500 3-Month Volatility Index, daily close | 2007-12-04 to 2026-11-12 | 4,766 | 2026-11-15 20:59 | `0cb586abd86a4a68` |
| [`VIXCLS`](https://fred.stlouisfed.org/series/VIXCLS) | Cboe VIX, daily close | 1990-01-02 to 2026-11-12 | 9,316 | 2026-11-15 20:59 | `31a10b41ad993871` |

Full checksums and URLs are in [`data/manifest.json`](../data/manifest.json). After aligning on days when all four are available, the replay uses 2,514 trading days from 2016-11-14 to 2026-11-13, which gives 2,451 overlapping 63-day windows. FRED carries only ten years of `SP500`, which sets the start date.

How the series are used:

* `SP500` is the underlying. It is a price index, so dividends are modelled separately.
* `DGS3MO` is converted from a bond-equivalent yield `y` to a continuous rate with `ln(1 + y * 91/365) / (91/365)`.
* `VXVCLS / 100` is the volatility the option is sold at. `VIXCLS` is loaded for reference only.
* Rates and vols are carried forward up to five days over dates when their market was closed and equities traded. The last equity date in the sample (2026-11-13) uses the previous day's rate and vol for that reason.

**An assumption that is not data:** the dividend yield is a constant `q = 1.3%`. It was chosen as a round, plausible figure for the S&P 500 and was not taken from any source.

**What is and is not in the repository.**

* `data/raw/DGS3MO.csv` is committed. It is a Federal Reserve Board statistical release (H.15), which I understand to be a U.S. government work in the public domain.
* `SP500`, `VXVCLS` and `VIXCLS` are **not** committed. They are owned by S&P Dow Jones Indices and Cboe, and my understanding is that FRED's series notes restrict reproducing them. The package downloads them on first use into `data/raw/` (git-ignored) and the manifest records exactly which files were used.
* FRED's series pages could not be opened from the build environment (the site refuses scripted page requests), so the licence wording above was **not** read from the pages themselves. Check the "Notes" on each series page before redistributing anything.

To download the data again and rebuild the manifest:

```bash
python -m delta_hedging --refresh-data
```

Caveats on interpretation are in Part 9 of the notebook and in the docstring of `delta_hedging/realdata.py`.

## 3. Bibliography

These are the references listed in the PDF (its section 4.3), with the details and links as given there. The PDF states they were checked against publisher, journal or repository pages. They were **not** independently re-verified while building this repository.

| Source | Details | Link or identifier | Used here for |
|---|---|---|---|
| Black, F. & Scholes, M. (1973), "The Pricing of Options and Corporate Liabilities" | *Journal of Political Economy* 81(3), 637-654 | https://doi.org/10.1086/260062 | Hedging argument, formula (notebook 2.3, 2.4) |
| Merton, R.C. (1973), "Theory of Rational Option Pricing" | *Bell Journal of Economics and Management Science* 4(1), 141-183 | https://doi.org/10.2307/3003143 | Alternative derivation; dividend-yield extension used in `black_scholes.py` |
| Hull, J.C., *Options, Futures, and Other Derivatives*, 11th ed. | Pearson, 2021 (© 2022); ISBN 978-0-13-693997-9 (US), 978-1-292-41065-4 (Global) | pearson.com | Greeks, textbook derivations, futures hedging |
| Kamal, M. & Derman, E., "When You Cannot Hedge Continuously: The Corrections to Black-Scholes" | Goldman Sachs Equity Derivatives Research, *Quantitative Insights* (Quantitative Strategies Research Notes), PDF marked © 1998 | https://emanuelderman.com/wp-content/uploads/1998/12/risk-non_continuous_hedge.pdf | Rule of thumb for hedging-error std (`derman_kamal_std`, notebook 4.1) |
| Boyle, P.P. & Emanuel, D. (1980), "Discretely Adjusted Option Hedges" | *Journal of Financial Economics* 8(3), 259-282 | https://doi.org/10.1016/0304-405X(80)90003-3 | Skewed per-step hedging error (notebook 4.2) |
| Leland, H.E. (1985), "Option Pricing and Replication with Transactions Costs" | *Journal of Finance* 40(5), 1283-1301 | https://doi.org/10.1111/j.1540-6261.1985.tb02383.x | Adjusted volatility (`leland_vol`, notebook 6.1) |
| Kabanov, Yu.M. & Safarian, M.M. (1997), "On Leland's strategy of option pricing with transactions costs" | *Finance and Stochastics* 1, 239-250 | https://link.springer.com/content/pdf/10.1007/s007800050023.pdf | Caveat on Leland's limit claim; one-way versus round-trip cost |
| Ahmad, R. & Wilmott, P. (2005), "Which Free Lunch Would You Like Today, Sir?: Delta Hedging, Volatility Arbitrage and Optimal Portfolios" | *Wilmott* magazine, November 2005, 64-79 | wilmott.com | Hedging at implied versus realised vol (notebook Part 7) |
| Carr, P., "FAQ's in Option Pricing Theory" | Working paper, Banc of America Securities (dated 1999 in most citations) | http://www.math.nyu.edu/research/carrp/papers/pdf/faq2.pdf | Choice of hedging volatility |
| El Karoui, N., Jeanblanc-Picqué, M. & Shreve, S.E. (1998), "Robustness of the Black and Scholes Formula" | *Mathematical Finance* 8(2), 93-126 | https://doi.org/10.1111/1467-9965.00047 | Gamma-weighted P&L formula (`gamma_weighted_pnl`, notebook 2.6, 5.1) |
| Hodges, S.D. & Neuberger, A. (1989), "Optimal Replication of Contingent Claims under Transactions Costs" | *Review of Futures Markets* 8(2), 222-239 | no DOI | No-trade band as the optimal policy |
| Whalley, A.E. & Wilmott, P. (1997), "An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs" | *Mathematical Finance* 7(3), 307-324 | https://doi.org/10.1111/1467-9965.00034 | Gamma-scaled band (`whalley_wilmott_band`, notebook 6.2) |
| Bakshi, G. & Kapadia, N. (2003), "Delta-Hedged Gains and the Negative Market Volatility Risk Premium" | *Review of Financial Studies* 16(2), 527-566 | https://doi.org/10.1093/rfs/hhg002 | Empirical evidence compared with the replay (notebook Part 9) |
| Glasserman, P. (2003), *Monte Carlo Methods in Financial Engineering* | Springer, Stochastic Modelling and Applied Probability vol. 53; ch. 3 | https://doi.org/10.1007/978-0-387-21617-1 | Exact GBM simulation, Monte Carlo error |
| Taleb, N.N. (1997), *Dynamic Hedging: Managing Vanilla and Exotic Options* | Wiley, ISBN 978-0-471-15280-4 | wiley.com | Practitioner background |
| Sinclair, E. (2013), *Volatility Trading*, 2nd ed. | Wiley, ISBN 978-1-118-34713-3 | wiley.com | Realised versus implied, practical volatility trading |
| SciPy `scipy.stats.norm` | Official reference | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html | `N(.)` and its density |
| NumPy `Generator.standard_normal`, `default_rng` | Official reference | https://numpy.org/doc/stable/reference/random/generator.html | Random draws |
| FRED | Federal Reserve Bank of St. Louis | https://fred.stlouisfed.org | Market data (section 2.2) |

## 4. Things in this repository that are not from the PDF

Stated so you can weigh them accordingly.

* **The constant in the Whalley-Wilmott band.** The PDF gives only the scaling `(cost x S x Gamma^2 / risk-aversion)^(1/3)`. The code uses `(3/2 x cost x S x e^(-r tau) x Gamma^2 / risk-aversion)^(1/3)`, the form usually quoted from the 1997 paper, written from memory and not checked against the paper. The qualitative result (bands beat clock-based rebalancing) does not depend on the constant, because the experiment sweeps the risk-aversion parameter.
* **Trading back to the edge of the band.** The PDF's snippet trades all the way to the target delta. The frontier chart uses trade-to-edge, which is what the cost literature recommends. Both are implemented (`band_mode`), and both appear in `outputs/results/08_band_frontier.csv`.
* **The `rmse` column in the cost study**, `sqrt(mean^2 + std^2)`, is my own summary for showing that an interior rebalancing frequency exists. It is not from the PDF and is one of several reasonable ways to combine cost and risk.
* **Dividends.** The dividend yield `q` in the pricing functions (Merton's extension) and the dividend cash flow in the hedging loop were added for the index replay.
* **The real-data experiment design**: rolling 63-day windows, hedging at the start-date vol, P&L in per cent of the index level, the choice of `VXVCLS` as the implied-vol input and `SP500` as the underlying. The PDF suggests the FRED rate and volatility series and leaves the price source open.

## 5. Software

Built and run with Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2, pytest 9.2.0, nbconvert 7.17.2.
