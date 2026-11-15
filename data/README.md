# Data

* `manifest.json` records every market-data file the real-data replay used: series ID, download URL, retrieval time, date range, row count and SHA-256.
* `raw/DGS3MO.csv` is the 3-month Treasury yield as downloaded from FRED (a Federal Reserve Board H.15 series).
* `raw/SP500.csv`, `raw/VXVCLS.csv` and `raw/VIXCLS.csv` are **not** committed, because S&P Dow Jones Indices and Cboe own them. They are downloaded automatically the first time you run the notebook or `python -m delta_hedging`, and kept in `raw/` (git-ignored).

To download everything again and rewrite the manifest:

```bash
python -m delta_hedging --refresh-data
```

Details, licensing notes and caveats: [`docs/SOURCES.md`](../docs/SOURCES.md).

The synthetic experiments use no files from this folder. Their inputs are a handful of parameters and a random seed, listed in `docs/SOURCES.md`.
