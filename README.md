# Uranium Market Signals Lab

An educational research workflow for monthly uranium-related equity predictions,
long-only backtests, and historical 3- and 5-year investment outcomes.

Start with the seven numbered notebooks, which explain calculations and include
verification checkpoints.
Reusable code lives in `src/uranium_signals/`.

## Setup and run

Ubuntu/WSL2, Python 3.12 or later, and `uv` are required. Dependencies were approved
for this project and are declared in `pyproject.toml`; `uv.lock` pins the resolved
environment. Do not edit `.env` or commit secrets.

```bash
uv sync --locked
uv run jupyter lab
```

Open `notebooks/01_market_question_and_sources.ipynb` and continue in order.
M2 performs acquisition; M3–M7 require the validated cache. To acquire all inputs
and run the workflow directly from the project root:

```bash
uv run python -m uranium_signals.run --config config/project.yaml
```

To reproduce from existing inputs without network access:

```bash
uv run python -m uranium_signals.run --config config/project.yaml --offline
uv run pytest -q
```

The configured end date is **exclusive** and deliberately fixed at 2026-11-20.
Update it explicitly for a later run. Completed month-ends form signals; an
incomplete current month is excluded. `--refresh` obtains a new provider snapshot
and retains earlier files; it cannot be combined with `--offline`. Cache
checksums are verified before reuse. Never substitute today's holdings or prices
for a missing historical snapshot and call that an exact reproduction.

## Notebook guide for new developers

Read the notebooks in numerical order and run each one's cells from the beginning
in the project environment. They explain the research decisions and calculations;
`src/uranium_signals/` contains the shared implementation used by the notebooks
and CLI.

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [Market question and sources](notebooks/01_market_question_and_sources.ipynb) | Define the hypothesis, review holdings sources, and establish the candidate universe and metadata contracts. |
| 2 | [Download and validate data](notebooks/02_download_and_validate_data.ipynb) | Acquire and cache inputs; inspect currencies, history, missingness, corporate actions, and eligibility. |
| 3 | [Transformations and indicators](notebooks/03_transformations_and_indicators.ipynb) | Work through returns, technical indicators, relative strength, inflation adjustment, and information lags. |
| 4 | [Modeling and walk-forward evaluation](notebooks/04_modeling_and_walk_forward.ipynb) | Construct labels, compare baselines and models, and inspect chronological validation and calibration. |
| 5 | [Strategy and simulation](notebooks/05_strategy_and_simulation.ipynb) | Convert probabilities into holdings, account for execution timing and costs, and compare portfolio outcomes. |
| 6 | [Long-horizon returns](notebooks/06_long_horizon_returns.ipynb) | Analyze rolling 3- and 5-year nominal/real returns, drawdowns, benchmarks, and overlapping samples. |
| 7 | [Automation preparation](notebooks/07_automation_prep.ipynb) | Run the CLI and verify configuration, reports, provenance, and reproducibility. |

M1's snapshot checks require the original local holdings files. M2 can download
market inputs; M3–M7 require the resulting validated cache. See the acquisition
command above if starting without that cache. Launch notebooks from the project
root or `notebooks/`. Each notebook ends with a written-result checkpoint to
review before continuing.

When changing analytical behavior, update the relevant shared module and notebook
explanation together. Run the appropriate tests and use
`uv run python scripts/verify_notebooks.py` to verify notebook execution in clean
kernels. Executed copies are saved locally; source notebooks remain free of
large generated outputs.

## Inputs and study choices

- Initial discovery uses dated URNM, URNJ and NUKZ sponsor snapshots. URA, HURA
  and NLR discovery sources have documented access omissions. See
  `data/reference/m1_source_review.json`.
- `data/reference/universe.csv` contains 16 fixed candidate USD listings and broad
  manual exposure groups. `ticker_to_etf.csv` records observed memberships on
  2026-11-19. These are not historical memberships or historical business classes.
- Eligibility uses at least 252 sessions, a trailing 63-session median approximate
  dollar volume of $1 million, and an adjusted price of at least $1. All are
  configurable and applied at each signal date. Adjusted price times raw volume
  is only a liquidity proxy. New listings are not filled backward.
- Prices: [Yahoo via yfinance](https://ranaroussi.github.io/yfinance/), daily OHLC
  adjusted for vendor corporate actions with unadjusted share volume. Instrument
  metadata verifies USD; source metadata records the provider exchange/name.
  Provider histories can include predecessor entities. Reorganizations, spin-offs
  and ticker continuity are not independently reconstructed; inspect extreme
  returns and action records in M2 before drawing issuer-level conclusions.
- Uranium: [FRED PURANUSDM](https://fred.stlouisfed.org/series/PURANUSDM), IMF
  monthly period-average nominal USD per pound. It is not an executable spot quote.
- Inflation: [FRED CPIAUCNS](https://fred.stlouisfed.org/series/CPIAUCNS), BLS
  monthly US CPI-U all items, not seasonally adjusted, index 1982–84=100.
  October 2025 is missing in the downloaded series. It remains missing; real-return
  windows without endpoint CPI are excluded.
- Macro CSVs use FRED's official download endpoint with the HTTP transport already
  installed by yfinance. Latest revised history is lagged 60 days after observation
  month-end as a publication proxy. This does not establish verified historical
  availability and cannot eliminate revision leakage.
- URA is the primary investable benchmark, SPY the broad-market comparison, and
  URNM is included where history permits. URA begins during 2010, URNM during
  2019; common observation windows are constrained accordingly.

The source register contains provider URLs and snapshot dates. Market cache
manifests add retrieval timestamps, checksums, currency, units, coverage, instrument
metadata and terms notes. Raw data are retained in ignored `data/raw/`; cleaned
outputs in ignored `data/processed/`.

## Modeling and execution

The label is next-month adjusted-close total-return outperformance of URA.
Unavailable future labels stay missing. Model features are explicitly separated
from targets.

Annual expanding fits use only labels completed before each fit cutoff. Median
imputation and scaling are fit inside training pipelines. Majority/base-rate,
momentum, Logistic Regression and Random Forest are compared. XGBoost is omitted.
The selected ML model maximizes pre-2024 validation balanced accuracy; fixed
settings and portfolio rules precede the 2024-onward holdout. During holdout,
annual refits can use earlier outcomes already known at their cutoff. There is
no shuffled validation or holdout-based threshold tuning.

The strategy holds the top three eligible stocks with probability at least 0.5,
equally weighted, otherwise cash. It executes at the next month's first session
adjusted open and rebalances at the following month's first adjusted open. This
interval differs from the classification label. The backtest uses 10 bps per
dollar bought or sold, accounts for weight drift, charges initial purchases,
and does not force final liquidation. It fails on missing held execution prices
and excludes incomplete final intervals. Cash earns zero; reported Sharpe uses
zero risk-free return and 12 monthly periods per year by default.

Comparisons include a hand-written momentum strategy and buy-and-hold URA, SPY,
URNM where available, and an equal-weight universe purchased once at evaluation
inception. Listings unavailable at that inception are excluded from that baseline.

The separate long-horizon study fixes a six-stock basket (CCJ, DNN, UEC, UUUU,
URG, NXE) before viewing returns. It starts at their common monthly inception,
rebalances to equal weights each January month-end, and reports descriptive
returns before costs/tax. Rolling 36/60-month nominal and CPI-deflated outcomes,
CAGR, benchmark differences and monthly observed maximum drawdown accompany
rolling/non-overlapping sample counts. Intramonth losses can exceed monthly
observed drawdowns. Overlapping windows are not independent observations.
Cash/bill comparison is deferred; no long-horizon ML model is fitted.

## Outputs and reproducibility

The CLI prints the report path. `reports/latest_run.txt` points to the latest
successful run; each timestamped directory under `reports/runs/` contains:

- `report.html` plus PNG wealth and real-CAGR charts;
- CSV quality, features, predictions, fit audit, calibration, classification and
  economic metrics, portfolio trades, long-horizon outcomes and latest signals;
- `project.yaml`, a local `model.pkl`, `metadata.json`, and `run.log`.

Metadata captures input/configuration/universe hashes, versions, selection,
assumptions, output checksums and signal date. Only unpickle your own trusted
local model. Failures write `failure.json` and exit non-zero without advancing
the latest-success pointer. Reports are ignored by Git and are not published.

To validate notebook execution in clean kernels while keeping source notebooks
free of large outputs:

```bash
uv run python scripts/verify_notebooks.py
```

Executed copies and the execution summary stay under `reports/runs/notebook_checks_*`.
This requires the original local M1 snapshots as well as the market cache.

## Scheduling

A manual `.github/workflows/research.yml` workflow runs tests and the pipeline.
It becomes available only if you choose to push this repository; no remote
workflow or recurring schedule has been activated. It does not upload market data
or reports. Network/provider failures correctly fail the job.

For local scheduling, first confirm a manual run succeeds. A possible crontab
entry, **not installed by this project**, is:

```cron
0 19 3 * * cd /absolute/path/to/project && /absolute/path/to/uv run python -m uranium_signals.run --config config/project.yaml >> /absolute/path/to/project/reports/scheduled.log 2>&1
```

The fixed end date must be deliberately advanced for a new analysis; scheduling
an unchanged configuration reproduces the same research window. WSL must be
running and cron enabled. Historical evaluations and new signals must remain
clearly distinguished when changing dates.

## Limitations and data use

This is a current-constituent historical study with survivorship, membership and
issuer-selection bias. US listings and exposure groups limit generalization;
utilities and equipment firms have different uranium sensitivity from miners.
Latest-revised macro data and proxy lags are an information-set limitation.
The final holdout has now been inspected; changing settings after this run needs
a fresh future evaluation period, not another claim of untouched performance.

Yahoo/yfinance is intended for personal research and educational use; the
software license does not license market data. PURANUSDM carries IMF copyright;
review [FRED terms](https://fred.stlouisfed.org/legal/) and the linked IMF terms
before redistribution. CPIAUCNS is tagged public domain with citation requested.
Sponsor CSV access does not establish a redistribution license. No raw data,
derived public display, or automatic publication is cleared by this workflow.

A carefully evaluated strategy that underperforms is a valid result. Neither the
classifier, backtest nor long-horizon outcomes guarantee future returns.
