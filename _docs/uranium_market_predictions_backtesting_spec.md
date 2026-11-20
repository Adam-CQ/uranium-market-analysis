# Uranium Market Predictions and Backtesting Workshop

## 1. Project summary

**Working title:** Uranium Market Signals Lab

**Purpose:** Build a reproducible research workflow that collects historical uranium-related market data, teaches the participant how to clean and transform it, creates financial indicators, trains prediction models, and tests a simple trading strategy against relevant benchmarks.

The project is an educational analytics system, not an investment recommendation product. It will contain two related research tracks:

1. **Course-aligned tactical track:** Can public uranium-market and equity-market signals help identify uranium-related companies likely to outperform a relevant benchmark over the next one to three months?
2. **Long-horizon investment track:** How would a disciplined, long-run investment in a uranium-related company or basket have performed over rolling 3- and 5-year holding periods, and how does that compare with relevant benchmarks and inflation?

The second track tests historical investment outcomes. It does not claim to estimate intrinsic value or predict a precise future share price. A DCF-based fair-value estimate remains outside this project unless added later as a separately sourced extension.

The project should make each analysis explainable and repeatable. The main deliverable is a set of milestone notebooks that teach the reasoning and transformations, plus a small Python command-line workflow that runs the selected final pipeline.

## 2. Scope and course fit

The project follows the course progression:

1. Identify data sources and define the investable universe.
2. Clean, align, explore, and transform price and market data.
3. Form a prediction target, establish baselines, train and evaluate models.
4. Convert predictions into a trading rule and simulate historical results.
5. Move the selected workflow into Python modules and automate a repeatable run.

The monthly prediction and trading simulation remain the course-aligned core. The 3–5 year track is an additional historical investment-effectiveness analysis: it compares rolling holding-period outcomes for Cameco and a clearly defined uranium-related basket with benchmarks, including inflation-adjusted returns. Precise 3–5 year price targets, DCF valuation, real-time feeds, futures execution, and broker integration remain outside the first version.

## 3. Investment universe

Use holdings from the following ETFs as candidate sources:

- Global X Uranium ETF (URA)
- Sprott Uranium Miners ETF (URNM)
- Global X Uranium Index ETF (HURA)
- Sprott Junior Uranium Miners ETF (URNJ)
- VanEck Uranium and Nuclear Energy ETF (NLR)
- Range Nuclear Renaissance Index ETF (NUKZ)

These funds have different mandates. URNM and URNJ emphasize miners (with URNJ focused on junior and smaller companies); URA and HURA provide uranium and nuclear-related equity exposure; NLR and NUKZ include broader nuclear value-chain exposure such as utilities, engineering, components, and reactor companies. Therefore, ETF membership is a candidate-discovery mechanism, not proof that all constituents have the same uranium-price sensitivity.

### Universe procedure

1. Download or record holdings from the fund sponsor pages, with retrieval date and source URL.
2. Normalize tickers, exchange, company name, share class, and security currency.
3. Deduplicate the union of holdings and retain a many-to-many `ticker_to_etf` mapping.
4. Assign an exposure group: `uranium_producer`, `uranium_developer_explorer`, `physical_uranium`, `nuclear_utility`, `nuclear_equipment_services`, or `other_related`.
5. Review the universe manually and document exclusions, such as insufficient price history, illiquidity, unavailable adjusted prices, or ticker ambiguity.
6. Keep an archived dated holdings snapshot. Do not assume current constituents were ETF members throughout historical backtests.

**Initial practical universe:** about 10–25 liquid securities with sufficient aligned history, plus URA and optionally URNM as investable sector benchmarks. Cameco (CCJ/CCO, depending on listing) is the featured case study, not the only model observation. The final ticker list should be selected after a data-availability audit rather than hard-coded in advance.

For an initial backtest, a fixed research universe is acceptable if clearly disclosed. A stronger later version uses historical constituent snapshots to reduce survivorship and membership bias.

## 4. Research question and prediction target

### Primary target

At each month-end, predict whether each eligible company will outperform URA over the next 1 month. Define the label using future total return:

`outperform_1m = 1 if stock_total_return_next_month > URA_total_return_next_month else 0`

Use adjusted prices where available so splits and distributions are treated consistently. Establish a second horizon (3 months) as an extension, not a separate project goal.

### Why relative performance

The target asks whether signals help rank uranium-related equities against an investable sector benchmark. Also report absolute-return results so benchmark-relative outperformance does not hide losses.

### Long-horizon track: 3- and 5-year investment effectiveness

For Cameco and a pre-defined uranium-related basket, calculate rolling 3-year and 5-year holding-period **total returns** from monthly investment dates. Report both nominal and inflation-adjusted returns, and compare them with URA, a broad equity benchmark, and (where practical) cash or short-term government-bill returns. Include annualized return/CAGR, the share of rolling periods with positive real returns, benchmark-relative return, and maximum drawdown experienced during each holding period. State the currency and CPI series used.

This is a historical outcome study, not a guarantee that the same return pattern will recur. Use adjusted/total-return series where available and document limitations. For the basket, define the constituents and weighting rule before examining performance (for example, equal-weight at a stated annual rebalance). A basket based on current ETF holdings must be labeled as a current-constituent historical study; it is not a point-in-time investable backtest unless historical membership data is available.

#### Important sample-size and validation caveat

Daily observations do not create thousands of independent 3- or 5-year outcomes. Long-horizon returns overlap heavily when calculated monthly, and uranium equities have experienced a limited number of distinct market cycles. Present the number of distinct/non-overlapping holding periods alongside the rolling chart. Do not randomly split overlapping windows into train and test sets. Treat this track primarily as descriptive historical analysis; if a model is attempted, use time-ordered evaluation with a gap/embargo at least as long as the forecast horizon and report the very small effective sample size. Do not tune a model on the same few cycles used to claim success.

This makes the long-run track a useful check on the investment thesis while keeping the more statistically tractable monthly model as the main machine-learning exercise.

### Baselines

Compare model results against:

- always predict the majority class / historical base rate;
- a simple momentum rule using past 3- or 6-month return;
- equal-weight buy-and-hold of the eligible universe;
- buy-and-hold URA (and optionally URNM).

## 5. Data plan

### Core data (first version)

| Data | Frequency | Initial role | Notes |
|---|---|---|---|
| Adjusted OHLCV for company equities | Daily | Price returns, trend, volatility, liquidity, technical indicators | Validate ticker, exchange, currency, corporate actions, missing dates, and provider terms. |
| URA adjusted prices | Daily | Main relative-return benchmark and sector context | ETF price history is an investable benchmark, not a uranium spot-price series. |
| URNM adjusted prices | Daily | Optional miner-focused benchmark | Helps distinguish miner basket performance from broad nuclear exposure. |
| Uranium spot benchmark | Monthly | Uranium-cycle context and lagged predictor | Public historical monthly benchmarks are more practical than daily spot quotes. Record the observation date and when it became available. |
| Inflation index (e.g. US CPI) | Monthly | Convert uranium benchmark to real dollars for analysis | Use a consistent base date; do not compare commodity dollars directly with equity share prices. |
| Broad equity benchmark (e.g. S&P 500 ETF) | Daily | Market-regime control / context | Optional in the first notebook; useful for separating uranium-specific from broad market moves. |

### Long-horizon return inputs

For the 3- and 5-year track, use monthly adjusted/total-return prices for Cameco, the selected basket constituents, URA, and a broad-market benchmark. Use a monthly CPI series to calculate real returns. Align each series to a common currency or report currency-specific results separately; do not silently mix CAD and USD returns. Account for ETF and equity inception dates and show the number of usable holding windows.

### Extensions (only after the core pipeline works)

- Long-term uranium price series, kept distinct from spot. Long-term contract pricing is not equivalent to exchange-traded futures.
- Historical futures series only if a consistent, legally usable source is identified and contract rolls can be handled correctly.
- Quarterly company financials (revenue, EBITDA, free cash flow, debt, shares outstanding) as dated, lagged features. This requires careful filing-date handling to avoid look-ahead bias.
- Historical ETF holdings constituents and weights to reconstruct point-in-time universes.

### Source and licensing controls

For every dataset, retain provider name, source URL, retrieval timestamp, observation/effective date, units, currency, frequency, and license/terms notes. Some uranium price assessments are proprietary or restricted. Use publicly accessible series only where their terms permit the intended project and clearly distinguish a public benchmark from a live tradable quote. Cameco's price page can be used as a reference for understanding spot versus long-term price context; check its current terms before redistributing data or publishing derived displays.

## 6. Data transformations and financial indicators

The notebooks should teach the calculations, not hide them behind a library call. At minimum, explain and implement:

### Price and return transformations

- Sort by security and date; enforce one row per security-date.
- Align calendars and currencies; inspect non-trading days rather than filling them blindly.
- Daily simple return: `P_t / P_(t-1) - 1`.
- Daily log return: `log(P_t / P_(t-1))`.
- Weekly/monthly resampling and cumulative returns.
- Forward target returns using future prices, then shift labels so features at time `t` only predict outcomes after `t`.
- Total-return handling through adjusted prices; explain limits of vendor-adjusted data.

### Candidate equity indicators

- Lagged returns: 1, 5, 21, 63, and 126 trading-day windows.
- Moving-average distances and crossovers (e.g. 20/50/200-day).
- Rolling volatility and downside volatility.
- Drawdown from rolling peak.
- Volume change and rolling average volume; turnover/liquidity filter where data allows.
- Relative strength versus URA, URNM, and broad-market benchmark.
- Cross-sectional rank of recent returns across the eligible universe.

### Uranium and inflation transformations

- Spot price percentage/log changes and rolling change over 1, 3, and 6 months.
- Real uranium price: nominal monthly price divided by CPI relative to a stated base period.
- Spot-versus-long-term spread only when both series are public, consistent, and temporally available.
- Carry the release/availability date. For a monthly predictor, use it only after the observation could have been known (a conservative one-month lag is a suitable first assumption).

### Company financial indicator extensions

- Margins, year-over-year revenue growth, EBITDA/FCF measures, leverage, and valuation ratios only if historical point-in-time data is available.
- Join by publication/filing date, not merely fiscal period end. Apply a reporting lag and preserve missingness. Do not backfill future-reported values into earlier rows.

## 7. Modeling and validation

Start with interpretable baselines before tree models.

1. Majority-class baseline.
2. Momentum threshold / simple rule.
3. Logistic regression as a basic statistical model.
4. Random Forest.
5. XGBoost (optional if installation and dependencies are manageable).

Use time-ordered evaluation, such as expanding-window or rolling walk-forward splits. Never randomly shuffle observations across time. All transformations that learn parameters (imputation, scaling, feature selection, threshold selection) must be fit using only the training period. Keep a final untouched test period for a final report.

Report both predictive and economic measures:

- balanced accuracy, precision/recall, ROC AUC, and PR AUC where class imbalance warrants it;
- confusion matrix and calibration or probability-bin check;
- strategy cumulative return, annualized return, maximum drawdown, volatility, Sharpe ratio (with stated assumptions), turnover, and number of trades;
- results versus URA buy-and-hold and simple momentum after assumed transaction costs.

Interpret results cautiously: a good classifier need not produce a profitable strategy, and a backtest is not evidence of future profitability.

## 8. Trading simulation

Keep the initial strategy deliberately simple and reproducible:

- At each monthly rebalance date, rank stocks by predicted probability of outperforming URA.
- Hold the top 3–5 eligible stocks equally weighted, or hold cash when no probability clears a pre-defined threshold.
- Apply a configurable transaction-cost assumption and execute no earlier than the next tradable session after the signal is formed.
- Compare with URA, URNM, equal-weight universe, and the momentum baseline.
- Include a long-only version first; shorting and leverage are out of scope.

A separate hand-written signal strategy should be retained as a comparison to the ML model, consistent with the course approach.

## 9. Workshop milestones and notebook deliverables

Each milestone is a bounded learning unit with a notebook, a short written result, and a checkpoint before moving on.

| Milestone | Notebook | Learning outcome | Main deliverable |
|---|---|---|---|
| M1 — Market question and sources | `notebooks/01_market_question_and_sources.ipynb` | Turn the uranium investing idea into a measurable prediction question; understand ETF mandates, ticker overlap, data permissions, and data dictionary design. | Dated ETF holdings inventory, candidate ticker universe, source register, and research hypothesis. |
| M2 — Data acquisition and quality | `notebooks/02_download_and_validate_data.ipynb` | Fetch price, benchmark, uranium-price, and CPI data; inspect missingness, duplicate dates, currencies, splits, ticker history, and frequency. | Raw/clean data files and a documented data-quality report. |
| M3 — DataFrame analysis and feature engineering | `notebooks/03_transformations_and_indicators.ipynb` | Calculate returns, align monthly/daily observations, build technical indicators, real uranium prices, and lagged features; visualize distributions and relationships. | Feature table plus plots and explanations for each transformation. |
| M4 — Prediction and evaluation | `notebooks/04_modeling_and_walk_forward.ipynb` | Define labels, compare baselines, fit Logistic Regression/Random Forest/XGBoost, use chronological validation, and inspect feature importance. | Model comparison with a final time-based holdout result. |
| M5 — Trading strategy and backtest | `notebooks/05_strategy_and_simulation.ipynb` | Turn probabilities into portfolio rules, account for timing and trading costs, compare benchmarks, analyze drawdowns and failure periods. | Reproducible monthly strategy backtest and limitations discussion. |
| M6 — Long-run investment effectiveness | `notebooks/06_long_horizon_returns.ipynb` | Calculate rolling 3- and 5-year nominal and real total returns for Cameco and the pre-defined basket; compare with URA and broad-market benchmarks; understand overlapping-window and small-cycle limitations. | Long-horizon outcome report with CAGR, positive-real-return frequency, benchmark comparison, drawdowns, and effective sample-size caveats. |
| M7 — Automation and reproducibility | `notebooks/07_automation_prep.ipynb` plus Python package/script | Refactor the selected monthly pipeline and long-horizon report calculations, add configuration and logging, persist outputs, run from CLI, and schedule locally or through GitHub Actions. | Final CLI run that updates data where available, generates latest signals and both evaluation reports, and records run metadata. |

The user prefers a workshop approach: each milestone should be taught one bounded step at a time, with runnable commands, explanation of why each step is used, and a verification checkpoint before proceeding.

## 10. Final automation design

The final deliverable should be a small Python workflow, not a notebook-only handoff. Notebooks remain the instructional and exploratory record; reusable logic moves into importable modules.

Suggested command:

`python -m uranium_signals.run --config config/project.yaml`

The automation should:

1. Read configuration and the approved ticker universe.
2. Fetch or update raw data (with retry, rate-limit awareness, and a local cache).
3. Validate schema, freshness, missingness, and minimum history.
4. Build features using only historically available information.
5. Load the selected model or retrain according to a documented schedule.
6. Generate predictions as of the latest valid date.
7. Apply the fixed strategy rules and produce a signal table.
8. Save data and outputs with timestamps and run metadata.
9. Write a compact HTML/CSV/PNG report and clear logs.
10. Exit with a non-zero status on material data or model errors.

A single `.py` file can serve as the final entry point, but the project should avoid putting all fetching, transformations, model logic, and backtesting into one monolithic script. Use a small package of Python modules and one `run.py`/CLI entry point. This makes the automation maintainable while keeping the visible final command simple.

## 11. Suggested repository layout

```text
uranium-market-signals/
├── README.md
├── pyproject.toml
├── uv.lock
├── config/
│   └── project.yaml
├── data/
│   ├── raw/                 # ignored by git unless redistribution is permitted
│   ├── processed/
│   └── reference/           # dated holdings/source metadata
├── notebooks/
│   ├── 01_market_question_and_sources.ipynb
│   ├── 02_download_and_validate_data.ipynb
│   ├── 03_transformations_and_indicators.ipynb
│   ├── 04_modeling_and_walk_forward.ipynb
│   ├── 05_strategy_and_simulation.ipynb
│   ├── 06_long_horizon_returns.ipynb
│   └── 07_automation_prep.ipynb
├── src/uranium_signals/
│   ├── config.py
│   ├── universe.py
│   ├── providers.py
│   ├── data_quality.py
│   ├── features.py
│   ├── targets.py
│   ├── modeling.py
│   ├── backtest.py
│   ├── reporting.py
│   └── run.py
├── reports/
├── tests/
└── .github/workflows/
    └── scheduled-run.yml
```

The project can start with notebooks and a minimal Python package. Do not add a database, API, dashboard, or complex cloud deployment unless the core analytical pipeline works and there is a clear need.

## 12. Key risks and decisions to resolve during M1–M2

- ETF holdings change; current constituents create survivorship bias if projected backward.
- ETFs differ in their exposure: a nuclear utility or reactor-equipment maker may not respond to uranium prices like a miner.
- Uranium spot data is generally less frequent and less transparent than listed equity data; public history may be monthly.
- Futures and long-term contract pricing are distinct data concepts and should not be combined under a generic “uranium futures” label.
- International listings introduce currency, timezone, holiday-calendar, and ticker-mapping issues.
- Junior miners may have short histories, low liquidity, suspensions, or substantial dilution; missingness and investability need explicit treatment.
- Monthly targets produce a smaller sample than daily targets, while daily targets may overstate signal availability from monthly uranium data. The first version should choose one primary rebalance frequency and keep it consistent.
- Data terms may limit storage, redistribution, or public display. The README must document sources and permitted use.
- Feature importance is descriptive, not causal. Avoid claiming that uranium prices cause an equity move based only on model explanations.

## 13. Success criteria

The project is successful when a reviewer can:

- understand the question and why the chosen universe is relevant;
- reproduce the data preparation from documented sources or included permitted data;
- inspect how raw data becomes financial indicators and prediction labels;
- verify chronological model validation and absence of obvious leakage;
- compare ML results with simple rules and sector benchmarks;
- run the documented Python command to generate predictions and a backtest/report;
- see the limitations, data terms, and assumptions clearly stated;
- inspect 3- and 5-year rolling real-return comparisons with suitable benchmarks and an honest account of the limited number of independent uranium market cycles.

Profitability is not a success criterion. A carefully evaluated strategy that fails to beat its benchmark can still be a strong project if the analysis is sound and reproducible.

## 14. Research references

- Independent Trader, “Jak inwestować w uran w 2026 roku? Wszystko, co musisz wiedzieć” (article supplied by the user; updated July 2026): https://independenttrader.pl/jak-inwestowac-w-uran-wszystko-co-musisz-wiedziec/
- Global X URA: https://www.globalxetfs.com/funds/ura
- Sprott URNM: https://sprottetfs.com/urnm-sprott-uranium-miners-etf
- Global X Canada HURA: https://www.globalx.ca/product/HURA
- Sprott URNJ: https://sprottetfs.com/urnj-sprott-junior-uranium-miners-etf
- VanEck NLR: https://www.vaneck.com/us/en/investments/uranium-nuclear-energy-etf-nlr/holdings/
- Range NUKZ: https://www.rangeetfs.com/nukz
- FRED IMF uranium price series: https://fred.stlouisfed.org/series/PURANUSDM
- Cameco uranium price reference: https://www.cameco.com/invest/markets/uranium-price
