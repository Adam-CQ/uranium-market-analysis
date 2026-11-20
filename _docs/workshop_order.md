# Workshop order — Uranium Market Signals Lab

Follow this guide one bounded step at a time. Each step should explain the reasoning, provide runnable notebook cells or commands, and finish with a verification checkpoint. Record a short result before moving to the next milestone.

The [project specification](uranium_market_predictions_backtesting_spec.md) defines scope and analytical requirements. This guide translates it into a learning sequence. All seven notebooks and the CLI below are implemented; the checkpoints remain prompts for your own review and learning.

## Current position

As of November 20, 2026:

- M1–M7 have an implemented first version, and all seven notebooks have passed execution in clean kernels.
- URNM, URNJ and NUKZ snapshots dated November 19 support the initial 16-candidate USD universe. URA, HURA and NLR candidate sources are omitted with documented access limitations; URA remains the price benchmark.
- The [source review](../data/reference/m1_source_review.json) and [universe](../data/reference/universe.csv) record those decisions. Yahoo/yfinance prices and official FRED uranium/CPI downloads are cached locally.
- The approved dependencies are installed with `uv` and pinned in `uv.lock`. Financial timing/calculation tests and the full CLI run have passed.
- No recurring schedule, remote workflow, publication, commit or push has been activated. A manual workflow template and local scheduling instructions are supplied.

### First-run results and review notes

| Milestone | Verified result | Remaining interpretation or limitation |
|---|---|---|
| M1 | Three dated holdings inventories, 16 USD candidates, and 22 observed ETF memberships. | Fixed current constituents and partial sponsor coverage create selection/membership bias. |
| M2 | 72,196 daily observations across 19 price series; all meet the minimum total-history rule. | Listing starts differ; predecessor histories and extreme returns need interpretation. October 2025 CPI is missing and preserved as missing. |
| M3 | 2,834 monthly candidate rows with trailing indicators, eligibility and lagged macro inputs. | Warm-up rows are retained; revised macro history plus a proxy lag is not a verified vintage information set. |
| M4 | Majority, momentum, Logistic Regression and Random Forest evaluated chronologically; Random Forest selected using validation balanced accuracy. | Holdout has now been inspected. Do not retune and describe another run on the same period as untouched. |
| M5 | 33 completed holdout trading periods, with next-session opens, weight drift, costs and common-interval benchmarks. | Label and execution intervals differ; current-constituent bias limits performance conclusions. |
| M6 | Rolling 36/60-month nominal/real outcomes, benchmarks, drawdowns and sample counts. | The fixed basket has only two non-overlapping five-year periods; monthly drawdown misses intramonth losses. |
| M7 | Full CLI, offline reproduction, local reports/metadata, clean-kernel notebook execution, and manual automation template. | Scheduling is not activated; data/display redistribution remains uncleared. |

Open the latest successful HTML report using the path in `reports/latest_run.txt`. It contains numerical results and limitations; `metadata.json` records provenance and checksums. Notebook execution evidence is saved locally under `reports/runs/notebook_checks_*`.

## Preparation — Environment and notebook use

1. Read `AGENTS.md` and the specification to understand the project scope and working rules.
2. Check the existing Python, `uv`, and notebook environment. Ask before installing dependencies; declare approved dependencies in `pyproject.toml` and use `uv` to manage them.
3. Open each notebook in order and run its cells from the beginning. Do not rely on variables left over from earlier interactive sessions.
4. Keep raw provider downloads local when redistribution is not cleared. Preserve source URLs, retrieval timestamps, effective dates, and checksums.

**Checkpoint:** the first notebook runs in the available environment, and the distinction between raw inputs, reference metadata, processed data, and reports is clear. Jupyter is installed in the approved environment, and all notebooks have executed in clean kernels.

## M1 — Market question and sources

Notebook: [01_market_question_and_sources.ipynb](../notebooks/01_market_question_and_sources.ipynb)

1. **Define the study.** Write the hypothesis and primary one-month target: a stock's future total return exceeds URA's. Separate this prediction exercise from the descriptive 3- and 5-year study.
2. **Review and archive sources.** Inspect sponsor pages for URA, URNM, HURA, URNJ, NLR, and NUKZ. Save dated holdings snapshots and source metadata. Record access failures and partial lists explicitly; do not treat a top-ten list as a complete portfolio.
3. **Normalize security identities.** Record ticker, exchange, company, share class, currency, and identifiers. Distinguish securities from issuers, resolve duplicate listings, and preserve the many-to-many security-to-ETF mapping.
4. **Classify exposure.** Separate producers, developers/explorers, physical uranium vehicles, utilities, equipment/services, and other related businesses. Document ambiguous classifications and exclusions.
5. **Choose data routes.** Review adjusted price coverage, corporate-action handling, access limits, cost, and terms. Select uranium benchmark, CPI, and broad equity benchmark inputs. The first version uses CPIAUCNS and Yahoo/yfinance; see the source register for terms and limitations.
6. **Define eligibility rules.** Set minimum history, liquidity, currency, and data-quality requirements before examining strategy performance. Form a provisional candidate list; finalize roughly 10–25 securities after the M2 availability audit. Define the long-run basket and weighting rule before examining its performance.

**Output:** research hypothesis, dated holdings inventory, source register, provisional candidate universe, and inclusion/exclusion rules.

**Checkpoint:** every candidate can be traced to a source and listing. Explain current-constituent bias, the difference between observation and availability dates, and why ETF membership does not imply equal uranium exposure. Document unresolved sources before proceeding.

## M2 — Data acquisition and quality

Notebook: `notebooks/02_download_and_validate_data.ipynb`

1. **Download a small sample.** Verify the chosen provider with Cameco and URA before expanding to the candidate universe. Preserve original responses and retrieval metadata.
2. **Collect the inputs.** Obtain adjusted equity/ETF prices, monthly uranium benchmark, CPI, and broad-market benchmark data. Respect provider limits and cache permitted downloads.
3. **Validate identity and schema.** Check listing, currency, dates, numeric fields, duplicate security-date rows, and corporate-action conventions.
4. **Audit history and missingness.** Inspect inception dates, suspensions, calendar differences, gaps, splits, distributions, and liquidity. Do not fill missing prices blindly.
5. **Finalize eligibility.** Apply the M1 rules, record exclusions, approve the research universe, and identify usable periods for each analysis.
6. **Save clean data and a quality report.** Keep raw inputs intact and explain every material cleaning decision.

**Output:** raw and clean datasets, approved universe, and data-quality report.

**Checkpoint:** unique security-date keys, explicit currency handling, documented missingness, and sufficient benchmark-aligned history. State whether historical membership is available or the study uses current constituents.

## M3 — Transformations and indicators

Notebook: `notebooks/03_transformations_and_indicators.ipynb`

1. **Calculate returns explicitly.** Teach simple and log returns, compounding, and weekly/monthly resampling using adjusted prices.
2. **Build technical indicators.** Calculate lagged returns, moving-average distances, volatility, downside volatility, drawdown, and volume/liquidity measures.
3. **Add relative indicators.** Calculate benchmark-relative strength and cross-sectional ranks using the eligible observations at each date.
4. **Transform macro inputs.** Calculate monthly uranium-price changes and real prices using CPI and a stated base period.
5. **Align information in time.** Join monthly data by availability date, with documented conservative lags where needed. Preserve missingness and avoid backfilling future information.
6. **Inspect features.** Plot distributions and relationships, investigate outliers, and save a documented monthly feature table.

**Output:** feature table, transformation explanations, and diagnostic plots.

**Checkpoint:** hand-check selected calculations and confirm that each feature uses only information available when the signal is formed.

## M4 — Prediction and chronological evaluation

Notebook: `notebooks/04_modeling_and_walk_forward.ipynb`

1. **Construct labels.** Calculate next-month stock and URA total returns. Keep unavailable future labels missing and separate label columns from predictors.
2. **Design chronological splits.** Define walk-forward training/validation periods and a final untouched holdout. Keep all stocks from a signal date in the same split and ensure training labels are known by the fitting date.
3. **Establish baselines.** Evaluate majority-class predictions and a pre-defined momentum rule.
4. **Fit interpretable models first.** Start with Logistic Regression, then Random Forest. XGBoost is optional and requires approval before adding a dependency.
5. **Evaluate and choose.** Compare balanced accuracy, precision/recall, ROC AUC, PR AUC where appropriate, confusion matrices, and calibration. Fit imputation, scaling, feature selection, and thresholds inside training periods only.
6. **Freeze the selected workflow.** Use validation results to select the model and settings; retain dated out-of-sample probabilities for the strategy notebook.

**Output:** model comparison, out-of-sample predictions, and a final time-based holdout result once model and strategy rules are frozen.

**Checkpoint:** no shuffled splits or future leakage. Keep the final holdout sealed during M5 strategy development as well; evaluate the frozen model and strategy together without retuning on that result.

## M5 — Trading strategy and simulation

Notebook: `notebooks/05_strategy_and_simulation.ipynb`

1. **Specify portfolio rules.** Rank eligible stocks monthly and hold the top 3–5 equally weighted, with a pre-defined probability threshold and cash rule if used.
2. **Implement realistic timing.** Execute no earlier than the next tradable session after signal formation. Align realized portfolio returns with the actual holding interval.
3. **Account for turnover and costs.** Apply configurable transaction costs and document rebalancing, unavailable prices, and cash treatment.
4. **Run comparisons.** Compare the ML strategy with a separate hand-written momentum strategy, equal-weight buy-and-hold, URA, and optionally URNM.
5. **Analyze outcomes.** Report cumulative and annualized returns, drawdowns, volatility, Sharpe assumptions, turnover, trades, and weak periods. Show absolute returns alongside relative performance.
6. **Complete the frozen holdout evaluation.** Report the final result and limitations without adjusting rules to improve that result.

**Output:** reproducible monthly backtest, benchmark comparison, and limitations discussion.

**Checkpoint:** inspect individual rebalance examples, verify transaction-cost calculations, and confirm that strategy choices did not use future returns or holdout results.

## M6 — Long-run investment effectiveness

Notebook: `notebooks/06_long_horizon_returns.ipynb`

1. **Confirm the study design.** Use Cameco and the previously defined basket, with explicit constituents, weighting, rebalancing, currency, and inception constraints.
2. **Calculate holding-period outcomes.** Compute rolling 36- and 60-month nominal total returns and CAGR from monthly investment dates.
3. **Calculate real outcomes.** Deflate returns with the stated CPI series and compare with URA, a broad-market benchmark, and cash/bills where practical.
4. **Measure risk within each window.** Calculate maximum drawdown experienced during each holding period and the frequency of positive real returns.
5. **Explain sample limitations.** Show usable rolling windows and distinct non-overlapping periods. Discuss current-constituent bias and the small number of market cycles.

**Output:** long-horizon outcome report with nominal/real returns, benchmark differences, drawdowns, and sample-size caveats.

**Checkpoint:** hand-check a holding period, distinguish total return from CAGR, and explain why overlapping windows cannot be randomly split as independent observations. Keep this track primarily descriptive.

## M7 — Automation and reproducibility

Notebook: `notebooks/07_automation_prep.ipynb`

1. **Extract proven logic.** Move the selected acquisition, validation, features, targets, modeling, backtest, and reporting calculations into small modules under `src/uranium_signals/`.
2. **Add configuration.** Make the approved universe, dates, lags, costs, model settings, and file paths explicit and validate them at startup.
3. **Create the CLI.** Connect the workflow through one entry point, with caching, bounded retries, logging, timestamps, and run metadata.
4. **Produce reports.** Save latest signals and monthly/long-horizon evaluation outputs in compact HTML, CSV, and PNG formats as appropriate. Distinguish live signals from historical evaluations.
5. **Verify reproducibility and failure behavior.** Compare module results with notebook examples, run appropriate checks, and ensure material data/model failures produce a non-zero exit status.
6. **Document and schedule.** Write setup and run instructions, data terms, and limitations. Configure local or GitHub Actions scheduling after a manual end-to-end run succeeds.

Final command:

```bash
uv run python -m uranium_signals.run --config config/project.yaml
```

**Output:** documented CLI workflow, timestamped signals, both evaluation reports, and run metadata.

**Checkpoint:** a reviewer can follow setup instructions and reproduce the selected workflow from documented permitted inputs. A profitable result is not required; sound analysis and honest reporting are.
