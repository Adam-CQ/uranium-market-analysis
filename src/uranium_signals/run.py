"""Run the same analytical functions used in the milestone notebooks."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from .backtest import buy_hold_path, performance, portfolio_path
from .config import load_config
from .data_quality import validate_macro, validate_prices
from .features import FEATURE_COLUMNS, monthly_features
from .long_horizon import analyze_long_horizon
from .modeling import classification_metrics, latest_signals, select_model, walk_forward
from .providers import acquire
from .reporting import LIMITATIONS, write_reports
from .targets import build_targets
from .universe import load_universe

LOG = logging.getLogger(__name__)


def research_tables(cfg, prices, macro, universe):
    tickers = set(universe.ticker) | {cfg['benchmark'], cfg['broad_benchmark']} | set(cfg['optional_benchmarks'])
    prices, quality = validate_prices(prices, cfg, tickers)
    validate_macro(macro)
    features = monthly_features(prices, macro, cfg, set(universe.ticker))
    targets = build_targets(prices, cfg)
    table = features.merge(targets, on=['date', 'ticker'], validate='one_to_one')
    if table[FEATURE_COLUMNS].isna().all().any():
        raise ValueError('An entire feature is missing; inspect source coverage')
    # Missing latest macro inputs should fail instead of silently imputing stale information.
    latest = table[table.date.eq(table.date.max())]
    if latest.macro_observation_date.isna().any() or (latest.date - latest.macro_observation_date).dt.days.max() > 180:
        raise ValueError('Macro inputs missing or stale at latest signal')
    return prices, quality, targets, table


def evaluate(cfg, table, targets):
    predictions, audits = walk_forward(table, cfg)
    metrics, calibration = classification_metrics(predictions)
    selected = select_model(metrics)
    holdout = predictions[(predictions.model == selected) & (predictions.split == 'holdout')]
    # Only complete executable holding periods enter the economic report.
    model_path = portfolio_path(holdout, targets, cfg)
    dates = list(model_path.date)
    momentum_candidates = table[table.eligible & table.date.isin(dates)]
    momentum_path = portfolio_path(momentum_candidates, targets, cfg, kind='momentum')
    returns = pd.DataFrame({'date': dates, 'model': model_path['return'].to_numpy(),
                            'momentum': momentum_path['return'].to_numpy()})
    baseline_members = {}
    initial = table[table.date.eq(dates[0]) & table.eligible].ticker.tolist()
    for name, tickers in [('equal_weight_buy_hold', initial), (cfg['benchmark'], [cfg['benchmark']]),
                          (cfg['broad_benchmark'], [cfg['broad_benchmark']])]:
        series, members = buy_hold_path(targets, dates, tickers, cfg)
        returns[name] = series.to_numpy()
        baseline_members[name] = members
    for benchmark in cfg['optional_benchmarks']:
        # Optional funds can have shorter histories; include only if present for every interval.
        subset = targets[targets.ticker.eq(benchmark) & targets.date.isin(dates)]
        if len(subset) == len(dates) and subset.trade_return.notna().all():
            returns[benchmark] = buy_hold_path(targets, dates, [benchmark], cfg)[0].to_numpy()
    economics = []
    for name in returns.columns.drop('date'):
        row = {'strategy': name, **performance(returns[name], cfg['risk_free_annual'])}
        if name in ('model', 'momentum'):
            path = model_path if name == 'model' else momentum_path
            row.update(total_trades=int(path.trades.sum()), average_one_way_turnover=path.one_way_turnover.mean(),
                       total_cost_fraction=path.cost_fraction.sum())
        economics.append(row)
    return predictions, audits, metrics, calibration, selected, returns, pd.DataFrame(economics), model_path, momentum_path, baseline_members


def run(config_path, offline=False, refresh=False):
    cfg = load_config(config_path)
    if offline and refresh:
        raise ValueError('--offline and --refresh are mutually exclusive')
    universe = load_universe(cfg)
    tickers = set(universe.ticker) | {cfg['benchmark'], cfg['broad_benchmark']} | set(cfg['optional_benchmarks'])
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = cfg['root'] / 'reports/runs' / run_id
    output.mkdir(parents=True)
    handler = logging.FileHandler(output / 'run.log')
    logging.getLogger().addHandler(handler)
    try:
        LOG.info('Run %s started', run_id)
        prices, macro, source_metadata = acquire(cfg, tickers, refresh=refresh, offline=offline)
        prices, quality, targets, table = research_tables(cfg, prices, macro, universe)
        LOG.info('Validated %s prices; %s monthly feature rows', len(prices), len(table))
        results = evaluate(cfg, table, targets)
        predictions, audits, metrics, calibration, selected, returns, economics, model_path, momentum_path, members = results
        outcomes, long_summary, levels = analyze_long_horizon(prices, macro, cfg)
        signals, model = latest_signals(table, cfg, selected)
        write_reports(output, metrics, calibration, returns, economics, outcomes, long_summary, signals, quality, cfg, selected)
        for name, data in [('features', table), ('predictions', predictions), ('fit_audit', audits),
                           ('model_trades', model_path), ('momentum_trades', momentum_path), ('long_horizon_levels', levels.reset_index())]:
            data.to_csv(output / f'{name}.csv', index=False)
        with (output / 'model.pkl').open('wb') as stream:
            pickle.dump(model, stream)
        config_bytes = Path(config_path).read_bytes()
        (output / 'project.yaml').write_bytes(config_bytes)
        metadata = {'run_id': run_id, 'status': 'success', 'selected_model': selected,
                    'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                    'universe_sha256': hashlib.sha256((cfg['root'] / cfg['universe']).read_bytes()).hexdigest(),
                    'source_metadata': source_metadata, 'features': FEATURE_COLUMNS,
                    'missing_macro_observations': macro.loc[macro.value.isna(), ['date', 'series']].astype(str).to_dict('records'),
                    'baseline_initial_members': members, 'limitations': LIMITATIONS,
                    'versions': {p: version(p) for p in ('pandas', 'numpy', 'scikit-learn', 'yfinance', 'matplotlib')},
                    'signal_as_of': str(signals.date.max().date()),
                    'latest_is_month_end_not_live': True,
                    'outputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name != 'run.log'}}
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        processed = cfg['root'] / 'data/processed'
        processed.mkdir(exist_ok=True)
        table.to_csv(processed / 'monthly_features.csv', index=False)
        (cfg['root'] / 'reports/latest_run.txt').write_text(str(output.relative_to(cfg['root'])) + '\n')
        LOG.info('Selected %s; report: %s', selected, output / 'report.html')
        return output
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}, indent=2) + '\n')
        LOG.exception('Research run failed')
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/project.yaml')
    parser.add_argument('--offline', action='store_true', help='Require an exact validated local cache')
    parser.add_argument('--refresh', action='store_true', help='Acquire a new provider snapshot; preserve older data')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        output = run(args.config, args.offline, args.refresh)
    except Exception:
        LOG.exception('Unable to complete research run')
        return 1
    print(output / 'report.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())
