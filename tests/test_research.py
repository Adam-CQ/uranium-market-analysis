"""Financial timing and calculation invariants, using explicitly synthetic inputs."""
import numpy as np
import pandas as pd
import pytest
from uranium_signals.backtest import performance, portfolio_path
from uranium_signals.config import load_config
from uranium_signals.features import daily_features, macro_features
from uranium_signals.long_horizon import basket_wealth
from uranium_signals.targets import build_targets
from uranium_signals.modeling import walk_forward
from uranium_signals.features import FEATURE_COLUMNS


@pytest.fixture
def cfg():
    return load_config('config/project.yaml')


def prices_from_values(dates, ticker, values, opens=None):
    return pd.DataFrame({'date': pd.to_datetime(dates), 'ticker': ticker, 'currency': 'USD',
                         'adj_close': values, 'adj_open': values if opens is None else opens,
                         'adj_high': np.array(values) * 1.1, 'adj_low': np.array(values) * .9,
                         'volume': 1000000})


def test_target_missing_is_not_negative_and_execution_is_after_signal(cfg):
    cfg = dict(cfg, end='2020-04-15')
    dates = ['2020-01-31', '2020-02-03', '2020-02-28', '2020-03-02', '2020-03-31', '2020-04-01']
    p = pd.concat([prices_from_values(dates, 'URA', [10, 11, 12, 13, 14, 15]),
                   prices_from_values(dates, 'CCJ', [20, 22, 28, 26, 30, 31])])
    t = build_targets(p, cfg)
    stock = t[t.ticker.eq('CCJ')].set_index('date')
    jan = stock.loc['2020-01-31']
    assert jan.outperform_1m == 1
    assert jan.forward_return == pytest.approx(.4)
    assert jan.trade_return == pytest.approx(26 / 22 - 1)
    assert jan.execution_date > pd.Timestamp('2020-01-31')
    assert pd.isna(stock.loc['2020-03-31', 'outperform_1m'])
    assert pd.isna(stock.loc['2020-03-31', 'trade_return'])


def test_future_price_cannot_change_past_features(cfg):
    dates = pd.bdate_range('2018-01-01', periods=400)
    p = prices_from_values(dates, 'CCJ', np.linspace(10, 30, 400))
    original = daily_features(p, cfg)
    changed = p.copy()
    changed.loc[350:, 'adj_close'] *= 10
    altered = daily_features(changed, cfg)
    pd.testing.assert_frame_equal(original.iloc[:350], altered.iloc[:350])


def test_macro_join_dates_are_after_observation_month(cfg):
    dates = pd.date_range('2010-01-01', periods=24, freq='MS')
    macro = pd.concat([pd.DataFrame({'date': dates, 'series': s, 'value': np.arange(24) + 100}) for s in ['PURANUSDM', 'CPIAUCNS']])
    m = macro_features(macro, cfg)
    assert (m.macro_available_at > m.macro_observation_date + pd.offsets.MonthEnd(0)).all()
    assert (m.macro_available_at - (m.macro_observation_date + pd.offsets.MonthEnd(0))).dt.days.eq(60).all()


def test_transaction_cost_charged_for_purchase_and_switch(cfg):
    dates = pd.to_datetime(['2020-01-31', '2020-02-28'])
    pred = pd.DataFrame({'date': dates, 'ticker': ['CCJ', 'DNN'], 'probability': [.9, .9]})
    targets = pd.DataFrame({'date': np.repeat(dates, 2), 'ticker': ['CCJ', 'DNN'] * 2,
                            'trade_return': [0., 0., 0., 0.],
                            'execution_date': np.repeat(pd.to_datetime(['2020-02-03', '2020-03-02']), 2),
                            'exit_date': np.repeat(pd.to_datetime(['2020-03-02', '2020-04-01']), 2)})
    path = portfolio_path(pred, targets, dict(cfg, top_n=1, transaction_cost_bps=10))
    assert list(path.traded_value) == [1, 2]
    assert path['return'].to_list() == pytest.approx([-.001, -.002])


def test_missing_held_execution_price_fails(cfg):
    date = pd.Timestamp('2020-01-31')
    pred = pd.DataFrame({'date': [date], 'ticker': ['CCJ'], 'probability': [.9]})
    targets = pd.DataFrame({'date': [date], 'ticker': ['CCJ'], 'trade_return': [np.nan],
                            'execution_date': [pd.Timestamp('2020-02-03')], 'exit_date': [pd.Timestamp('2020-03-02')]})
    with pytest.raises(ValueError, match='held security'):
        portfolio_path(pred, targets, cfg)


def test_drawdown_includes_initial_capital():
    metrics = performance([-.5, 1.])
    assert metrics['max_drawdown'] == -.5
    assert metrics['cumulative_return'] == 0


def test_annual_basket_weights_drift_before_rebalance():
    index = pd.period_range('2020-01', periods=4, freq='M')
    prices = pd.DataFrame({'A': [1, 2, 4, 8], 'B': [1, 1, 1, 1]}, index=index)
    wealth = basket_wealth(prices, ['A', 'B'])
    assert wealth.to_list() == pytest.approx([1, 1.5, 2.5, 4.5])


def test_training_label_end_precedes_fit_cutoff(cfg):
    dates = pd.date_range('2010-01-31', '2017-12-31', freq='ME')
    rows = []
    for i, date in enumerate(dates):
        for ticker in ('CCJ', 'DNN'):
            row = dict(date=date, ticker=ticker, eligible=True, label_end=date + pd.offsets.MonthEnd(1),
                       outperform_1m=float((i + (ticker == 'CCJ')) % 2), forward_return=.01,
                       benchmark_forward_return=0., trade_return=.01,
                       execution_date=date + pd.offsets.BDay(1), exit_date=date + pd.offsets.MonthEnd(1) + pd.offsets.BDay(1))
            row.update({f: i / 100 for f in FEATURE_COLUMNS})
            rows.append(row)
    predictions, audit = walk_forward(pd.DataFrame(rows), cfg)
    assert not predictions.empty
    assert (audit.latest_training_label_end < audit.fit_cutoff).all()
    assert set(predictions.model) == {'majority', 'logistic', 'random_forest', 'momentum'}


def test_long_horizon_real_return_and_missing_cpi_windows(cfg):
    from uranium_signals.long_horizon import analyze_long_horizon
    dates = pd.date_range('2010-01-31', periods=85, freq='ME')
    frame = pd.concat([prices_from_values(dates, ticker, 100 * 1.01 ** np.arange(85))
                       for ticker in ('CCJ', 'URA', 'SPY', 'DNN')], ignore_index=True)
    cpi = pd.DataFrame({'date': dates.to_period('M').to_timestamp(), 'series': 'CPIAUCNS',
                        'value': 100 * 1.002 ** np.arange(85)})
    cpi.loc[40, 'value'] = np.nan
    settings = dict(cfg, end='2017-02-01', basket=['CCJ', 'DNN'])
    outcomes, summary, levels = analyze_long_horizon(frame, cpi, settings)
    first = outcomes[(outcomes.ticker == 'CCJ') & (outcomes.horizon_months == 36)].iloc[0]
    assert first.nominal_return == pytest.approx(1.01 ** 36 - 1)
    assert first.real_return == pytest.approx((1.01 / 1.002) ** 36 - 1)
    assert first.monthly_max_drawdown == 0
    missing_month = str(dates[40].to_period('M'))
    assert not outcomes.start.eq(missing_month).any()
    assert not outcomes.end.eq(missing_month).any()
    assert (summary.nonoverlapping_windows <= summary.rolling_windows).all()


def test_drift_causes_turnover_even_for_unchanged_target_weights(cfg):
    dates = pd.to_datetime(['2020-01-31', '2020-02-28'])
    pred = pd.DataFrame({'date': np.repeat(dates, 2), 'ticker': ['CCJ', 'DNN'] * 2,
                         'probability': [.9] * 4})
    targets = pred[['date', 'ticker']].copy()
    targets['trade_return'] = [1., 0., 0., 0.]
    targets['execution_date'] = np.repeat(pd.to_datetime(['2020-02-03', '2020-03-02']), 2)
    targets['exit_date'] = np.repeat(pd.to_datetime(['2020-03-02', '2020-04-01']), 2)
    path = portfolio_path(pred, targets, dict(cfg, top_n=2))
    assert path.traded_value.iloc[1] == pytest.approx(1 / 3)


def test_quality_rejects_duplicate_security_dates(cfg):
    from uranium_signals.data_quality import validate_prices
    p = prices_from_values(['2026-11-19'], 'URA', [10])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_prices(pd.concat([p, p]), cfg, {'URA'})


def test_offline_cache_tampering_fails_before_loading(tmp_path, cfg):
    import json
    from uranium_signals.providers import acquire
    cache = tmp_path / 'data/raw/market' / f"{cfg['start']}_{cfg['end']}"
    cache.mkdir(parents=True)
    (cache / 'prices.csv').write_text('altered bytes')
    (cache / 'manifest.json').write_text(json.dumps({'tickers': ['URA'],
        'cache_path': str(cache.relative_to(tmp_path)), 'sha256': {'prices.csv': 'incorrect'}}))
    with pytest.raises(ValueError, match='checksum mismatch'):
        acquire(dict(cfg, root=tmp_path), {'URA'}, offline=True)


def test_cli_invalid_flag_combination_is_nonzero():
    import subprocess
    import sys
    completed = subprocess.run([sys.executable, '-m', 'uranium_signals.run', '--offline', '--refresh'],
                               capture_output=True, text=True)
    assert completed.returncode == 1
    assert 'mutually exclusive' in completed.stderr
