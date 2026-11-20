"""Fail on material input errors and report eligibility without filling gaps."""
import numpy as np
import pandas as pd


def validate_prices(prices, cfg, expected):
    required = {'date', 'ticker', 'currency', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume'}
    if required - set(prices):
        raise ValueError(f'Missing price columns: {required - set(prices)}')
    if prices.duplicated(['ticker', 'date']).any():
        raise ValueError('Duplicate security-date rows')
    if set(prices.ticker) != set(expected):
        raise ValueError('Provider response does not match requested tickers')
    if prices[sorted(required)].isna().any().any():
        raise ValueError('Missing required price values; inspect raw input')
    if not prices.currency.eq(cfg['currency']).all():
        raise ValueError('Mixed or unexpected currencies')
    numbers = prices[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']]
    if not np.isfinite(numbers).all().all() or (numbers.iloc[:, :4] <= 0).any().any() or (prices.volume < 0).any():
        raise ValueError('Nonfinite, nonpositive price or negative volume')
    if (prices.adj_high < prices.adj_low).any():
        raise ValueError('High price below low price')
    if ((prices.date < pd.Timestamp(cfg['start'])) | (prices.date >= pd.Timestamp(cfg['end']))).any():
        raise ValueError('Prices outside configured interval')
    report = prices.groupby('ticker').agg(first_date=('date', 'min'), last_date=('date', 'max'),
                                          sessions=('date', 'size'), median_volume=('volume', 'median'))
    report['staleness_days'] = (pd.Timestamp(cfg['end']) - report.last_date).dt.days
    if (report.staleness_days > cfg['max_staleness_days']).any():
        raise ValueError('Stale price series: ' + ', '.join(report.index[report.staleness_days > cfg['max_staleness_days']]))
    report['minimum_history_met'] = report.sessions >= cfg['min_history_sessions']
    if not report.minimum_history_met.all():
        raise ValueError('Some candidates lack minimum history; revise universe with documented reasons')
    report['largest_abs_daily_return'] = prices.sort_values('date').groupby('ticker').adj_close.pct_change(fill_method=None).abs().groupby(prices.sort_values('date').ticker).max()
    # Gaps are reported against URA's sessions, never automatically filled.
    calendar = set(prices.loc[prices.ticker.eq(cfg['benchmark']), 'date'])
    report['missing_benchmark_sessions'] = [len(calendar - set(prices.loc[prices.ticker.eq(t), 'date'])) for t in report.index]
    report['missing_sessions_during_history'] = [
        len({d for d in calendar if report.loc[t, 'first_date'] <= d <= report.loc[t, 'last_date']}
            - set(prices.loc[prices.ticker.eq(t), 'date'])) for t in report.index
    ]
    if report.missing_sessions_during_history.gt(0).any():
        raise ValueError('Missing sessions during listed history; inspect before calculating trading-session indicators')
    report['extreme_return_review_flag'] = report.largest_abs_daily_return > 0.5
    return prices.sort_values(['ticker', 'date']).reset_index(drop=True), report.reset_index()


def validate_macro(macro):
    if set(macro.series) != {'PURANUSDM', 'CPIAUCNS'} or macro.duplicated(['series', 'date']).any():
        raise ValueError('Invalid macro series or duplicate dates')
    observed = macro.value.dropna()
    if (observed <= 0).any() or not np.isfinite(observed).all():
        raise ValueError('Invalid macro values')
    if not macro.groupby('series').value.count().ge(12).all():
        raise ValueError('Insufficient observed macro history')
    # Genuine unpublished observations stay missing. Long-horizon windows with
    # missing CPI endpoints are excluded, rather than interpolated silently.
