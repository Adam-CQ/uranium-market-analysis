"""Monthly observed drawdowns and overlapping historical holding outcomes."""
import numpy as np
import pandas as pd


def basket_wealth(prices, constituents):
    """Annual January month-end rebalance; first date is an inception NAV of one."""
    selected = prices[constituents]
    start = max(selected[col].first_valid_index() for col in constituents)
    selected = selected.loc[start:]
    if selected.isna().any().any():
        raise ValueError('Gaps in fixed basket after common inception; do not silently fill')
    weights = pd.Series(1 / len(constituents), index=constituents)
    wealth = [1.0]
    returns = selected.pct_change(fill_method=None)
    for date in selected.index[1:]:
        r = returns.loc[date]
        gross = float((weights * r).sum())
        wealth.append(wealth[-1] * (1 + gross))
        weights = weights * (1 + r) / (1 + gross)
        if date.month == 1:
            weights[:] = 1 / len(weights)
    return pd.Series(wealth, index=selected.index, name='BASKET')


def analyze_long_horizon(prices, macro, cfg):
    # Preserve every benchmark month. Never compress gaps into shorter horizons.
    daily = prices.pivot(index='date', columns='ticker', values='adj_close').sort_index()
    calendar = prices[prices.ticker.eq(cfg['benchmark'])].groupby(prices[prices.ticker.eq(cfg['benchmark'])].date.dt.to_period('M')).date.max()
    dates = calendar[calendar.index < pd.Timestamp(cfg['end']).to_period('M')]
    monthly = daily.reindex(dates).copy()
    monthly.index = pd.PeriodIndex(dates.index, freq='M')
    cpi = macro[macro.series.eq('CPIAUCNS')].copy()
    cpi.index = cpi.date.dt.to_period('M')
    cpi = cpi.value.reindex(monthly.index)
    basket = basket_wealth(monthly, cfg['basket'])
    levels = monthly[['CCJ', cfg['benchmark'], cfg['broad_benchmark']]].copy()
    levels['BASKET'] = basket
    records = []
    for ticker in levels:
        series = levels[ticker]
        for horizon in (36, 60):
            for i in range(len(levels) - horizon):
                window = series.iloc[i:i + horizon + 1]
                inflation = cpi.iloc[[i, i + horizon]]
                if window.isna().any() or inflation.isna().any():
                    continue
                nominal = window.iloc[-1] / window.iloc[0] - 1
                real = (1 + nominal) / (inflation.iloc[-1] / inflation.iloc[0]) - 1
                ura = levels[cfg['benchmark']].iloc[[i, i + horizon]]
                broad = levels[cfg['broad_benchmark']].iloc[[i, i + horizon]]
                records.append({'ticker': ticker, 'horizon_months': horizon,
                                'start': str(window.index[0]), 'end': str(window.index[-1]),
                                'nominal_return': nominal, 'real_return': real,
                                'nominal_cagr': (1 + nominal) ** (12 / horizon) - 1,
                                'real_cagr': (1 + real) ** (12 / horizon) - 1,
                                'excess_return_ura': nominal - (ura.iloc[-1] / ura.iloc[0] - 1),
                                'excess_return_spy': nominal - (broad.iloc[-1] / broad.iloc[0] - 1),
                                'monthly_max_drawdown': (window / window.cummax() - 1).min()})
    outcomes = pd.DataFrame(records)
    if outcomes.empty:
        raise ValueError('No complete long-horizon windows')
    summaries = []
    for (ticker, horizon), g in outcomes.groupby(['ticker', 'horizon_months']):
        last_end = None
        count = 0
        for row in g.sort_values('start').itertuples():
            start = pd.Period(row.start, freq='M')
            end = pd.Period(row.end, freq='M')
            if last_end is None or start >= last_end:
                count += 1
                last_end = end
        summaries.append({'ticker': ticker, 'horizon_months': horizon, 'rolling_windows': len(g),
                          'nonoverlapping_windows': count, 'positive_real_frequency': g.real_return.gt(0).mean(),
                          'median_nominal_cagr': g.nominal_cagr.median(), 'median_real_cagr': g.real_cagr.median(),
                          'median_excess_return_ura': g.excess_return_ura.median(),
                          'median_excess_return_spy': g.excess_return_spy.median(),
                          'worst_monthly_drawdown': g.monthly_max_drawdown.min()})
    return outcomes, pd.DataFrame(summaries), levels
