"""Long-only monthly portfolios with next-session open execution and costs."""
import numpy as np
import pandas as pd


def portfolio_path(predictions, targets, cfg, kind='model'):
    dates = sorted(predictions.date.unique())
    trade = targets.pivot(index='date', columns='ticker', values='trade_return')
    previous = pd.Series(0.0, index=trade.columns)
    records = []
    for date in dates:
        candidates = predictions[predictions.date.eq(date)].copy()
        date_targets = targets[targets.date.eq(date)]
        if date_targets.exit_date.isna().all():
            continue  # do not mark an incomplete holding period as realized
        if kind == 'model':
            chosen = candidates[candidates.probability >= cfg['probability_threshold']].sort_values(
                ['probability', 'ticker'], ascending=[False, True]).head(cfg['top_n']).ticker
        elif kind == 'momentum':
            chosen = candidates[candidates.rel_63 > 0].sort_values(['rel_63', 'ticker'], ascending=[False, True]).head(cfg['top_n']).ticker
        else:
            raise ValueError(f'Unknown strategy {kind}')
        weights = previous * 0
        if len(chosen):
            weights.loc[chosen] = 1 / len(chosen)
        returns = trade.loc[date]
        if returns[weights > 0].isna().any():
            raise ValueError(f'Missing execution return for a held security on {date}')
        traded_value = float((weights - previous).abs().sum())
        cost = traded_value * cfg['transaction_cost_bps'] / 10000
        gross = float((weights * returns.fillna(0)).sum())
        net = (1 - cost) * (1 + gross) - 1
        execution = date_targets.execution_date.dropna().iloc[0]
        exit_date = date_targets.exit_date.dropna().iloc[0]
        records.append({'date': date, 'execution_date': execution, 'exit_date': exit_date,
                        'return': net, 'gross_return': gross, 'cost_fraction': cost,
                        'traded_value': traded_value, 'one_way_turnover': traded_value / 2,
                        'trades': int(((weights - previous).abs() > 1e-10).sum()),
                        'holdings': '|'.join(chosen), 'cash_weight': 1 - weights.sum()})
        # Drift holdings through the period; costs are funded proportionally.
        previous = weights * (1 + returns.fillna(0)) / (1 + gross)
    if not records:
        raise ValueError('No complete strategy holding periods')
    return pd.DataFrame(records)


def buy_hold_path(targets, dates, tickers, cfg):
    """Buy equally once; no subsequent rebalancing. Missing held returns fail."""
    pivot = targets.pivot(index='date', columns='ticker', values='trade_return').reindex(dates)
    selected = pivot.loc[dates[0], tickers].dropna().index
    if len(selected) == 0:
        raise ValueError('No securities available at buy-and-hold inception')
    weights = pd.Series(1 / len(selected), index=selected)
    returns = []
    for i, date in enumerate(dates):
        r = pivot.loc[date, selected]
        if r.isna().any():
            raise ValueError(f'Missing buy-and-hold return at {date}')
        gross = float((weights * r).sum())
        cost = cfg['transaction_cost_bps'] / 10000 if i == 0 else 0
        returns.append((1 - cost) * (1 + gross) - 1)
        weights = weights * (1 + r) / (1 + gross)
    return pd.Series(returns, index=dates), list(selected)


def performance(returns, risk_free_annual=0):
    r = pd.Series(returns).dropna().astype(float)
    if r.empty or (r <= -1).any():
        raise ValueError('Empty or invalid strategy returns')
    wealth = np.r_[1.0, (1 + r).cumprod().to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    annualized = wealth[-1] ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    monthly_rf = (1 + risk_free_annual) ** (1 / 12) - 1
    return {'months': len(r), 'cumulative_return': wealth[-1] - 1,
            'annualized_return': annualized, 'max_drawdown': drawdown.min(),
            'annualized_volatility': vol,
            'sharpe': (r.mean() - monthly_rf) * 12 / vol if vol > 0 else np.nan}
