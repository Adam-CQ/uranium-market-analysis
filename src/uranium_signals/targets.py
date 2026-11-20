"""Separate close-to-close prediction labels from executable open-to-open returns."""
import pandas as pd


def build_targets(prices, cfg):
    calendar = prices[prices.ticker.eq(cfg['benchmark'])].sort_values('date')
    grouped = calendar.groupby(calendar.date.dt.to_period('M'))
    last = grouped.date.max()
    first = grouped.date.min()
    close = prices.pivot(index='date', columns='ticker', values='adj_close')
    opening = prices.pivot(index='date', columns='ticker', values='adj_open')
    rows = []
    for period, signal_date in last.items():
        if period >= pd.Timestamp(cfg['end']).to_period('M'):
            continue
        next_end = last.get(period + 1)
        if period + 1 >= pd.Timestamp(cfg['end']).to_period('M'):
            next_end = None  # partial next-month close is not a completed label
        execution = first.get(period + 1)
        exit_date = first.get(period + 2)
        for ticker in close.columns:
            target_return = (close.at[next_end, ticker] / close.at[signal_date, ticker] - 1) if next_end is not None else float('nan')
            trade_return = (opening.at[exit_date, ticker] / opening.at[execution, ticker] - 1) if execution is not None and exit_date is not None else float('nan')
            rows.append({'date': signal_date, 'ticker': ticker, 'label_end': next_end,
                         'forward_return': target_return, 'execution_date': execution,
                         'exit_date': exit_date, 'trade_return': trade_return})
    out = pd.DataFrame(rows)
    benchmark = out[out.ticker.eq(cfg['benchmark'])][['date', 'forward_return']].rename(columns={'forward_return': 'benchmark_forward_return'})
    out = out.merge(benchmark, on='date', validate='many_to_one')
    known = out.forward_return.notna() & out.benchmark_forward_return.notna()
    out['outperform_1m'] = (out.forward_return > out.benchmark_forward_return).astype(float).where(known)
    return out
