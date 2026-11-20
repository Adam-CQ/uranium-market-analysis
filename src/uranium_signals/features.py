"""Explicit trailing calculations; no learned preprocessing or future filling."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'ret_1', 'ret_5', 'ret_21', 'ret_63', 'ret_126',
    'ma_dist_20', 'ma_dist_50', 'ma_dist_200', 'ma_cross_20_50',
    'vol_21', 'downside_vol_21', 'drawdown_126', 'volume_change_21',
    'rel_63', 'broad_rel_63', 'momentum_rank',
    'uranium_change_1', 'uranium_change_3', 'uranium_change_6',
]


def daily_features(prices, cfg):
    blocks = []
    for ticker, frame in prices.groupby('ticker'):
        g = frame.sort_values('date').copy().set_index('date')
        p = g.adj_close
        daily_return = p.pct_change(fill_method=None)
        for window in (1, 5, 21, 63, 126):
            g[f'ret_{window}'] = p.pct_change(window, fill_method=None)
        for window in (20, 50, 200):
            g[f'ma_dist_{window}'] = p / p.rolling(window).mean() - 1
        g['ma_cross_20_50'] = p.rolling(20).mean() / p.rolling(50).mean() - 1
        g['vol_21'] = daily_return.rolling(21).std() * np.sqrt(252)
        g['downside_vol_21'] = daily_return.clip(upper=0).pow(2).rolling(21).mean().pow(0.5) * np.sqrt(252)
        g['drawdown_126'] = p / p.rolling(126).max() - 1
        g['volume_change_21'] = g.volume / g.volume.rolling(21).mean() - 1
        g['median_dollar_volume_63'] = (p * g.volume).rolling(63).median()
        # Adjusted price * raw volume is only an approximate liquidity measure.
        g['history_sessions'] = np.arange(1, len(g) + 1)
        blocks.append(g.reset_index())
    return pd.concat(blocks, ignore_index=True)


def macro_features(macro, cfg):
    wide = macro.pivot(index='date', columns='series', values='value').sort_index()
    # A regular monthly grid prevents a missing month from masquerading as a one-month change.
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='MS'))
    out = pd.DataFrame(index=wide.index)
    for months in (1, 3, 6):
        out[f'uranium_change_{months}'] = wide.PURANUSDM.pct_change(months, fill_method=None)
    out['uranium_nominal'] = wide.PURANUSDM
    out['cpi'] = wide.CPIAUCNS
    base = wide.CPIAUCNS.dropna().iloc[-1]
    out['uranium_real_latest_cpi_base'] = wide.PURANUSDM / (wide.CPIAUCNS / base)
    out['macro_observation_date'] = out.index
    out['macro_available_at'] = out.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=cfg['macro_lag_days'])
    # The real-price display uses a latest CPI base and is deliberately excluded from predictors.
    return out.dropna(subset=['uranium_change_6']).reset_index(drop=True)


def monthly_features(prices, macro, cfg, tickers):
    daily = daily_features(prices, cfg)
    benchmark = daily[daily.ticker.eq(cfg['benchmark'])].sort_values('date')
    sessions = benchmark.groupby(benchmark.date.dt.to_period('M')).date.max()
    # End is exclusive. Exclude an incomplete final month from signal formation.
    completed = sessions[sessions.index < pd.Timestamp(cfg['end']).to_period('M')]
    monthly = daily[daily.date.isin(completed)].copy()
    for source, output in [(cfg['benchmark'], 'rel_63'), (cfg['broad_benchmark'], 'broad_rel_63')]:
        reference = monthly[monthly.ticker.eq(source)][['date', 'ret_63']].rename(columns={'ret_63': 'benchmark_ret'})
        monthly = monthly.merge(reference, on='date', how='left', validate='many_to_one')
        monthly[output] = monthly.ret_63 - monthly.benchmark_ret
        monthly = monthly.drop(columns='benchmark_ret')
    monthly = monthly[monthly.ticker.isin(tickers)].copy()
    monthly['eligible'] = ((monthly.history_sessions >= cfg['min_history_sessions']) &
                           (monthly.median_dollar_volume_63 >= cfg['min_median_dollar_volume']) &
                           (monthly.adj_close >= cfg['min_price']))
    monthly['momentum_rank'] = monthly.where(monthly.eligible).groupby('date').ret_63.rank(pct=True)
    # Rank only eligible securities, using the information set of that month.
    monthly = pd.merge_asof(monthly.sort_values('date'), macro_features(macro, cfg).sort_values('macro_available_at'),
                            left_on='date', right_on='macro_available_at', direction='backward')
    monthly[FEATURE_COLUMNS] = monthly[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return monthly.sort_values(['date', 'ticker']).reset_index(drop=True)
