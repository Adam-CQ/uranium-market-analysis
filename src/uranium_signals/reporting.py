"""Local CSV/HTML/PNG reports; no publishing or external messaging."""
from html import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LIMITATIONS = [
    'Fixed November 2026 current-constituent universe: survivorship, membership and issuer-selection bias; not a point-in-time investable universe.',
    'URA/HURA/NLR candidate discovery omitted due to documented access limits. US-dollar listings only; utilities and services differ from miners.',
    'Vendor-adjusted OHLC are total-return proxies, not audited total-return indices. Adjusted price times raw volume approximates liquidity.',
    'Monthly macro features use latest revised history and an assumed 60-day publication lag. Historical release vintages are not verified, so leakage from revisions cannot be ruled out.',
    'Chronological holdout is 2024 onward. Model choice uses earlier validation only; annual refits during holdout use labels already known at each cutoff.',
    'Label returns are close-to-close; simulated trades are first-session open-to-open. These intervals differ, and good classification need not produce a profitable strategy.',
    'Transaction cost is per dollar bought or sold. Turnover is also reported as half of absolute traded weight; initial purchase is charged, final liquidation is not.',
    'Buy-and-hold comparisons purchase once at evaluation inception; newer listings unavailable at inception are excluded. Cash earns zero; Sharpe uses the configured fixed risk-free rate.',
    'Long-horizon basket is fixed before performance review, equal-weight at inception and each January month-end; its descriptive returns exclude costs and tax.',
    'Long-horizon maximum drawdown is observed at monthly closes and may understate intramonth losses. Rolling windows overlap; non-overlapping counts are not proof of statistical independence.',
    'Missing CPI observations remain missing; real-return windows lacking endpoint CPI are excluded. The downloaded CPIAUCNS series has no October 2025 value.',
    'Data and reports are local educational research. Yahoo personal-use terms and IMF copyright apply; redistribution has not been cleared.',
]


def write_reports(output, metrics, calibration, portfolio_returns, economics, long_outcomes,
                  long_summary, signals, quality, cfg, selected):
    tables = {'classification': metrics, 'calibration': calibration,
              'portfolio_returns': portfolio_returns, 'economic_metrics': economics,
              'long_horizon_outcomes': long_outcomes, 'long_horizon_summary': long_summary,
              'latest_signals': signals, 'data_quality': quality}
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, series in portfolio_returns.set_index('date').items():
        ax.plot(series.index, (1 + series).cumprod(), label=name)
    ax.set(title='Holdout portfolios after assumed costs', ylabel='Growth of $1', xlabel='Signal date')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'holdout_wealth.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, horizon in zip(axes, (36, 60)):
        for ticker, g in long_outcomes[long_outcomes.horizon_months.eq(horizon)].groupby('ticker'):
            ax.plot(pd.to_datetime(g.start), g.real_cagr, label=ticker)
        ax.axhline(0, color='black', linewidth=.5)
        ax.set(ylabel='Real CAGR', title=f'{horizon // 12}-year outcomes by investment month')
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'long_horizon_real_cagr.png', dpi=150); plt.close(fig)
    sections = [f'<h1>Uranium Market Signals Lab</h1><p>Selected model: {escape(selected)}. '
                f'Primary holdout begins {escape(cfg["holdout_start"])}; signal date {signals.date.max():%Y-%m-%d}. '
                'Research results, not investment recommendations.</p>']
    for title, table in [('Data quality', quality), ('Predictive evaluation', metrics),
                         ('Holdout economic evaluation', economics), ('Latest monthly signals', signals),
                         ('Long-horizon outcomes', long_summary)]:
        sections.append(f'<h2>{title}</h2>{table.to_html(index=False, float_format=lambda x: f"{x:.4f}")}')
    sections.extend(['<img src="holdout_wealth.png" alt="Holdout wealth paths">',
                     '<img src="long_horizon_real_cagr.png" alt="Rolling real CAGR">',
                     '<h2>Assumptions and limitations</h2><ul>' + ''.join(f'<li>{escape(x)}</li>' for x in LIMITATIONS) + '</ul>'])
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Uranium research report</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}table{border-collapse:collapse;font-size:13px}td,th{padding:.4rem;border:1px solid #ddd}img{max-width:100%}li{margin:.5rem}</style>' + ''.join(sections) + '</html>'
    (output / 'report.html').write_text(html)
