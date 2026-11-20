"""Read explicit research settings without modifying user secrets."""
from pathlib import Path
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    required = {'start', 'end', 'universe', 'benchmark', 'broad_benchmark',
                'holdout_start', 'validation_start', 'basket', 'top_n',
                'transaction_cost_bps', 'macro_lag_days', 'min_train_months'}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f'Missing configuration: {sorted(missing)}')
    if not cfg['start'] < cfg['validation_start'] < cfg['holdout_start'] < cfg['end']:
        raise ValueError('Dates must satisfy start < validation < holdout < end')
    if cfg['top_n'] < 1 or cfg['transaction_cost_bps'] < 0:
        raise ValueError('Invalid portfolio size or cost')
    if not 0 <= cfg['probability_threshold'] <= 1:
        raise ValueError('Probability threshold must be between zero and one')
    if cfg['macro_lag_days'] < 31 or cfg['min_train_months'] < 12:
        raise ValueError('Insufficient macro lag or training period')
    if cfg.get('currency') != 'USD' or cfg.get('basket_rebalance') != 'annual':
        raise ValueError('Initial workflow supports USD and annual basket rebalancing')
    cfg['root'] = path.parent.parent
    return cfg
