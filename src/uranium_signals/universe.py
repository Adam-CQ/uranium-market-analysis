"""Fixed current-constituent research universe; never inferred from returns."""
import pandas as pd


def load_universe(cfg):
    frame = pd.read_csv(cfg['root'] / cfg['universe'])
    required = {'ticker', 'currency', 'company_name', 'exposure_group', 'source_etfs'}
    if required - set(frame):
        raise ValueError('Incomplete universe schema')
    if frame.ticker.duplicated().any() or frame.ticker.isna().any():
        raise ValueError('Universe tickers must be present and unique')
    if not frame.currency.eq(cfg['currency']).all():
        raise ValueError('Mixed currencies require explicit FX conversion')
    if not set(cfg['basket']) <= set(frame.ticker):
        raise ValueError('Basket must be fixed within the candidate universe')
    return frame
