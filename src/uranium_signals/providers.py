"""Bounded downloads and exact, immutable local cache manifests.

Yahoo prices are for personal educational research. IMF redistribution remains
uncleared. Cache files are ignored by Git; do not publish them automatically.
"""
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
import logging
import time
import pandas as pd

LOG = logging.getLogger(__name__)


def _retry(call):
    for attempt in range(3):
        try:
            return call()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _key(cfg):
    return f"{cfg['start']}_{cfg['end']}"


def acquire(cfg, tickers, refresh=False, offline=False):
    """Fetch requested inputs, or validate and reuse exact cached bytes."""
    cache = cfg['root'] / 'data/raw/market' / _key(cfg)
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists() and not refresh:
        metadata = json.loads(manifest_path.read_text())
        if metadata['tickers'] != sorted(tickers):
            raise ValueError('Cached universe differs; use --refresh to acquire a new snapshot')
        cache = cfg['root'] / metadata['cache_path']
        for filename, digest in metadata['sha256'].items():
            if sha256((cache / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Cache checksum mismatch: {filename}')
        return (pd.read_csv(cache / 'prices.csv', parse_dates=['date']),
                pd.read_csv(cache / 'macro.csv', parse_dates=['date']), metadata)
    if offline:
        raise FileNotFoundError(f'No validated market cache: {cache}')
    cache.mkdir(parents=True, exist_ok=True)
    import yfinance as yf
    yf.set_tz_cache_location(str(cfg['root'] / 'data/raw/yfinance_cache'))
    frames, instruments = [], {}
    for ticker in sorted(tickers):
        checkpoint = cache / f'{ticker}_checkpoint.csv'
        checkpoint_meta = cache / f'{ticker}_checkpoint.json'
        if checkpoint.exists() and checkpoint_meta.exists() and not refresh:
            saved = json.loads(checkpoint_meta.read_text())
            if sha256(checkpoint.read_bytes()).hexdigest() != saved['sha256']:
                raise ValueError(f'Invalid checkpoint for {ticker}')
            frames.append(pd.read_csv(checkpoint, parse_dates=['date']))
            instruments[ticker] = saved['instrument']
            continue
        LOG.info('Downloading %s', ticker)
        def fetch():
            security = yf.Ticker(ticker)
            history = security.history(start=cfg['start'], end=cfg['end'],
                                       auto_adjust=True, actions=True, raise_errors=True)
            if history.empty:
                raise ValueError(f'Empty provider response for {ticker}')
            meta = security.history_metadata
            if meta.get('currency') != cfg['currency']:
                raise ValueError(f'{ticker}: expected {cfg["currency"]}, got {meta.get("currency")}')
            return history, meta
        history, meta = _retry(fetch)
        instruments[ticker] = {k: meta.get(k) for k in
                               ('currency', 'exchangeName', 'instrumentType', 'longName', 'firstTradeDate')}
        history.index = history.index.tz_localize(None).normalize()
        selected = history.rename(columns={'Open': 'adj_open', 'High': 'adj_high',
                                            'Low': 'adj_low', 'Close': 'adj_close',
                                            'Volume': 'volume', 'Dividends': 'dividends',
                                            'Stock Splits': 'stock_splits'})
        selected.index.name = 'date'
        selected = selected.reset_index()
        selected['ticker'] = ticker
        selected['currency'] = meta['currency']
        columns = ['date', 'ticker', 'currency', 'adj_open', 'adj_high', 'adj_low',
                   'adj_close', 'volume', 'dividends', 'stock_splits']
        frames.append(selected[columns])
        if not refresh:
            selected[columns].to_csv(checkpoint, index=False)
            checkpoint_meta.write_text(json.dumps({'instrument': instruments[ticker], 'sha256': sha256(checkpoint.read_bytes()).hexdigest(), 'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}, default=str))
        time.sleep(0.25)
    prices = pd.concat(frames, ignore_index=True)
    macro_frames = []
    for series in ('PURANUSDM', 'CPIAUCNS'):
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
        LOG.info('Downloading official FRED CSV: %s', series)
        def fetch_csv():
            # Reuse the HTTP transport installed with yfinance. The standard
            # library HTTPS client times out against this endpoint in WSL.
            from curl_cffi import requests
            response = requests.get(url, impersonate='chrome', timeout=20)
            response.raise_for_status()
            return response.content.decode('utf-8-sig')
        frame = pd.read_csv(StringIO(_retry(fetch_csv)), na_values='.')
        if list(frame.columns) != ['observation_date', series]:
            raise ValueError(f'Unexpected FRED CSV schema: {series}: {list(frame.columns)}')
        frame = frame.rename(columns={'observation_date': 'date', series: 'value'})
        frame['date'] = pd.to_datetime(frame['date'])
        frame['series'] = series
        # Preserve pre-start history for lagged macro changes.
        frame = frame[frame.date < pd.Timestamp(cfg['end'])]
        macro_frames.append(frame)
    macro = pd.concat(macro_frames, ignore_index=True)
    cache.mkdir(parents=True, exist_ok=True)
    # Refresh keeps older snapshots instead of overwriting them.
    if manifest_path.exists():
        cache = cache / datetime.now(timezone.utc).strftime('refresh_%Y%m%dT%H%M%S%fZ')
        cache.mkdir()
        manifest_path = cache / 'manifest.json'
    prices.to_csv(cache / 'prices.csv', index=False)
    macro.to_csv(cache / 'macro.csv', index=False)
    metadata = {
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'start': cfg['start'], 'end_exclusive': cfg['end'], 'tickers': sorted(tickers),
        'price_provider': 'Yahoo Finance via yfinance; auto_adjust=True (OHLC adjusted)',
        'price_terms': 'https://ranaroussi.github.io/yfinance/ — personal educational use; no redistribution clearance',
        'macro_sources': {s: f'https://fred.stlouisfed.org/series/{s}' for s in ('PURANUSDM', 'CPIAUCNS')},
        'macro_terms': 'https://fred.stlouisfed.org/legal/; PURANUSDM has IMF copyright; local research only',
        'units': {'prices': 'USD/share adjusted', 'volume': 'shares', 'PURANUSDM': 'nominal USD/lb monthly average', 'CPIAUCNS': 'index 1982-84=100 NSA'},
        'availability': 'Latest revised macro history; observation month-end + configured lag is a proxy, not verified vintages',
        'instruments': instruments,
        'cache_path': str(cache.relative_to(cfg['root'])),
        'sha256': {f: sha256((cache / f).read_bytes()).hexdigest() for f in ('prices.csv', 'macro.csv')},
    }
    manifest_path.write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    active_manifest = cfg['root'] / 'data/raw/market' / _key(cfg) / 'manifest.json'
    if manifest_path != active_manifest:
        archive = active_manifest.with_name(f'manifest_before_{cache.name}.json')
        archive.write_bytes(active_manifest.read_bytes())
        active_manifest.write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    return prices, macro, metadata
