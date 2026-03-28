"""
TradeSage - NSE Data Fetcher (data/fetch_nse_data.py)

Fetches historical daily OHLCV data for NSE stocks via yfinance.
Features:
  - Disk caching (avoids repeated downloads)
  - Parallel download with ThreadPoolExecutor
  - Automatic column normalisation
  - Minimum data length validation
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def _progress(iterable, total: int, desc: str = ""):
    if TQDM_AVAILABLE:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def fetch_single_stock(
    symbol: str,
    years: int = 10,
    cache_dir: str = "data_cache_yfinance",
    force_refresh: bool = False,
) -> tuple:
    """
    Fetch OHLCV data for one NSE stock (appends '.NS' suffix for yfinance).

    Returns (symbol, DataFrame | None)
    """
    os.makedirs(cache_dir, exist_ok=True)
    base_symbol = symbol.replace(".NS", "")
    cache_file = os.path.join(cache_dir, f"{base_symbol}_{years}y.csv")

    # Return cached file if fresh enough (< 24 hours)
    if not force_refresh and os.path.exists(cache_file):
        cache_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
        if cache_age < 86_400:
            try:
                df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
                if len(df) >= 200:
                    return base_symbol, df
            except Exception:
                pass

    yf_symbol = f"{base_symbol}.NS" if not symbol.endswith(".NS") else symbol

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{years}y")

        if df.empty or len(df) < 200:
            return base_symbol, None

        df = df.reset_index()
        time_col = "Date" if "Date" in df.columns else "Datetime"
        df = df.rename(
            columns={
                time_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(cache_file)
        return base_symbol, df

    except Exception:
        return base_symbol, None


def fetch_nse_data(
    symbols: list,
    years: int = 10,
    max_workers: int = 10,
    cache_dir: str = "data_cache_yfinance",
    force_refresh: bool = False,
) -> tuple:
    """
    Fetch historical data for a list of NSE symbols in parallel.

    Parameters
    ----------
    symbols      : list of NSE stock symbols (without '.NS')
    years        : years of history to fetch (default 10)
    max_workers  : parallel download threads (default 10)
    cache_dir    : directory for disk cache
    force_refresh: ignore cache and re-download

    Returns
    -------
    (results_dict, failed_list)
        results_dict : {symbol: pd.DataFrame}
        failed_list  : symbols that could not be fetched
    """
    results: dict = {}
    failed: list = []

    print(f"📥 Fetching {years}y data for {len(symbols)} NSE stocks…")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_single_stock, s, years, cache_dir, force_refresh
            ): s
            for s in symbols
        }
        for future in _progress(as_completed(futures), len(symbols), "Downloading"):
            sym, df = future.result()
            if df is not None:
                results[sym] = df
            else:
                failed.append(sym)

    print(f"✓ Fetched {len(results)} stocks  |  ⚠ Failed: {len(failed)}")
    return results, failed


def load_symbol_universe(
    path: str = "data/nifty500.json",
) -> list:
    """
    Load a list of NSE stock symbols from a JSON file.

    The JSON may be:
    - A flat list: ["RELIANCE", "TCS", ...]
    - A dict with 'symbols' key
    - A list of objects with a 'symbol' field
    """
    with open(path) as fh:
        data = json.load(fh)

    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            for key in ("symbol", "Symbol", "ticker", "Ticker"):
                if key in data[0]:
                    return [item[key] for item in data]
    if isinstance(data, dict):
        for key in ("symbols", "Symbols", "stocks", "Stocks"):
            if key in data:
                return data[key]

    raise ValueError(f"Unrecognised symbol file format in {path}")
