#!/usr/bin/env python3
"""
TradeSage — Fetch 20 Years of NSE Daily OHLCV via yfinance

Reads symbol list from data_cache_angel/ filenames, fetches 20yr history
from Yahoo Finance, and saves to data_cache_yfinance/.

Usage:
    python scripts/fetch_yfinance_10y.py              # Full fetch
    python scripts/fetch_yfinance_10y.py --update      # Skip recently cached
    python scripts/fetch_yfinance_10y.py --max 100     # Limit to N stocks
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANGEL_CACHE  = PROJECT_ROOT / 'data_cache_angel'
YF_CACHE     = PROJECT_ROOT / 'data_cache_yfinance'
LOG_DIR      = PROJECT_ROOT / 'logs'
FAILED_LOG   = LOG_DIR / 'yfinance_failed.txt'

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def get_symbols_from_angel_cache():
    """Read symbol list from data_cache_angel/ filenames (strip _daily.csv)."""
    if not ANGEL_CACHE.exists():
        print(f"ERROR: {ANGEL_CACHE} not found")
        sys.exit(1)

    symbols = []
    skip_keywords = ['BEES', 'IETF', 'BETA', 'ETF', 'NIFTY', 'SENSEX',
                     'GOLD', 'SILVER', 'LIQUID', 'GILT', 'BOND', 'NSEI',
                     'instruments']

    for f in sorted(ANGEL_CACHE.glob('*_daily.csv')):
        if f.stat().st_size < 1000:
            continue
        sym = f.stem.replace('_daily', '')
        if any(k in sym.upper() for k in skip_keywords):
            continue
        symbols.append(sym)

    return symbols


def should_skip(symbol, cache_dir, days_threshold=7):
    """Return True if cached file exists and last row is within threshold days."""
    cache_file = cache_dir / f"{symbol}_daily.csv"
    if not cache_file.exists():
        return False

    try:
        # Read only last few rows for speed
        df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
        if len(df) < 200:
            return False  # Too short, re-fetch
        last_date = df.index[-1]
        if pd.Timestamp.now() - last_date < timedelta(days=days_threshold):
            return True
    except Exception:
        return False

    return False


def fetch_single_stock(symbol, years=20, cache_dir=YF_CACHE):
    """Fetch one stock from yfinance and save to CSV."""
    yf_symbol = f"{symbol}.NS"
    cache_file = cache_dir / f"{symbol}_daily.csv"

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{years}y")

        if df is None or df.empty or len(df) < 100:
            return symbol, 'failed', f"Only {len(df) if df is not None else 0} rows"

        # Normalize columns to match Angel One schema
        df = df.reset_index()
        time_col = 'Date' if 'Date' in df.columns else 'Datetime'
        df = df.rename(columns={
            time_col: 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        })

        # Keep only OHLCV
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Remove timezone info
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Drop zero-volume rows (partial days / bad data)
        df = df[df['volume'] > 0]

        if len(df) < 100:
            return symbol, 'failed', f"Only {len(df)} valid rows after cleaning"

        df.to_csv(cache_file)
        return symbol, 'downloaded', f"{len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})"

    except Exception as e:
        err = str(e)[:80]
        return symbol, 'failed', err


def main():
    parser = argparse.ArgumentParser(description='Fetch 20yr NSE data from yfinance')
    parser.add_argument('--update', action='store_true',
                        help='Skip symbols cached within last 7 days')
    parser.add_argument('--max', type=int, default=None,
                        help='Limit to N stocks')
    parser.add_argument('--years', type=int, default=20,
                        help='Years of history to fetch (default: 20)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0)')
    args = parser.parse_args()

    # Setup
    YF_CACHE.mkdir(exist_ok=True, parents=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print(f"  TRADESAGE — FETCH {args.years}yr NSE DATA (yfinance)")
    print("=" * 70)

    symbols = get_symbols_from_angel_cache()
    if args.max:
        symbols = symbols[:args.max]

    print(f"  Symbols found:  {len(symbols)}")
    print(f"  History:        {args.years} years")
    print(f"  Cache dir:      {YF_CACHE}")
    print(f"  Rate limit:     {args.delay}s between requests")
    print(f"  Update mode:    {'ON' if args.update else 'OFF'}")

    downloaded = 0
    skipped = 0
    failed = 0
    failed_list = []

    start_time = time.time()

    for symbol in tqdm(symbols, desc="Fetching", unit="stock"):
        # Skip check
        if args.update and should_skip(symbol, YF_CACHE, days_threshold=7):
            skipped += 1
            continue

        sym, status, detail = fetch_single_stock(symbol, years=args.years, cache_dir=YF_CACHE)

        if status == 'downloaded':
            downloaded += 1
        else:
            failed += 1
            failed_list.append(f"{sym}: {detail}")

        # Rate limit
        time.sleep(args.delay)

    elapsed = time.time() - start_time

    # Write failed log
    with open(FAILED_LOG, 'w', encoding='utf-8') as f:
        f.write(f"# yfinance fetch failures — {datetime.now().isoformat()}\n")
        f.write(f"# Total: {failed} / {len(symbols)}\n\n")
        for line in failed_list:
            f.write(line + '\n')

    # Summary
    print("\n" + "=" * 70)
    print("  FETCH COMPLETE")
    print("=" * 70)
    print(f"  Downloaded:  {downloaded}")
    print(f"  Skipped:     {skipped}")
    print(f"  Failed:      {failed}")
    print(f"  Total time:  {elapsed / 60:.1f} minutes")
    print(f"  Failed log:  {FAILED_LOG}")
    print("=" * 70)

    # Count total cached files
    cached_count = len(list(YF_CACHE.glob('*_daily.csv')))
    print(f"\n  Total cached files in {YF_CACHE.name}/: {cached_count}")


if __name__ == '__main__':
    main()
