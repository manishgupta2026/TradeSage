#!/usr/bin/env python3
"""
TradeSage Unified Training Script
Replaces: train_angel_one.py, train_production.py, train_from_cache.py,
           retrain_v2.py, train_colab.py

Usage:
    python scripts/train.py --source cache                    # Train from Angel One cached CSVs
    python scripts/train.py --source yfinance                 # Train from yfinance 20yr CSVs
    python scripts/train.py --source angel                    # Fetch live from Angel One
    python scripts/train.py --source cache --ensemble         # Enable ensemble
    python scripts/train.py --forward-days 5 --threshold 0.04 # Custom config
"""

import argparse
import json
import logging
import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────────────

def load_from_cache(cache_dir, max_stocks=None):
    """Load stock data from cached CSV files."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return {}

    csv_files = sorted(cache_path.glob('*_daily.csv'))
    if max_stocks:
        csv_files = csv_files[:max_stocks]

    logger.info(f"  Found {len(csv_files)} cached CSV files")

    stock_data = {}
    failed = 0

    for csv_file in tqdm(csv_files, desc="Loading CSVs"):
        symbol = csv_file.stem.replace('_daily', '')
        try:
            # Try both index column names
            try:
                df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
            except Exception:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [str(c).lower().strip() for c in df.columns]

            required = ['open', 'high', 'low', 'close', 'volume']
            if len(df) >= 200 and all(c in df.columns for c in required):
                stock_data[symbol] = df
            else:
                failed += 1
        except Exception:
            failed += 1

    logger.info(f"  Loaded: {len(stock_data)} stocks  |  Skipped: {failed}")
    return stock_data


def load_from_angel(symbols, period_days=1095, max_workers=5):
    """Fetch stock data from Angel One API."""
    from src.angel.angel_one_api import AngelOneAPI
    from src.angel.angel_data_fetcher import AngelDataFetcher

    api = AngelOneAPI(str(PROJECT_ROOT / 'config' / 'angel_config.json'))
    fetcher = AngelDataFetcher(api)

    stock_data, failed = fetcher.fetch_multiple_symbols(
        symbols, period_days=period_days, max_workers=max_workers
    )
    return stock_data


def load_nifty_index():
    """Try to load Nifty50 index data for market context."""
    try:
        import yfinance as yf
        logger.info("  Fetching Nifty50 via yfinance...")
        nifty = yf.download('^NSEI', period='3y', progress=False)
        if len(nifty) > 200:
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            nifty.columns = [str(c).lower() for c in nifty.columns]
            if nifty.index.tz is not None:
                nifty.index = nifty.index.tz_localize(None)
            logger.info(f"  ✓ Nifty50: {len(nifty)} rows")
            return nifty
    except Exception as e:
        logger.warning(f"  Could not fetch Nifty50: {e}")
    return None


# ──────────────────────────────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='TradeSage Unified Training')
    parser.add_argument('--source', choices=['angel', 'cache', 'yfinance'], default='cache',
                        help='Data source: angel (live API), cache (Angel One CSVs), yfinance (20yr CSVs)')
    parser.add_argument('--cache-dir', default=str(PROJECT_ROOT / 'data_cache_angel'),
                        help='Cache directory for CSV files')
    parser.add_argument('--model-path', default=str(PROJECT_ROOT / 'models' / 'tradesage_model.pkl'),
                        help='Output model path')
    parser.add_argument('--forward-days', type=int, default=5,
                        help='Prediction horizon in trading days (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.04,
                        help='Gain threshold for positive label (default: 0.04 = 4%%)')
    parser.add_argument('--max-drawdown', type=float, default=-0.03,
                        help='Max drawdown filter for target variable (default: -0.03 = -3%%)')
    parser.add_argument('--max-rows-per-stock', type=int, default=2000,
                        help='Max rows per stock to limit memory (default: 2000)')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='Max number of stocks to process (default: all)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Enable ensemble mode (XGB + LightGBM + CatBoost)')
    parser.add_argument('--no-ensemble', dest='ensemble', action='store_false')
    parser.set_defaults(ensemble=True)
    parser.add_argument('--max-workers', type=int, default=5,
                        help='Parallel workers for Angel One fetching')
    parser.add_argument('--stock-list', default=None,
                        help='Path to JSON file with stock symbols (for --source angel)')
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 80)
    logger.info("  TRADESAGE UNIFIED TRAINING")
    logger.info("=" * 80)
    logger.info(f"  Source:          {args.source}")
    logger.info(f"  Model:           {args.model_path}")
    logger.info(f"  Forward days:    {args.forward_days}")
    logger.info(f"  Gain threshold:  {args.threshold * 100}%")
    logger.info(f"  Max drawdown:    {args.max_drawdown * 100}%")
    logger.info(f"  Ensemble:        {'ON' if args.ensemble else 'OFF'}")
    logger.info(f"  Max rows/stock:  {args.max_rows_per_stock}")

    # ── Step 1: Load data ──
    logger.info(f"\n📥 Step 1: Loading data ({args.source})...")

    if args.source == 'cache':
        stock_data = load_from_cache(args.cache_dir, max_stocks=args.max_stocks)
    elif args.source == 'yfinance':
        yf_cache = str(PROJECT_ROOT / 'data_cache_yfinance')
        logger.info(f"  Loading from {yf_cache}...")
        stock_data = load_from_cache(yf_cache, max_stocks=args.max_stocks)
    else:
        # Load symbol list
        stock_list_path = args.stock_list or str(PROJECT_ROOT / 'data' / 'nse_top_3000_angel.json')
        logger.info(f"  Loading symbols from {stock_list_path}...")
        with open(stock_list_path, 'r') as f:
            symbols = json.load(f)
        if args.max_stocks:
            symbols = symbols[:args.max_stocks]
        period_days = 1095  # 3 years
        stock_data = load_from_angel(symbols, period_days=period_days,
                                     max_workers=args.max_workers)

    if len(stock_data) < 10:
        logger.error(f"Not enough data: only {len(stock_data)} stocks. Need ≥10.")
        sys.exit(1)

    logger.info(f"  ✓ Loaded {len(stock_data)} stocks")

    # ── Step 2: Market context ──
    logger.info(f"\n🌐 Step 2: Loading market context...")
    nifty_df = load_nifty_index()
    if nifty_df is not None:
        logger.info("  Market context: ENABLED")
    else:
        logger.info("  Market context: DISABLED")

    # ── Step 3: Feature engineering ──
    logger.info(f"\n🔧 Step 3: Feature engineering for {len(stock_data)} stocks...")
    engineer = FeatureEngineer()

    all_data = []
    failed_count = 0

    for symbol, df in tqdm(stock_data.items(), desc="Engineering features"):
        try:
            if args.max_rows_per_stock and len(df) > args.max_rows_per_stock:
                df = df.iloc[-args.max_rows_per_stock:]

            df_feat = engineer.add_technical_indicators(df, index_df=nifty_df)
            df_final = engineer.create_target_variable(
                df_feat,
                forward_days=args.forward_days,
                gain_threshold=args.threshold,
                max_drawdown=args.max_drawdown,
            )
            df_final['symbol'] = symbol
            all_data.append(df_final)
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                logger.warning(f"  Failed {symbol}: {e}")

    logger.info(f"  Processed: {len(all_data)} / {len(stock_data)}  |  Failed: {failed_count}")

    if not all_data:
        logger.error("No valid data after feature engineering!")
        sys.exit(1)

    # ── Step 4: Combine and split ──
    logger.info(f"\n📊 Step 4: Walk-forward date split (90 / 5 / 5)...")
    combined = pd.concat(all_data).sort_index()
    X, y, feature_names = engineer.prepare_training_data(combined)

    dates = X.index.unique().sort_values()
    n_dates = len(dates)
    val_start = dates[int(n_dates * 0.90)]
    test_start = dates[int(n_dates * 0.95)]

    train_mask = X.index < val_start
    val_mask = (X.index >= val_start) & (X.index < test_start)
    test_mask = X.index >= test_start

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Train: {len(X_train):,} samples (up to {val_start.date()})")
    logger.info(f"  Val:   {len(X_val):,} samples ({val_start.date()} → {test_start.date()})")
    logger.info(f"  Test:  {len(X_test):,} samples ({test_start.date()} → end)")
    logger.info(f"  Positive rate (train): {y_train.mean() * 100:.1f}%")

    # ── Step 5: Train ──
    logger.info(f"\n🤖 Step 5: Training model...")
    trainer = TradingModelTrainer()
    trainer.train_model(X_train, y_train, X_val, y_val, use_ensemble=args.ensemble)

    # ── Step 6: Evaluate ──
    logger.info(f"\n📈 Step 6: Evaluation...")
    logger.info(f"\n--- VALIDATION SET ---")
    val_metrics = trainer.evaluate_model(X_val, y_val)

    logger.info(f"\n--- TEST SET (held-out, never seen during training) ---")
    test_metrics = trainer.evaluate_model(X_test, y_test)

    # ── Step 7: Feature importance ──
    logger.info(f"\n🔍 Step 7: Feature importance...")
    importance_df = trainer.get_feature_importance(top_n=20)

    # ── Step 8: Save ──
    logger.info(f"\n💾 Step 8: Saving model...")
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    trainer.save_model(args.model_path)

    # Save report
    elapsed = time.time() - start_time
    report_path = args.model_path.replace('.pkl', '_report.json')

    def safe_metric(m, key):
        v = m.get(key, 0)
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return v

    report = {
        'training_date': datetime.now().isoformat(),
        'version': 'unified-v1',
        'elapsed_seconds': round(elapsed, 1),
        'data_source': args.source,
        'stocks_trained': len(all_data),
        'total_samples': len(X),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features': len(feature_names),
        'ensemble_mode': args.ensemble,
        'market_context': nifty_df is not None,
        'split': {
            'train_until': str(val_start.date()),
            'val_until': str(test_start.date()),
            'test_from': str(test_start.date()),
        },
        'parameters': {
            'forward_days': args.forward_days,
            'gain_threshold': args.threshold,
            'max_drawdown': args.max_drawdown,
            'max_rows_per_stock': args.max_rows_per_stock,
        },
        'val_metrics': {
            'auc_score': safe_metric(val_metrics, 'auc_score'),
            'precision': safe_metric(val_metrics, 'precision'),
            'recall': safe_metric(val_metrics, 'recall'),
            'f1': safe_metric(val_metrics, 'f1'),
            'profit_score': safe_metric(val_metrics, 'profit_score'),
            'predicted_win_rate': safe_metric(val_metrics, 'predicted_win_rate'),
        },
        'test_metrics': {
            'auc_score': safe_metric(test_metrics, 'auc_score'),
            'precision': safe_metric(test_metrics, 'precision'),
            'recall': safe_metric(test_metrics, 'recall'),
            'f1': safe_metric(test_metrics, 'f1'),
            'profit_score': safe_metric(test_metrics, 'profit_score'),
            'predicted_win_rate': safe_metric(test_metrics, 'predicted_win_rate'),
        },
        'top_features': importance_df.head(20).to_dict('records'),
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ── Final Summary ──
    logger.info("\n" + "=" * 80)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Time:    {elapsed / 60:.1f} minutes")
    logger.info(f"  Model:   {args.model_path}")
    logger.info(f"  Report:  {report_path}")
    logger.info(f"")
    logger.info(f"  VAL  AUC: {val_metrics.get('auc_score', 0):.4f}  "
                f"Prec: {val_metrics.get('precision', 0):.4f}  "
                f"Win: {val_metrics.get('predicted_win_rate', 0) * 100:.1f}%")
    logger.info(f"  TEST AUC: {test_metrics.get('auc_score', 0):.4f}  "
                f"Prec: {test_metrics.get('precision', 0):.4f}  "
                f"Win: {test_metrics.get('predicted_win_rate', 0) * 100:.1f}%")

    test_auc = test_metrics.get('auc_score', 0)
    if test_auc >= 0.75:
        logger.info(f"\n  ★ EXCELLENT — AUC {test_auc:.4f} >= 0.75 target!")
    elif test_auc >= 0.70:
        logger.info(f"\n  ● GOOD — AUC {test_auc:.4f} >= 0.70 (clears gate)")
    elif test_auc >= 0.65:
        logger.info(f"\n  ○ FAIR — AUC {test_auc:.4f} >= 0.65 (below gate, investigate labels)")
    else:
        logger.info(f"\n  ✗ NEEDS WORK — AUC {test_auc:.4f} < 0.65")

    logger.info("=" * 80)


if __name__ == '__main__':
    main()
