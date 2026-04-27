"""
TradeSage - Large Scale Training Module v2
Train on 1200+ stocks with parallel processing, caching, and progress tracking.
Supports ensemble mode and Optuna tuning.
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import joblib
import os
from pathlib import Path
from datetime import datetime
import json

from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer


class LargeScaleTrainer:
    """Train TradeSage on large datasets (100-1200+ stocks)"""

    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.engineer = FeatureEngineer()

    def fetch_with_cache(self, symbol, period='20y', force_refresh=False):
        """Fetch data with disk caching to avoid re-downloading"""
        cache_options = [
            self.cache_dir / f"{symbol}_{period}.pkl",
            self.cache_dir / f"{symbol}_daily.csv"
        ]

        if not force_refresh:
            for cache_file in cache_options:
                if cache_file.exists():
                    try:
                        if cache_file.suffix == '.pkl':
                            return joblib.load(cache_file)
                        else:
                            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                            if df.index.tz is not None:
                                df.index = df.index.tz_localize(None)
                            return df
                    except Exception as e:
                        print(f"Warning: Failed to load cache {cache_file}: {e}")

        if not hasattr(self, 'fetcher'):
            from data_fetcher import MarketDataFetcher
            self.fetcher = MarketDataFetcher()

        df = self.fetcher.fetch_stock_data(symbol, period=period)

        if df is not None:
            save_path = self.cache_dir / f"{symbol}_{period}.pkl"
            joblib.dump(df, save_path)

        return df

    def fetch_stocks_parallel(self, symbols, period='20y', max_workers=10):
        """Fetch multiple stocks in parallel"""
        print(f"\n📥 Fetching data for {len(symbols)} stocks (parallel)...")

        results = {}
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_with_cache, symbol, period): symbol
                for symbol in symbols
            }

            with tqdm(total=len(symbols), desc="Downloading", unit="stock") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if df is not None and len(df) > 500:
                            results[symbol] = df
                        else:
                            failed.append(symbol)
                    except Exception:
                        failed.append(symbol)
                    pbar.update(1)

        print(f"✓ Successfully fetched: {len(results)} stocks")
        if failed:
            print(f"⚠ Failed to fetch: {len(failed)} stocks")
            if len(failed) <= 10:
                print(f"  Failed symbols: {', '.join(failed)}")

        return results, failed

    def prepare_training_data(self, stock_data_dict, forward_days=5, threshold=0.04,
                              max_rows_per_stock=2000, index_df=None,
                              max_drawdown=-0.03):
        """
        Prepare training data from multiple stocks.
        Now passes index_df through for market context features.
        v5: Also injects fundamental features via Obscura+Screener batch fetch.
        """
        print(f"\n🔧 Engineering features for {len(stock_data_dict)} stocks...")
        if max_rows_per_stock:
            print(f"   (capped at {max_rows_per_stock} most-recent rows per stock)")

        # ── Batch-fetch fundamentals for all symbols (uses disk cache) ──
        fund_cache = {}
        try:
            from src.core.screener_scraper import ScreenerScraper
            scraper = ScreenerScraper()
            symbols_list = list(stock_data_dict.keys())
            print(f"\n📊 Fetching fundamentals for {len(symbols_list)} stocks (cached where possible)...")
            fund_cache = scraper.fetch_fundamentals_batch(symbols_list, delay=1.0)
            print(f"   ✓ Got fundamentals for {len(fund_cache)}/{len(symbols_list)} stocks")
        except Exception as e:
            print(f"   ⚠ Fundamental fetch failed: {e} — training continues with technical features only")

        # Pre-load the cache into the FeatureEngineer
        self.engineer.set_fundamentals_cache(fund_cache)

        all_data = []
        failed_stocks = []

        with tqdm(total=len(stock_data_dict), desc="Processing", unit="stock") as pbar:
            for symbol, df in stock_data_dict.items():
                try:
                    if max_rows_per_stock and len(df) > max_rows_per_stock:
                        df = df.iloc[-max_rows_per_stock:]

                    df_features = self.engineer.add_technical_indicators(
                        df, index_df=index_df, symbol=symbol
                    )

                    df_final = self.engineer.create_target_variable(
                        df_features,
                        forward_days=forward_days,
                        gain_threshold=threshold,
                        max_drawdown=max_drawdown,
                    )

                    df_final['symbol'] = symbol
                    all_data.append(df_final)

                except ValueError as e:
                    failed_stocks.append((symbol, str(e)))
                except Exception as e:
                    failed_stocks.append((symbol, str(e)))

                pbar.update(1)

        if failed_stocks:
            print(f"\n⚠ Skipped {len(failed_stocks)} stocks due to data issues")
            if len(failed_stocks) <= 20:
                print("Failed stocks:", ", ".join([s[0] for s in failed_stocks]))

        if len(all_data) == 0:
            raise ValueError("No valid stock data after processing!")

        combined_df = pd.concat(all_data, ignore_index=False)

        X, y, feature_names = self.engineer.prepare_training_data(combined_df)

        # Count how many fundamental features made it in
        fund_feats = [f for f in feature_names if f.startswith('fund_')]
        print(f"\n✓ Successfully processed: {len(all_data)}/{len(stock_data_dict)} stocks")
        print(f"✓ Total training samples: {len(X):,}")
        print(f"✓ Features: {len(feature_names)} ({len(fund_feats)} fundamental)")
        print(f"✓ Positive samples: {y.sum():,} ({y.mean()*100:.1f}%)")
        print(f"✓ Date range: {combined_df.index.min()} to {combined_df.index.max()}")

        return X, y, feature_names, combined_df

    def train_large_scale(self, symbols, model_path='tradesage_large.pkl',
                         period='20y', forward_days=5, threshold=0.04,
                         max_workers=10, save_progress=True,
                         max_rows_per_stock=2000, use_ensemble=True,
                         index_df=None, max_drawdown=-0.03):
        """
        Train on large dataset with all optimizations.
        """
        print("="*80)
        print("🚀 TRADESAGE LARGE-SCALE TRAINING v2")
        print("="*80)
        print(f"Stocks to train: {len(symbols)}")
        print(f"Data period: {period}")
        print(f"Prediction horizon: {forward_days} days")
        print(f"Return threshold: {threshold*100}%")
        print(f"Parallel workers: {max_workers}")
        print(f"Ensemble mode: {'ON' if use_ensemble else 'OFF'}")
        print(f"Max drawdown: {max_drawdown*100}%")

        # Step 1: Fetch data
        stock_data, failed = self.fetch_stocks_parallel(
            symbols, period=period, max_workers=max_workers
        )

        if len(stock_data) < 10:
            print("\n❌ Not enough data fetched.")
            return None

        if save_progress:
            progress_file = self.cache_dir / 'fetch_progress.json'
            with open(progress_file, 'w') as f:
                json.dump({
                    'total': len(symbols),
                    'successful': len(stock_data),
                    'failed': failed,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

        # Step 2: Engineer features
        X, y, feature_names, combined_df = self.prepare_training_data(
            stock_data,
            forward_days=forward_days,
            threshold=threshold,
            max_rows_per_stock=max_rows_per_stock,
            index_df=index_df,
            max_drawdown=max_drawdown,
        )

        if save_progress:
            data_file = self.cache_dir / 'processed_data.pkl'
            print(f"\n💾 Saving processed data to {data_file}...")
            joblib.dump({
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'metadata': {
                    'stocks': len(stock_data),
                    'samples': len(X),
                    'features': len(feature_names),
                    'positive_rate': y.mean(),
                    'date_range': (str(combined_df.index.min()), str(combined_df.index.max()))
                }
            }, data_file)

        # Step 3: Walk-forward split (90/5/5 by date)
        dates = combined_df.index.unique().sort_values()
        n_dates = len(dates)
        val_start_date  = dates[int(n_dates * 0.90)]
        test_start_date = dates[int(n_dates * 0.95)]

        train_mask = X.index < val_start_date
        val_mask   = (X.index >= val_start_date) & (X.index < test_start_date)
        test_mask  = X.index >= test_start_date

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        print(f"\n📊 Walk-Forward Split (90 / 5 / 5 by date):")
        print(f"  Train : {len(X_train):,} samples  (up to {val_start_date.date()})")
        print(f"  Val   : {len(X_val):,} samples  ({val_start_date.date()} → {test_start_date.date()})")
        print(f"  Test  : {len(X_test):,} samples  ({test_start_date.date()} → end)  [held-out]")

        # Step 4: Train
        print(f"\n🤖 Training model...")
        trainer = TradingModelTrainer()
        trainer.train_model(X_train, y_train, X_val, y_val, use_ensemble=use_ensemble)

        # Step 5: Evaluate
        print(f"\n📈 Evaluating on val set...")
        metrics = trainer.evaluate_model(X_val, y_val)

        print(f"\n📈 Evaluating on TEST set (held-out)...")
        test_metrics = trainer.evaluate_model(X_test, y_test) if len(X_test) > 0 else {}

        # Step 6: Feature importance
        print(f"\n🔍 Feature importance analysis...")
        importance_df = trainer.get_feature_importance(top_n=20)

        # Step 7: Save
        print(f"\n💾 Saving model to {model_path}...")
        trainer.save_model(model_path)

        # Training report
        if save_progress:
            report_file = model_path.replace('.pkl', '_report.json')
            safe_test_metrics = {}
            for k, v in test_metrics.items():
                if k != 'confusion_matrix':
                    safe_test_metrics[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
            safe_val_metrics = {}
            for k, v in metrics.items():
                if k != 'confusion_matrix':
                    safe_val_metrics[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v

            with open(report_file, 'w') as f:
                json.dump({
                    'training_date': datetime.now().isoformat(),
                    'version': 'v2',
                    'stocks_trained': len(stock_data),
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples':   len(X_val),
                    'test_samples':  len(X_test),
                    'split': {
                        'train_until': str(val_start_date.date()),
                        'val_until':   str(test_start_date.date()),
                        'test_from':   str(test_start_date.date()),
                    },
                    'features': len(feature_names),
                    'ensemble_mode': use_ensemble,
                    'val_metrics': safe_val_metrics,
                    'test_metrics': safe_test_metrics,
                    'parameters': {
                        'period': period,
                        'forward_days': forward_days,
                        'threshold': threshold,
                        'max_rows_per_stock': max_rows_per_stock,
                    },
                    'top_features': importance_df.head(10).to_dict('records')
                }, f, indent=2)
            print(f"✓ Training report saved to {report_file}")

        print("\n" + "="*80)
        print("✅ LARGE-SCALE TRAINING v2 COMPLETE!")
        print("="*80)

        return trainer, metrics

    def load_processed_data(self):
        """Load previously processed data for faster retraining"""
        data_file = self.cache_dir / 'processed_data.pkl'

        if not data_file.exists():
            print("❌ No processed data found. Run full training first.")
            return None

        print(f"📂 Loading processed data from {data_file}...")
        data = joblib.load(data_file)

        print("✓ Loaded:")
        print(f"  Samples: {len(data['X']):,}")
        print(f"  Features: {len(data['feature_names'])}")
        print(f"  Positive rate: {data['metadata']['positive_rate']*100:.1f}%")

        return data['X'], data['y'], data['feature_names']

    def quick_retrain(self, model_path='tradesage_large.pkl', use_ensemble=False):
        """Quickly retrain on cached processed data"""
        data = self.load_processed_data()

        if data is None:
            return None

        X, y, feature_names = data

        # Walk-forward split
        dates = X.index.unique().sort_values()
        n_dates = len(dates)
        val_start_date  = dates[int(n_dates * 0.90)]
        test_start_date = dates[int(n_dates * 0.95)]

        train_mask = X.index < val_start_date
        val_mask   = (X.index >= val_start_date) & (X.index < test_start_date)
        test_mask  = X.index >= test_start_date

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,} [held-out]")

        print("\n🤖 Training model on cached data...")
        trainer = TradingModelTrainer()
        trainer.train_model(X_train, y_train, X_val, y_val, use_ensemble=use_ensemble)

        metrics      = trainer.evaluate_model(X_val, y_val)
        test_metrics = trainer.evaluate_model(X_test, y_test) if len(X_test) > 0 else {}
        if test_metrics:
            print(f"\n📈 TEST (held-out) AUC: {test_metrics.get('auc_score', 0):.4f}  "
                  f"Precision: {test_metrics.get('precision', 0):.4f}  "
                  f"ProfitScore: {test_metrics.get('profit_score', 0):.4f}")

        trainer.save_model(model_path)

        return trainer, metrics


def load_stock_list_from_file(filepath):
    """Load stock symbols from various file formats"""
    filepath = Path(filepath)

    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        symbols = df.iloc[:, 0].tolist()
    elif filepath.suffix == '.txt':
        with open(filepath, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                symbols = data
            elif isinstance(data, dict):
                symbols = data.get('symbols', data.get('stocks', data.get('data', [])))
            else:
                raise ValueError("JSON must be a list or object with 'symbols' key")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    symbols = [s.replace('.NS', '').replace('.BO', '').strip().upper() for s in symbols]

    return symbols


if __name__ == "__main__":
    print("="*80)
    print("LARGE-SCALE TRADESAGE TRAINING v2")
    print("="*80)

    test_symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
        'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'BAJFINANCE', 'HCLTECH'
    ]

    print(f"\nTraining on {len(test_symbols)} stocks for demo...")

    trainer = LargeScaleTrainer(cache_dir='data_cache')

    model, metrics = trainer.train_large_scale(
        symbols=test_symbols,
        model_path='tradesage_large.pkl',
        period='20y',
        forward_days=5,
        threshold=0.04,
        max_workers=10,
        use_ensemble=True,
        max_drawdown=-0.03,
    )

    if model:
        print(f"\n✅ Training successful!")
        print(f"Model saved to: tradesage_large.pkl")
        print(f"AUC Score: {metrics.get('auc_score', 0):.4f}")
        print(f"Win Rate: {metrics.get('predicted_win_rate', 0)*100:.2f}%")
        print(f"Profit Score: {metrics.get('profit_score', 0):.4f}")