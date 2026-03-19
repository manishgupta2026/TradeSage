"""
TradeSage - Large Scale Training Module
Train on 1200+ stocks with parallel processing, caching, and progress tracking
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

from data_fetcher import MarketDataFetcher
from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer

class LargeScaleTrainer:
    """Train TradeSage on large datasets (100-1200+ stocks)"""
    
    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.fetcher = MarketDataFetcher()
        self.engineer = FeatureEngineer()
        
    def fetch_with_cache(self, symbol, period='2y', force_refresh=False):
        """Fetch data with disk caching to avoid re-downloading"""
        cache_file = self.cache_dir / f"{symbol}_{period}.pkl"
        
        # Check cache
        if cache_file.exists() and not force_refresh:
            # Check if cache is less than 1 day old
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                return joblib.load(cache_file)
        
        # Fetch fresh data
        df = self.fetcher.fetch_stock_data(symbol, period=period)
        
        # Cache it
        if df is not None:
            joblib.dump(df, cache_file)
        
        return df
    
    def fetch_stocks_parallel(self, symbols, period='2y', max_workers=10):
        """Fetch multiple stocks in parallel"""
        print(f"\n📥 Fetching data for {len(symbols)} stocks (parallel)...")
        
        results = {}
        failed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_with_cache, symbol, period): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(symbols), desc="Downloading", unit="stock") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if df is not None and len(df) > 500:  # Minimum 500 rows (~2 years of data)
                            results[symbol] = df
                        else:
                            failed.append(symbol)
                    except Exception as e:
                        failed.append(symbol)
                    
                    pbar.update(1)
        
        print(f"✓ Successfully fetched: {len(results)} stocks")
        if failed:
            print(f"⚠ Failed to fetch: {len(failed)} stocks")
            if len(failed) <= 10:
                print(f"  Failed symbols: {', '.join(failed)}")
        
        return results, failed
    
    def prepare_training_data(self, stock_data_dict, forward_days=5, threshold=0.02):
        """Prepare training data from multiple stocks"""
        print(f"\n🔧 Engineering features for {len(stock_data_dict)} stocks...")
        
        all_data = []
        failed_stocks = []
        
        with tqdm(total=len(stock_data_dict), desc="Processing", unit="stock") as pbar:
            for symbol, df in stock_data_dict.items():
                try:
                    # Add technical indicators
                    df_features = self.engineer.add_technical_indicators(df)
                    
                    # Create target
                    df_final = self.engineer.create_target_variable(
                        df_features, 
                        forward_days=forward_days, 
                        threshold=threshold
                    )
                    
                    # Add symbol identifier
                    df_final['symbol'] = symbol
                    
                    all_data.append(df_final)
                    
                except ValueError as e:
                    # Handle data validation errors silently
                    failed_stocks.append((symbol, str(e)))
                except Exception as e:
                    # Print error for unexpected issues
                    print(f"\n⚠ Error processing {symbol}: {e}")
                    failed_stocks.append((symbol, str(e)))
                
                pbar.update(1)
        
        # Report failed stocks if significant
        if failed_stocks:
            print(f"\n⚠ Skipped {len(failed_stocks)} stocks due to data issues")
            if len(failed_stocks) <= 20:
                print("Failed stocks:", ", ".join([s[0] for s in failed_stocks]))
        
        if len(all_data) == 0:
            raise ValueError("No valid stock data after processing!")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=False)
        
        # Prepare features
        X, y, feature_names = self.engineer.prepare_training_data(combined_df)
        
        print(f"\n✓ Successfully processed: {len(all_data)}/{len(stock_data_dict)} stocks")
        print(f"✓ Total training samples: {len(X):,}")
        print(f"✓ Features: {len(feature_names)}")
        print(f"✓ Positive samples: {y.sum():,} ({y.mean()*100:.1f}%)")
        print(f"✓ Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        
        return X, y, feature_names, combined_df
    
    def train_large_scale(self, symbols, model_path='tradesage_large.pkl', 
                         period='2y', forward_days=5, threshold=0.02,
                         max_workers=10, save_progress=True):
        """
        Train on large dataset with all optimizations
        
        Args:
            symbols: List of stock symbols (can be 1200+)
            model_path: Where to save trained model
            period: Data period ('1y', '2y', '3y')
            forward_days: Prediction horizon
            threshold: Minimum return for positive label
            max_workers: Parallel download threads
            save_progress: Save intermediate results
        """
        print("="*80)
        print("🚀 TRADESAGE LARGE-SCALE TRAINING")
        print("="*80)
        print(f"Stocks to train: {len(symbols)}")
        print(f"Data period: {period}")
        print(f"Prediction horizon: {forward_days} days")
        print(f"Return threshold: {threshold*100}%")
        print(f"Parallel workers: {max_workers}")
        
        # Step 1: Fetch data in parallel
        stock_data, failed = self.fetch_stocks_parallel(
            symbols, 
            period=period, 
            max_workers=max_workers
        )
        
        if len(stock_data) < 10:
            print("\n❌ Not enough data fetched. Check your internet connection.")
            return None
        
        # Save fetch results
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
            threshold=threshold
        )
        
        # Save processed data (for faster retraining)
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
        
        # Step 3: Split data (time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"\n📊 Data Split:")
        print(f"  Training: {len(X_train):,} samples")
        print(f"  Validation: {len(X_val):,} samples")
        
        # Step 4: Train model
        print(f"\n🤖 Training XGBoost model...")
        trainer = TradingModelTrainer()
        trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Step 5: Evaluate
        print(f"\n📈 Evaluating model...")
        metrics = trainer.evaluate_model(X_val, y_val)
        
        # Step 6: Feature importance
        print(f"\n🔍 Feature importance analysis...")
        importance_df = trainer.get_feature_importance(top_n=20)
        
        # Step 7: Save model
        print(f"\n💾 Saving model to {model_path}...")
        trainer.save_model(model_path)
        
        # Save training report
        if save_progress:
            report_file = model_path.replace('.pkl', '_report.json')
            with open(report_file, 'w') as f:
                json.dump({
                    'training_date': datetime.now().isoformat(),
                    'stocks_trained': len(stock_data),
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'features': len(feature_names),
                    'metrics': {
                        'auc_score': float(metrics.get('auc_score', 0)),
                        'predicted_win_rate': float(metrics.get('predicted_win_rate', 0)),
                        'actual_win_rate': float(metrics.get('actual_win_rate', 0))
                    },
                    'parameters': {
                        'period': period,
                        'forward_days': forward_days,
                        'threshold': threshold
                    },
                    'top_features': importance_df.head(10).to_dict('records')
                }, f, indent=2)
            print(f"✓ Training report saved to {report_file}")
        
        print("\n" + "="*80)
        print("✅ LARGE-SCALE TRAINING COMPLETE!")
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
    
    def quick_retrain(self, model_path='tradesage_large.pkl'):
        """Quickly retrain on cached processed data"""
        data = self.load_processed_data()
        
        if data is None:
            return None
        
        X, y, feature_names = data
        
        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train
        print("\n🤖 Training model on cached data...")
        trainer = TradingModelTrainer()
        trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = trainer.evaluate_model(X_val, y_val)
        
        # Save
        trainer.save_model(model_path)
        
        return trainer, metrics


def load_stock_list_from_file(filepath):
    """Load stock symbols from various file formats"""
    filepath = Path(filepath)
    
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        # Assume first column has symbols
        symbols = df.iloc[:, 0].tolist()
    elif filepath.suffix == '.txt':
        with open(filepath, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, list):
                symbols = data
            elif isinstance(data, dict):
                # Try different possible keys
                symbols = data.get('symbols', data.get('stocks', data.get('data', [])))
            else:
                raise ValueError("JSON must be a list or object with 'symbols' key")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Clean symbols (remove .NS suffix if present, we'll add it back)
    symbols = [s.replace('.NS', '').replace('.BO', '').strip().upper() for s in symbols]
    
    return symbols


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("LARGE-SCALE TRADESAGE TRAINING")
    print("="*80)
    
    # Option 1: Train on list of symbols
    print("\nOption 1: Enter symbols manually")
    print("Example: Use Nifty 500 stocks")
    
    # For testing, use a subset
    test_symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
        'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'BAJFINANCE', 'HCLTECH'
    ]
    
    print(f"\nTraining on {len(test_symbols)} stocks for demo...")
    
    # Initialize trainer
    trainer = LargeScaleTrainer(cache_dir='data_cache')
    
    # Train
    model, metrics = trainer.train_large_scale(
        symbols=test_symbols,
        model_path='tradesage_large.pkl',
        period='2y',
        forward_days=5,
        threshold=0.02,
        max_workers=10
    )
    
    if model:
        print("\n✅ Training successful!")
        print(f"Model saved to: tradesage_large.pkl")
        print(f"AUC Score: {metrics.get('auc_score', 0):.4f}")
        print(f"Win Rate: {metrics.get('predicted_win_rate', 0)*100:.2f}%")