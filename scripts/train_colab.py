import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train_from_cache(cache_dir='data_cache_yfinance', model_path='models/tradesage_colab.pkl'):
    """
    Trains TradeSage using pre-downloaded CSVs.
    Perfect for Google Colab and 10+ years of yfinance data.
    """
    cache_path = Path(PROJECT_ROOT) / cache_dir
    model_out = Path(PROJECT_ROOT) / model_path
    
    logger.info("="*80)
    logger.info(f" TRADESAGE COLAB LARGE-SCALE TRAINING ({cache_dir})")
    logger.info("="*80)
    
    if not cache_path.exists():
        logger.error(f"Cache directory {cache_dir} not found! Run fetch_yfinance_10y.py first.")
        return
        
    data_files = list(cache_path.glob('*_daily.csv'))
    if not data_files:
        logger.error(f"No CSVs found in {cache_dir}!")
        return

    logger.info(f"Found {len(data_files)} stocks in cache. Loading and engineering features...")
    
    engineer = FeatureEngineer()
    all_data = []
    
    for file in data_files:
        symbol = file.stem.replace('_daily', '')
        try:
            df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            if len(df) < 200:
                continue
                
            # 1. Add indicators
            df_features = engineer.add_technical_indicators(df)
            
            # 2. Add Target
            df_final = engineer.create_target_variable(
                df_features, 
                forward_days=10, 
                gain_threshold=0.05
            )
            df_final['symbol'] = symbol
            all_data.append(df_final)
            
        except Exception as e:
            pass
            
    if not all_data:
        logger.error("No valid features could be extracted!")
        return
        
    logger.info(f"\nSuccessfully engineered features for {len(all_data)} stocks.")
    logger.info("Splitting each stock 80/20 chronologically (Train/Val)...")
    
    train_parts, val_parts = [], []
    for df in all_data:
        n = len(df)
        sp = int(n * 0.8)
        train_parts.append(df.iloc[:sp])
        val_parts.append(df.iloc[sp:])

    stocks_trained_count = len(all_data)
    del all_data # Free up original memory
    import gc
    gc.collect()
    
    train_df = pd.concat(train_parts)
    val_df   = pd.concat(val_parts)
    del train_parts, val_parts # Free list overhead
    gc.collect()

    logger.info("Downcasting float64 to float32 to save 50% RAM...")
    for col in train_df.select_dtypes(include=['float64']).columns:
        train_df[col] = train_df[col].astype('float32')
        if col in val_df.columns:
            val_df[col] = val_df[col].astype('float32')

    X_train, y_train, feature_names = engineer.prepare_training_data(train_df)
    del train_df
    gc.collect()
    
    X_val,   y_val,   _             = engineer.prepare_training_data(val_df)
    del val_df
    gc.collect()

    # Align columns
    for col in set(feature_names) - set(X_val.columns):
        X_val[col] = 0
    X_val = X_val[feature_names]

    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Train: {len(X_train):,} samples  (pos={y_train.mean()*100:.1f}%)")
    logger.info(f"Val:   {len(X_val):,} samples  (pos={y_val.mean()*100:.1f}%)")
    
    # Train
    trainer = TradingModelTrainer()
    trainer.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    logger.info("\nEvaluating model...")
    metrics = trainer.evaluate_model(X_val, y_val)
    
    # Save
    os.makedirs(model_out.parent, exist_ok=True)
    trainer.save_model(str(model_out))
    
    # Save Feature Importance Report
    report_file = str(model_out).replace('.pkl', '_report.json')
    try:
        importance_df = trainer.get_feature_importance(top_n=15)
        top_features = importance_df.head(15).to_dict('records')
    except:
        top_features = []
        
    with open(report_file, 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'stocks_trained': stocks_trained_count,
            'total_samples': len(X_train) + len(X_val),
            'features': len(feature_names),
            'metrics': {k: float(v) if not isinstance(v, list) else v for k, v in metrics.items() if k != 'confusion_matrix'},
            'top_features': top_features
        }, f, indent=2)
        
    logger.info(f"Training report saved to {report_file}")
    logger.info("COLAB TRAINING COMPLETE!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data_cache_yfinance", help="Folder containing raw CSVs")
    parser.add_argument("--model", default="models/tradesage_colab.pkl", help="Output model path")
    args = parser.parse_args()
    
    train_from_cache(cache_dir=args.cache, model_path=args.model)
