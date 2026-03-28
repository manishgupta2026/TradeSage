
"""
Train TradeSage with Angel One Data
Automates fetching the top NSE stocks, calling Angel One for daily OHLCV data,
and passing them to the Feature Engineer and XGBoost model.
"""

import json
import logging
import os
import sys
import pandas as pd
from datetime import datetime
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.angel_one_api import AngelOneAPI
from data.angel_data_fetcher import AngelDataFetcher
from features.technical_indicators import FeatureEngineer
from models.train_xgboost import TradingModelTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Training logic
def train_on_angel_one(symbols, config):
    logger.info("="*80)
    logger.info(" TRADESAGE ANGEL ONE LARGE-SCALE TRAINING")
    logger.info("="*80)
    
    # 1. Setup API
    try:
        api = AngelOneAPI(os.path.join(PROJECT_ROOT, 'config', 'angel_config.json'))
    except Exception as e:
        logger.error(f"Cannot connect to Angel One: {e}")
        return
        
    fetcher = AngelDataFetcher(api)
    engineer = FeatureEngineer()
    trainer = TradingModelTrainer()

    # 2. Fetch Data
    # 3 years = roughly 1095 days
    period_days = 1095 if config['period'] == '3y' else 730
    stock_data, failed = fetcher.fetch_multiple_symbols(
        symbols, 
        period_days=period_days, 
        max_workers=config['max_workers']
    )
    
    if len(stock_data) < 10:
        logger.error("\n Not enough valid data fetched. Exiting.")
        return
        
    # 3. Engineer Features
    logger.info(f"\n Engineering features for {len(stock_data)} stocks...")
    all_data = []
    
    for symbol, df in stock_data.items():
        try:
            df_features = engineer.add_technical_indicators(df)
            df_final = engineer.create_target_variable(
                df_features, 
                forward_days=config['forward_days'],
                gain_threshold=config.get('gain_threshold', 0.04)
            )
            df_final['symbol'] = symbol
            all_data.append(df_final)
        except Exception as e:
            # We skip silent errors
            pass
            
    if not all_data:
        logger.error("No valid stock data after processing!")
        return
        
    # Per-stock 80/20 chronological split  prevents market-regime leakage
    # (splitting globally by date means val = recent bear market only  biased AUC)
    logger.info(f"\n Successfully processed: {len(all_data)} stocks")
    logger.info(" Splitting each stock 80/20 chronologically...")

    train_parts, val_parts = [], []
    for df in all_data:
        n = len(df)
        sp = int(n * 0.8)
        train_parts.append(df.iloc[:sp])
        val_parts.append(df.iloc[sp:])

    train_df = pd.concat(train_parts)
    val_df   = pd.concat(val_parts)

    X_train, y_train, feature_names = engineer.prepare_training_data(train_df)
    X_val,   y_val,   _             = engineer.prepare_training_data(val_df)

    # Align val columns to train
    for col in set(feature_names) - set(X_val.columns):
        X_val[col] = 0
    X_val = X_val[feature_names]

    logger.info(f" Features: {len(feature_names)}")
    logger.info(f" Train: {len(X_train):,} samples  (pos={y_train.mean()*100:.1f}%)")
    logger.info(f" Val:   {len(X_val):,} samples  (pos={y_val.mean()*100:.1f}%)")
    
    # 4. Train Model
    trainer.train_model(X_train, y_train, X_val, y_val)
    
    # 5. Evaluate and Save
    logger.info("\n Evaluating model...")
    metrics = trainer.evaluate_model(X_val, y_val)
    
    logger.info(f"\n Saving model to {config['model_path']}...")
    trainer.save_model(config['model_path'])
    
    # Save training report
    report_file = config['model_path'].replace('.pkl', '_report.json')
    try:
        importance_df = trainer.get_feature_importance(top_n=10)
        top_features = importance_df.head(10).to_dict('records')
    except:
        top_features = []
        
    with open(report_file, 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'stocks_trained': len(stock_data),
            'total_samples': len(X_train) + len(X_val),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'features': len(feature_names),
            'metrics': {
                'auc_score': float(metrics.get('auc_score', 0)),
                'predicted_win_rate': float(metrics.get('predicted_win_rate', 0)),
                'accuracy': float(metrics.get('accuracy', 0)),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'f1': float(metrics.get('f1', 0))
            },
            'parameters': config,
            'top_features': top_features
        }, f, indent=2)
    logger.info(f" Training report saved to {report_file}")
    
    logger.info("\n" + "="*80)
    logger.info(" LARGE-SCALE TRAINING COMPLETE!")
    logger.info("="*80)

def menu():
    print("\nOptions:")
    print("1. Get top 1500 NSE stocks (from Angel One)")
    print("2. Train on Angel One data")
    print("3. Full pipeline (get stocks + train)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    config = {
        'period': '3y',
        'forward_days': 10,        # look-ahead window for max-high target
        'gain_threshold': 0.05,    # 5% gain achievable in 10 days  ~40-45% positive rate
        'tp_mult': 2.0,            # kept for API compat, unused by new target
        'sl_mult': 2.0,
        'max_workers': 5,
        'model_path': os.path.join(PROJECT_ROOT, 'models', 'tradesage_angel.pkl')
    }

    if choice in ['1', '3']:
        try:
            api = AngelOneAPI(os.path.join(PROJECT_ROOT, 'config', 'angel_config.json'))
        except Exception as e:
            print(" Cannot connect to API. Please setup angel_config.json")
            return
            
        fetcher = AngelDataFetcher(api)
        # Fetching 3000 to drastically increase training dataset size
        symbols = fetcher.get_top_nse_stocks(count=3000, save_path=os.path.join(PROJECT_ROOT, 'data', 'nse_top_3000_angel.json'))
    
    if choice in ['2', '3']:
        if choice == '2':
            try:
                with open(os.path.join(PROJECT_ROOT, 'data', 'nse_top_3000_angel.json'), 'r') as f:
                    symbols = json.load(f)
            except:
                print(" Stock list not found. Please run option 1 first.")
                return
        
        train_on_angel_one(symbols, config)

if __name__ == "__main__":
    menu()

