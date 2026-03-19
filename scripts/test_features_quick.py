"""
Quick offline test  uses existing data_cache_angel CSVs.
No API call needed. Tests feature engineering + model training.
Run: python scripts/test_features_quick.py
"""

import os, sys, glob, json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer

CACHE_DIR  = os.path.join(PROJECT_ROOT, 'data_cache_angel')
N_STOCKS   = 2000
MIN_ROWS   = 250
REPORT_OUT = os.path.join(PROJECT_ROOT, 'models', 'test_report.json')

# ETF/index keywords to skip  we want real stocks only
SKIP_KEYWORDS = ['BEES','IETF','BETA','CASE','ADD','ETF','NIFTY','SENSEX',
                 'GOLD','SILVER','LIQUID','GILT','BOND','NSETEST']

def load_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower().strip() for c in df.columns]
    needed = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed].dropna()
    if len(df) < MIN_ROWS:
        return None
    return df

def main():
    print("="*60)
    print("TRADESAGE  OFFLINE FEATURE + MODEL TEST")
    print("="*60)

    csvs = sorted(glob.glob(os.path.join(CACHE_DIR, '*_daily.csv')))
    csvs = [f for f in csvs if not any(k in os.path.basename(f).upper() for k in SKIP_KEYWORDS)]
    csvs = csvs[:N_STOCKS]
    print(f"\nLoading up to {len(csvs)} stock CSVs...")

    engineer = FeatureEngineer()
    all_data, skipped = [], 0

    for path in csvs:
        symbol = os.path.basename(path).replace('_daily.csv', '')
        df = load_csv(path)
        if df is None:
            skipped += 1
            continue
        try:
            df = engineer.add_technical_indicators(df)
            df = engineer.create_target_variable(df, forward_days=10, gain_threshold=0.05)
            df['symbol'] = symbol
            all_data.append(df)
        except Exception:
            skipped += 1

    print(f"Processed: {len(all_data)} stocks  |  Skipped: {skipped}")
    if len(all_data) < 5:
        print("Not enough data. Exiting.")
        return

    # Per-stock 80/20 chronological split  prevents market-regime leakage
    print("Splitting each stock 80/20 chronologically...")
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

    print(f"\nFeature count : {len(feature_names)}")
    print(f"Train samples : {len(X_train):,}  (pos={y_train.mean()*100:.1f}%)")
    print(f"Val samples   : {len(X_val):,}  (pos={y_val.mean()*100:.1f}%)")

    # Train
    trainer = TradingModelTrainer()
    trainer.train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = trainer.evaluate_model(X_val, y_val)

    # Feature importance
    imp_df = trainer.get_feature_importance(top_n=15)

    # Save report
    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
    report = {
        'stocks_used':    len(all_data),
        'train_samples':  int(len(X_train)),
        'val_samples':    int(len(X_val)),
        'features':       len(feature_names),
        'train_pos_rate': round(float(y_train.mean()), 4),
        'val_pos_rate':   round(float(y_val.mean()), 4),
        'metrics':        {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
        'top_features':   imp_df.head(15).to_dict('records'),
    }
    with open(REPORT_OUT, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n Report saved  {REPORT_OUT}")
    print("\n" + "="*60)
    auc = metrics['auc_score']
    if auc >= 0.75:
        print(f"TARGET HIT! AUC = {auc:.4f}")
    elif auc >= 0.65:
        print(f"Good progress. AUC = {auc:.4f}  (target: 0.75)")
    else:
        print(f"AUC = {auc:.4f}   more data needed (run full train_angel_one.py)")
    print("="*60)

if __name__ == '__main__':
    main()

