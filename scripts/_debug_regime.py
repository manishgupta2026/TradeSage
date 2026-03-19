"""Check if the val positive rate drop is a regime issue"""
import pandas as pd, sys, os, glob
sys.path.append('.')
from src.core.feature_engineering import FeatureEngineer

CACHE_DIR = 'data_cache_angel'
SKIP = ['BEES','IETF','BETA','CASE','ADD','ETF','NIFTY','SENSEX','GOLD','SILVER','LIQUID','GILT','BOND','NSETEST']

csvs = sorted(glob.glob(os.path.join(CACHE_DIR, '*_daily.csv')))
csvs = [f for f in csvs if not any(k in os.path.basename(f).upper() for k in SKIP)][:200]

eng = FeatureEngineer()
train_pos, val_pos = [], []
date_ranges = []

for path in csvs[:50]:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if len(df) < 250: continue
    try:
        df = eng.add_technical_indicators(df)
        df = eng.create_target_variable(df, forward_days=15, tp_mult=2.0, sl_mult=2.0)
        df = df.dropna()
        n = len(df)
        sp = int(n * 0.8)
        train_pos.append(df.iloc[:sp]['target'].mean())
        val_pos.append(df.iloc[sp:]['target'].mean())
        date_ranges.append((df.index.min(), df.index[sp], df.index.max()))
    except: pass

import numpy as np
print(f"Train positive rate: {np.mean(train_pos)*100:.1f}%")
print(f"Val positive rate:   {np.mean(val_pos)*100:.1f}%")
print(f"\nSample date ranges (first 5):")
for i, (start, split, end) in enumerate(date_ranges[:5]):
    print(f"  {start.date()} → split: {split.date()} → {end.date()}")
