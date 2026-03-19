"""
Test different target definitions to find one with stronger signal.
The goal: find a target where individual features have AUC > 0.56
"""
import pandas as pd, numpy as np, sys, os, glob
sys.path.append('.')
from src.core.feature_engineering import FeatureEngineer
from sklearn.metrics import roc_auc_score

CACHE = 'data_cache_angel'
SKIP  = ['BEES','IETF','BETA','CASE','ADD','ETF','NIFTY','SENSEX',
         'GOLD','SILVER','LIQUID','GILT','BOND','NSETEST']

csvs = [f for f in sorted(glob.glob(os.path.join(CACHE,'*_daily.csv')))
        if not any(k in os.path.basename(f).upper() for k in SKIP)][:200]

eng = FeatureEngineer()
all_dfs = []
for path in csvs:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if len(df) < 250: continue
    try:
        df = eng.add_technical_indicators(df)
        all_dfs.append(df)
    except: pass

combined = pd.concat(all_dfs).dropna(subset=['close','atr','rsi_14','adx'])
print(f"Base rows: {len(combined):,}")

def test_target(name, target_series, X):
    # reset index to avoid duplicate label issues from multi-stock concat
    y = target_series.reset_index(drop=True).dropna()
    X2 = X.reset_index(drop=True).loc[y.index]
    if len(y) < 100 or y.nunique() < 2: return
    pos = y.mean()*100
    top_feats = ['atr_pct','above_sma50','full_uptrend','dist_sma50','macd_norm']
    aucs = []
    for f in top_feats:
        if f not in X2.columns: continue
        try:
            a = roc_auc_score(y, X2[f])
            aucs.append(max(a, 1-a))
        except: pass
    mean_auc = np.mean(aucs) if aucs else 0
    print(f"  {name:<45s} pos={pos:5.1f}%  mean_feat_AUC={mean_auc:.4f}")

X, _, _ = eng.prepare_training_data(
    combined.assign(target=0).reset_index(drop=False)  # dummy target just to get X
)

print("\nTarget variant comparison:")
print("-"*80)

# Variant 1: current (2x ATR TP/SL, 15 days)
t1 = eng.create_target_variable(combined.copy(), forward_days=15, tp_mult=2.0, sl_mult=2.0)['target']
test_target("Current: 2xATR TP/SL, 15d", t1, X)

# Variant 2: tighter TP/SL (1.5x), shorter window
t2 = eng.create_target_variable(combined.copy(), forward_days=10, tp_mult=1.5, sl_mult=1.5)['target']
test_target("1.5xATR TP/SL, 10d", t2, X)

# Variant 3: simple forward return > 3%
fwd5 = combined['close'].pct_change(5).shift(-5)
t3 = (fwd5 > 0.03).astype(float)
t3[t3.index[-5:]] = np.nan
test_target("Forward 5d return > 3%", t3, X)

# Variant 4: forward return > 0 (just up or down)
fwd10 = combined['close'].pct_change(10).shift(-10)
t4 = (fwd10 > 0).astype(float)
t4[t4.index[-10:]] = np.nan
test_target("Forward 10d return > 0 (direction)", t4, X)

# Variant 5: forward return > 2%
t5 = (fwd10 > 0.02).astype(float)
t5[t5.index[-10:]] = np.nan
test_target("Forward 10d return > 2%", t5, X)

# Variant 6: 3xATR TP, 1xATR SL (asymmetric, high quality signals)
t6 = eng.create_target_variable(combined.copy(), forward_days=20, tp_mult=3.0, sl_mult=1.0)['target']
test_target("3xATR TP / 1xATR SL, 20d (high R:R)", t6, X)

# Variant 7: 1xATR TP, 2xATR SL (easy TP, hard SL → high win rate)
t7 = eng.create_target_variable(combined.copy(), forward_days=10, tp_mult=1.0, sl_mult=2.0)['target']
test_target("1xATR TP / 2xATR SL, 10d (easy TP)", t7, X)

# Variant 8: max gain in 10 days > 2%
def max_gain_target(df, days=10, threshold=0.02):
    closes = df['close'].values
    n = len(closes)
    t = np.full(n, np.nan)
    for i in range(n - days):
        future_max = df['high'].values[i+1:i+days+1].max()
        t[i] = 1.0 if (future_max / closes[i] - 1) >= threshold else 0.0
    return pd.Series(t, index=df.index)

t8 = max_gain_target(combined, days=10, threshold=0.03)
test_target("Max high in 10d > 3% gain", t8, X)
