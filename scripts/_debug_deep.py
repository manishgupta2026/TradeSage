"""Deep diagnostic — understand why AUC is stuck at 0.55"""
import pandas as pd, numpy as np, sys, os, glob
sys.path.append('.')
from src.core.feature_engineering import FeatureEngineer

CACHE = 'data_cache_angel'
SKIP  = ['BEES','IETF','BETA','CASE','ADD','ETF','NIFTY','SENSEX',
         'GOLD','SILVER','LIQUID','GILT','BOND','NSETEST']

csvs = [f for f in sorted(glob.glob(os.path.join(CACHE,'*_daily.csv')))
        if not any(k in os.path.basename(f).upper() for k in SKIP)][:300]

eng = FeatureEngineer()
all_dfs = []
for path in csvs:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if len(df) < 250: continue
    try:
        df = eng.add_technical_indicators(df)
        df = eng.create_target_variable(df, forward_days=15, tp_mult=2.0, sl_mult=2.0)
        all_dfs.append(df.dropna())
    except: pass

combined = pd.concat(all_dfs)
print(f"Total rows: {len(combined):,}")
print(f"Overall positive rate: {combined['target'].mean()*100:.1f}%")

# Check feature correlation with target
X, y, feats = eng.prepare_training_data(combined)
print(f"\nFeature→target correlations (top 20):")
corrs = X.corrwith(y).abs().sort_values(ascending=False)
print(corrs.head(20).to_string())

# Check if any single feature has strong predictive power
print(f"\nTop feature AUC (individual):")
from sklearn.metrics import roc_auc_score
top_feats = corrs.head(10).index.tolist()
for feat in top_feats:
    try:
        auc = roc_auc_score(y, X[feat])
        auc = max(auc, 1-auc)  # flip if inverted
        print(f"  {feat:<30s} AUC={auc:.4f}")
    except: pass

# Check target quality — are last N rows always 0 (no forward data)?
print(f"\nLast 20 target values per stock (should be 0 due to forward window):")
for df in all_dfs[:3]:
    print(f"  {df['target'].tail(20).values}")
