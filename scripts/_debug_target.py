import pandas as pd, sys, os
sys.path.append('.')
from src.core.feature_engineering import FeatureEngineer

df = pd.read_csv('data_cache_angel/RELIANCE_daily.csv', index_col=0, parse_dates=True)
df.columns = [c.lower() for c in df.columns]
print(f'RELIANCE rows: {len(df)}')

eng = FeatureEngineer()
df = eng.add_technical_indicators(df)
df = eng.create_target_variable(df, forward_days=15, tp_mult=2.0, sl_mult=2.0)
print(f'After features+target, non-null rows: {df.dropna().shape[0]}')
print(f'Target distribution:')
print(df['target'].value_counts())
print(f'Positive rate: {df["target"].mean()*100:.1f}%')
print(f'ATR pct sample: {df["atr_pct"].dropna().head(5).values}')

# Check if ATR is reasonable
print(f'\nATR/close ratio mean: {(df["atr"] / df["close"]).mean():.4f}')
print(f'TP distance (2*ATR) as % of price: {2 * (df["atr"] / df["close"]).mean() * 100:.2f}%')
