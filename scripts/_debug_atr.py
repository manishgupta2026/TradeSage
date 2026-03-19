import pandas as pd, sys
sys.path.append('.')
from ta import volatility

df = pd.read_csv('data_cache_angel/RELIANCE_daily.csv', index_col=0, parse_dates=True)
df.columns = [c.lower() for c in df.columns]
print("Columns:", df.columns.tolist())
print("Head:\n", df.head(3))
print("dtypes:", df.dtypes)

atr = volatility.average_true_range(df['high'], df['low'], df['close'])
print("\nATR head:", atr.head(20).values)
print("ATR non-zero count:", (atr != 0).sum())
print("ATR describe:", atr.describe())
