import pandas as pd
import pandas_ta as ta

class IndicatorLibrary:
    @staticmethod
    def add_standard_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adds a standard set of swing trading indicators to the dataframe."""
        if df.empty:
            return df

        # Ensure we have a clean copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Trend Indicators
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        if len(df) >= 200:
            df.ta.ema(length=200, append=True)
        else:
            # Add a placeholder if not enough data
            df['EMA_200'] = float('nan')
        
        df.ta.adx(length=14, append=True) # ADX for Trend Strength

        # Momentum Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.stoch(append=True) # Stochastic Oscillator (%K, %D)
        df.ta.willr(append=True) # Williams %R
        
        # Volatility
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True) # Average True Range
        
        # Volume
        df.ta.obv(append=True)
        # Volume SMA (for volume spikes)
        df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Clean up column names (remove underscores or spaces if any)
        # pandas_ta columns often look like "STOCHk_14_3_3", "ATRr_14"
        # We normalize them for easier parsing
        df.columns = [c.replace(" ", "_") for c in df.columns]

        return df

    @staticmethod
    def check_condition(df: pd.DataFrame, column: str, operator: str, value: float) -> pd.Series:
        """Helper to check conditions like 'RSI_14 < 30'."""
        if column not in df.columns:
            return pd.Series([False] * len(df))
            
        if operator == "<":
            return df[column] < value
        elif operator == ">":
            return df[column] > value
        elif operator == "<=":
            return df[column] <= value
        elif operator == ">=":
            return df[column] >= value
        elif operator == "==":
            return df[column] == value
        
        return pd.Series([False] * len(df))
