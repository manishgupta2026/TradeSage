import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital

    def run_backtest(self, df: pd.DataFrame, signals: pd.Series) -> Dict:
        """
        Simulates buying on Signal=True and holding until next Signal=False 
        (Simple version, can be expanded to use exit_conditions).
        """
        if df.empty or signals.empty:
            return {}

        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]

        # Shift signals by 1 to avoid lookahead bias (trade on next open after signal)
        trade_signals = signals.shift(1).fillna(False)

        for i in range(len(df)):
            price = df.iloc[i]['Close']
            
            # Entry
            if trade_signals.iloc[i] and position == 0:
                position = capital / price
                capital = 0
                trades.append({
                    "type": "BUY",
                    "price": price,
                    "date": df.index[i]
                })
            
            # Exit (Signal turns False or last row)
            elif not trade_signals.iloc[i] and position > 0:
                capital = position * price
                position = 0
                trades.append({
                    "type": "SELL",
                    "price": price,
                    "date": df.index[i],
                    "profit": capital - self.initial_capital # Partial calculation
                })

            # Track equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)

        final_equity = capital + (position * price)
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "trade_count": len(trades) // 2,
            "trades": trades
        }
