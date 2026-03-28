"""
TradeSage Backtesting Module (backtesting.py)
Simulates trading with 3x ATR stop-loss, position sizing, and realistic fills.

NOTE: This module is named backtesting.py but uses internal class name Backtester
to avoid conflicts with any existing backtesting packages.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class Backtester:
    """Backtesting engine with 3x ATR stop-loss and realistic trade simulation."""

    def __init__(self, initial_capital=100000, position_size=0.10,
                 stop_loss_atr_multiplier=3.0, take_profit_pct=0.05):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital in INR (default: ₹1,00,000)
            position_size: Fraction of capital to risk per trade (default: 10%)
            stop_loss_atr_multiplier: ATR multiplier for stop-loss (default: 3x)
            take_profit_pct: Take profit percentage (default: 5%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.sl_atr_mult = stop_loss_atr_multiplier
        self.tp_pct = take_profit_pct
        self.brokerage_fee = 20.0  # ₹20 flat fee per order (buy/sell)
        self.slippage_pct = 0.001  # 0.1% slippage
        self.trades = []
        self.equity_curve = []

    def calculate_position_size(self, capital, price, atr):
        """
        Calculates position size based on risk.

        Args:
            capital: Current available capital
            price: Entry price
            atr: Current ATR value

        Returns:
            Number of shares to buy
        """
        risk_amount = capital * self.position_size
        stop_distance = self.sl_atr_mult * atr

        if stop_distance <= 0 or price <= 0:
            return 0

        shares = int(risk_amount / stop_distance)

        # Cap position value to available capital
        max_shares = int(capital / price)
        shares = min(shares, max_shares)

        return max(shares, 0)

    def run_backtest(self, df, predictions, probabilities, min_confidence=0.6):
        """
        Runs backtest simulation with realistic fills.

        Args:
            df: DataFrame with OHLCV data and 'atr' column
            predictions: Array of buy/hold predictions (1=buy, 0=hold)
            probabilities: Array of confidence scores
            min_confidence: Minimum confidence to enter trade

        Returns:
            Dictionary of performance metrics
        """
        self.trades = []
        capital = self.initial_capital
        self.equity_curve = [capital]
        position = None  # {shares, entry_price, stop_loss, take_profit, entry_date}

        # Ensure df has lowercase columns
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Align lengths — trim df to match predictions
        min_len = min(len(df), len(predictions), len(probabilities))
        df = df.iloc[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            current_open = row['open']
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']
            current_vol = row['volume']
            current_date = df.index[i]

            # Get ATR from previous day (no look-ahead)
            atr = prev_row.get('atr', 0)
            if pd.isna(atr) or atr <= 0:
                atr = abs(current_close * 0.02)  # Fallback: 2% of price

            # ─── CHECK EXITS FIRST (on today's bar) ───
            if position is not None:
                exit_reason = None
                exit_price = None

                # Stop-loss check: triggered if low <= stop
                if current_low <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']

                # Take-profit check: triggered if high >= target
                elif current_high >= position['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = position['take_profit']

                if exit_reason:
                    # Apply slippage on exit
                    actual_exit = exit_price * (1 - self.slippage_pct) if exit_reason == 'stop_loss' else exit_price * (1 - self.slippage_pct)
                    gross_pnl = (actual_exit - position['entry_price']) * position['shares']
                    net_pnl = gross_pnl - self.brokerage_fee  # Sell fee
                    pnl_pct = (actual_exit / position['entry_price'] - 1) * 100
                    capital += (actual_exit * position['shares']) - self.brokerage_fee

                    self.trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': actual_exit,
                        'shares': position['shares'],
                        'pnl': round(net_pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'exit_reason': exit_reason,
                        'confidence': position['confidence']
                    })
                    position = None

            # ─── CHECK ENTRIES (based on previous day's signal) ───
            if position is None and i > 0:
                prev_pred = predictions[i - 1]
                prev_prob = probabilities[i - 1]

                # Market Impact constraint
                if prev_pred == 1 and prev_prob >= min_confidence and prev_row.get('volume', float('inf')) >= 100000:
                    # Enter at today's open with slippage
                    actual_entry = current_open * (1 + self.slippage_pct)
                    shares = self.calculate_position_size(capital, actual_entry, atr)

                    cost = (actual_entry * shares) + self.brokerage_fee # Buy fee
                    if shares > 0 and cost <= capital:
                        stop_loss = actual_entry - (self.sl_atr_mult * atr)
                        take_profit = actual_entry * (1 + self.tp_pct)
                        capital -= cost

                        position = {
                            'shares': shares,
                            'entry_price': actual_entry,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_date': current_date,
                            'confidence': prev_prob
                        }

            # Track equity
            pos_value = position['shares'] * current_close if position else 0
            self.equity_curve.append(capital + pos_value)

        # ─── CLOSE ANY REMAINING POSITION ───
        if position is not None:
            exit_price = df.iloc[-1]['close'] * (1 - self.slippage_pct)
            gross_pnl = (exit_price - position['entry_price']) * position['shares']
            net_pnl = gross_pnl - self.brokerage_fee
            pnl_pct = (exit_price / position['entry_price'] - 1) * 100
            capital += (exit_price * position['shares']) - self.brokerage_fee

            self.trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'shares': position['shares'],
                'pnl': round(net_pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'exit_reason': 'end_of_data',
                'confidence': position['confidence']
            })

        # ─── CALCULATE RESULTS ───
        return self._calculate_metrics(capital)

    def _calculate_metrics(self, final_capital):
        """Calculates comprehensive performance metrics from trade log."""
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        if not self.trades:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': round(final_capital, 2),
                'total_return_pct': round(total_return, 2),
                'total_trades': 0,
                'message': 'No trades executed'
            }

        trades_df = pd.DataFrame(self.trades)
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] < 0]

        total_trades = len(trades_df)
        win_count = len(winning)
        loss_count = len(losing)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        total_wins = winning['pnl'].sum() if not winning.empty else 0
        total_losses = abs(losing['pnl'].sum()) if not losing.empty else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Average win/loss
        avg_win = winning['pnl'].mean() if not winning.empty else 0
        avg_loss = losing['pnl'].mean() if not losing.empty else 0

        # Max drawdown
        equity = pd.Series(self.equity_curve)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Sharpe ratio (annualized)
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Calmar Ratio
        annualized_return = (final_capital / self.initial_capital) ** (252 / len(equity)) - 1
        calmar = annualized_return / (abs(max_drawdown)/100) if max_drawdown < 0 else 0

        # Average Win/Loss Ratio
        aw_al_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

        # Average confidence
        avg_confidence = trades_df['confidence'].mean() * 100

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'aw_al_ratio': round(aw_al_ratio, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'calmar_ratio': round(calmar, 2),
            'avg_confidence': round(avg_confidence, 1),
            'stop_loss_atr_mult': self.sl_atr_mult,
            'exit_reasons': exit_reasons
        }

        self._print_results(results)
        return results

    def _print_results(self, results):
        """Pretty-prints backtest results."""
        print(f"\n{'═'*60}")
        print("BACKTEST RESULTS")
        print(f"{'═'*60}")
        print(f"Initial Capital:     ₹{results['initial_capital']:,.2f}")
        print(f"Final Capital:       ₹{results['final_capital']:,.2f}")
        print(f"Total Return:        {results['total_return_pct']:+.2f}%")
        print(f"")
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Winning Trades:      {results['winning_trades']}")
        print(f"Losing Trades:       {results['losing_trades']}")
        print(f"Win Rate:            {results['win_rate']:.2f}%")
        print(f"")
        print(f"Average Win:         ₹{results['avg_win']:,.2f}")
        print(f"Average Loss:        ₹{results['avg_loss']:,.2f}")
        print(f"Avg Win/Loss Ratio:  {results['aw_al_ratio']:.2f}")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"Calmar Ratio:        {results['calmar_ratio']:.2f}")
        print(f"")
        print(f"Stop Loss ATR Mult:  {results['stop_loss_atr_mult']}x")
        print(f"Avg Confidence:      {results['avg_confidence']:.1f}%")
        print(f"")
        print("Exit Reasons:")
        for reason, count in results['exit_reasons'].items():
            pct = count / results['total_trades'] * 100
            print(f"  {reason}: {count} trades ({pct:.1f}%)")

    def get_trades_df(self):
        """
        Returns all trades as a DataFrame.

        Returns:
            DataFrame with columns: entry_date, exit_date, entry_price,
            exit_price, shares, pnl, pnl_pct, exit_reason, confidence
        """
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=== Backtester Quick Test ===\n")

    np.random.seed(42)
    n = 200

    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    prices = 2000 + np.cumsum(np.random.normal(0, 15, n))

    df = pd.DataFrame({
        'open': prices + np.random.normal(0, 5, n),
        'high': prices + abs(np.random.normal(10, 5, n)),
        'low': prices - abs(np.random.normal(10, 5, n)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n),
        'atr': np.full(n, 20.0)
    }, index=dates)

    predictions = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    probabilities = np.random.uniform(0.4, 0.8, n)
    probabilities[predictions == 0] = np.random.uniform(0.2, 0.5, (predictions == 0).sum())

    bt = Backtester(initial_capital=100000, stop_loss_atr_multiplier=3.0)
    results = bt.run_backtest(df, predictions, probabilities, min_confidence=0.6)

    trades_df = bt.get_trades_df()
    if not trades_df.empty:
        print(f"\n--- Sample Trades ---")
        print(trades_df[['entry_date', 'exit_date', 'pnl', 'pnl_pct', 'exit_reason']].head(5).to_string())
