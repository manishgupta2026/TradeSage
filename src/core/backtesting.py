"""
TradeSage Backtesting Module v2
Realistic simulation: ATR-based stops, brokerage fees, STT, slippage,
volume filters, and comprehensive metrics (Sharpe, Calmar, recovery factor).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')


class Backtester:
    """Backtesting engine with realistic fills, fees, and comprehensive metrics."""

    def __init__(self, initial_capital=100000, position_size=0.10,
                 stop_loss_atr_multiplier=3.0, take_profit_pct=0.05,
                 brokerage_per_order=20.0, stt_pct=0.001,
                 slippage_pct=0.001, min_volume=100000):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital in INR (default: ₹1,00,000)
            position_size: Fraction of capital to risk per trade (default: 10%)
            stop_loss_atr_multiplier: ATR multiplier for stop-loss (default: 3x)
            take_profit_pct: Take profit percentage (default: 5%)
            brokerage_per_order: Per-order brokerage in INR (default: ₹20)
            stt_pct: Securities Transaction Tax on sell (default: 0.1%)
            slippage_pct: Slippage per trade (default: 0.1%)
            min_volume: Minimum avg volume filter (default: 100K)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.sl_atr_mult = stop_loss_atr_multiplier
        self.tp_pct = take_profit_pct
        self.brokerage = brokerage_per_order
        self.stt_pct = stt_pct
        self.slippage_pct = slippage_pct
        self.min_volume = min_volume
        self.trades = []
        self.equity_curve = []

    def calculate_position_size(self, capital, price, atr):
        """Risk-based position sizing."""
        risk_amount = capital * self.position_size
        stop_distance = self.sl_atr_mult * atr

        if stop_distance <= 0 or price <= 0:
            return 0

        shares = int(risk_amount / stop_distance)
        max_shares = int(capital / price)
        shares = min(shares, max_shares)
        return max(shares, 0)

    def run_backtest(self, df, predictions, probabilities, min_confidence=0.6):
        """
        Runs backtest simulation with realistic fills, fees, and slippage.
        """
        self.trades = []
        capital = self.initial_capital
        self.equity_curve = [capital]
        position = None

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        min_len = min(len(df), len(predictions), len(probabilities))
        df = df.iloc[:min_len]
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            current_open  = row['open']
            current_high  = row['high']
            current_low   = row['low']
            current_close = row['close']
            current_date  = df.index[i]

            atr = prev_row.get('atr', 0)
            if pd.isna(atr) or atr <= 0:
                atr = abs(current_close * 0.02)

            # ─── CHECK EXITS FIRST ───
            if position is not None:
                exit_reason = None
                exit_price = None

                if current_low <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                    exit_price = position['stop_loss']
                elif current_high >= position['take_profit']:
                    exit_reason = 'take_profit'
                    exit_price = position['take_profit']

                if exit_reason:
                    # Apply slippage on exit
                    if exit_reason == 'stop_loss':
                        exit_price *= (1 - self.slippage_pct)  # Worse price on SL
                    else:
                        exit_price *= (1 - self.slippage_pct / 2)  # Small slippage on TP

                    # Calculate P&L with fees
                    gross_pnl = (exit_price - position['entry_price']) * position['shares']
                    sell_value = exit_price * position['shares']
                    exit_fees = self.brokerage + (sell_value * self.stt_pct)
                    net_pnl = gross_pnl - position['entry_fees'] - exit_fees
                    pnl_pct = (exit_price / position['entry_price'] - 1) * 100

                    capital += sell_value - exit_fees

                    self.trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': round(exit_price, 2),
                        'shares': position['shares'],
                        'gross_pnl': round(gross_pnl, 2),
                        'fees': round(position['entry_fees'] + exit_fees, 2),
                        'net_pnl': round(net_pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'exit_reason': exit_reason,
                        'confidence': position['confidence']
                    })
                    position = None

            # ─── CHECK ENTRIES ───
            if position is None and i > 0:
                prev_pred = predictions[i - 1]
                prev_prob = probabilities[i - 1]

                if prev_pred == 1 and prev_prob >= min_confidence:
                    # Volume filter
                    avg_vol = df['volume'].iloc[max(0, i-20):i].mean()
                    if avg_vol < self.min_volume:
                        self.equity_curve.append(capital)
                        continue

                    # Apply slippage on entry (worse fill)
                    entry_price = current_open * (1 + self.slippage_pct)
                    shares = self.calculate_position_size(capital, entry_price, atr)

                    entry_fees = self.brokerage  # Buy-side brokerage
                    total_cost = entry_price * shares + entry_fees

                    if shares > 0 and total_cost <= capital:
                        stop_loss = entry_price - (self.sl_atr_mult * atr)
                        take_profit = entry_price * (1 + self.tp_pct)
                        capital -= total_cost

                        position = {
                            'shares': shares,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_date': current_date,
                            'confidence': prev_prob,
                            'entry_fees': entry_fees,
                        }

            pos_value = position['shares'] * current_close if position else 0
            self.equity_curve.append(capital + pos_value)

        # ─── CLOSE ANY REMAINING POSITION ───
        if position is not None:
            exit_price = df.iloc[-1]['close']
            gross_pnl = (exit_price - position['entry_price']) * position['shares']
            sell_value = exit_price * position['shares']
            exit_fees = self.brokerage + (sell_value * self.stt_pct)
            net_pnl = gross_pnl - position['entry_fees'] - exit_fees
            pnl_pct = (exit_price / position['entry_price'] - 1) * 100
            capital += sell_value - exit_fees

            self.trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'shares': position['shares'],
                'gross_pnl': round(gross_pnl, 2),
                'fees': round(position['entry_fees'] + exit_fees, 2),
                'net_pnl': round(net_pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'exit_reason': 'end_of_data',
                'confidence': position['confidence']
            })

        return self._calculate_metrics(capital)

    def _calculate_metrics(self, final_capital):
        """Comprehensive performance metrics."""
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
        winning = trades_df[trades_df['net_pnl'] > 0]
        losing  = trades_df[trades_df['net_pnl'] < 0]

        total_trades = len(trades_df)
        win_count    = len(winning)
        loss_count   = len(losing)
        win_rate     = (win_count / total_trades * 100) if total_trades > 0 else 0

        # Profit factor (using net P&L)
        total_wins   = winning['net_pnl'].sum() if not winning.empty else 0
        total_losses = abs(losing['net_pnl'].sum()) if not losing.empty else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Total fees paid
        total_fees = trades_df['fees'].sum()

        # Average win/loss
        avg_win  = winning['net_pnl'].mean() if not winning.empty else 0
        avg_loss = losing['net_pnl'].mean() if not losing.empty else 0

        # Win/loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Equity curve metrics
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

        # Calmar ratio (annualized return / max drawdown)
        trading_days = len(self.equity_curve)
        annual_return = ((final_capital / self.initial_capital) ** (252 / max(trading_days, 1)) - 1) * 100
        calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

        # Recovery factor (total return / max drawdown)
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

        # Average confidence
        avg_confidence = trades_df['confidence'].mean() * 100

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'win_loss_ratio': round(win_loss_ratio, 2),
            'profit_factor': round(profit_factor, 2),
            'total_fees': round(total_fees, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'calmar_ratio': round(calmar, 2),
            'recovery_factor': round(recovery_factor, 2),
            'avg_confidence': round(avg_confidence, 1),
            'stop_loss_atr_mult': self.sl_atr_mult,
            'exit_reasons': exit_reasons,
            'trading_costs': {
                'brokerage_per_order': self.brokerage,
                'stt_pct': self.stt_pct,
                'slippage_pct': self.slippage_pct,
            }
        }

        self._print_results(results)
        return results

    def _print_results(self, results):
        """Pretty-prints backtest results."""
        print(f"\n{'═'*60}")
        print("BACKTEST RESULTS (REALISTIC)")
        print(f"{'═'*60}")
        print(f"Initial Capital:     ₹{results['initial_capital']:,.2f}")
        print(f"Final Capital:       ₹{results['final_capital']:,.2f}")
        print(f"Total Return:        {results['total_return_pct']:+.2f}%")
        print(f"Annual Return:       {results['annual_return_pct']:+.2f}%")
        print(f"")
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Winning Trades:      {results['winning_trades']}")
        print(f"Losing Trades:       {results['losing_trades']}")
        print(f"Win Rate:            {results['win_rate']:.2f}%")
        print(f"Win/Loss Ratio:      {results['win_loss_ratio']:.2f}")
        print(f"")
        print(f"Average Win:         ₹{results['avg_win']:,.2f}")
        print(f"Average Loss:        ₹{results['avg_loss']:,.2f}")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"Total Fees Paid:     ₹{results['total_fees']:,.2f}")
        print(f"")
        print(f"Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"Calmar Ratio:        {results['calmar_ratio']:.2f}")
        print(f"Recovery Factor:     {results['recovery_factor']:.2f}")
        print(f"")
        print(f"Stop Loss ATR Mult:  {results['stop_loss_atr_mult']}x")
        print(f"Avg Confidence:      {results['avg_confidence']:.1f}%")
        print(f"")
        tc = results['trading_costs']
        print(f"Trading Costs:       ₹{tc['brokerage_per_order']}/order + "
              f"{tc['stt_pct']*100:.1f}% STT + {tc['slippage_pct']*100:.1f}% slippage")
        print(f"")
        print("Exit Reasons:")
        for reason, count in results['exit_reasons'].items():
            pct = count / results['total_trades'] * 100
            print(f"  {reason}: {count} trades ({pct:.1f}%)")

    def get_trades_df(self):
        """Returns trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def export_trades_csv(self, path='backtest_trades.csv'):
        """Export trade log to CSV for analysis."""
        trades_df = self.get_trades_df()
        if trades_df.empty:
            print("No trades to export.")
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        trades_df.to_csv(path, index=False)
        print(f"  Trades exported → {path}")


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=== Backtester v2 Quick Test ===\n")

    np.random.seed(42)
    n = 200

    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    prices = 2000 + np.cumsum(np.random.normal(0, 15, n))

    df = pd.DataFrame({
        'open':   prices + np.random.normal(0, 5, n),
        'high':   prices + abs(np.random.normal(10, 5, n)),
        'low':    prices - abs(np.random.normal(10, 5, n)),
        'close':  prices,
        'volume': np.random.randint(100000, 1000000, n),
        'atr':    np.full(n, 20.0)
    }, index=dates)

    predictions   = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    probabilities = np.random.uniform(0.4, 0.8, n)
    probabilities[predictions == 0] = np.random.uniform(0.2, 0.5, (predictions == 0).sum())

    bt = Backtester(
        initial_capital=100000,
        stop_loss_atr_multiplier=3.0,
        brokerage_per_order=20.0,
        stt_pct=0.001,
        slippage_pct=0.001,
    )
    results = bt.run_backtest(df, predictions, probabilities, min_confidence=0.6)

    trades_df = bt.get_trades_df()
    if not trades_df.empty:
        print(f"\n--- Sample Trades ---")
        print(trades_df[['entry_date', 'exit_date', 'net_pnl', 'fees', 'pnl_pct', 'exit_reason']].head(5).to_string())
