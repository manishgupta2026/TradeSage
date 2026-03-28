"""
TradeSage - Realistic Backtesting Engine (backtest/simulate_trades.py)

Simulates NSE swing trading with realistic fills including:
  - Brokerage fees (Angel One: ₹20/order flat)
  - STT (Securities Transaction Tax): 0.1% on sell side
  - Exchange charges: 0.00325% both sides
  - SEBI turnover charge: 0.0001%
  - Slippage: 0.1% each way
  - Market impact skip: skip if avg volume < 100k shares
  - 3x ATR stop-loss
  - Position sizing: max 5% of capital per trade

Performance metrics:
  Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, Calmar Ratio
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Cost constants (NSE / Angel One) ─────────────────────────────────────────

BROKERAGE_PER_ORDER = 20.0      # ₹20 flat per order (Angel One equity delivery)
STT_SELL_PCT = 0.001            # 0.1% STT on sell side
EXCHANGE_CHARGE_PCT = 0.0000325 # 0.00325% NSE exchange transaction charge
SEBI_CHARGE_PCT = 0.000001      # 0.0001% SEBI turnover charge
SLIPPAGE_PCT = 0.001            # 0.1% each way


def _total_cost(trade_value: float, side: str = "buy") -> float:
    """
    Calculate total round-trip or one-way cost for a trade.

    Parameters
    ----------
    trade_value : gross value of the trade (shares × price)
    side        : 'buy', 'sell', or 'both'
    """
    brokerage = BROKERAGE_PER_ORDER
    exchange = trade_value * EXCHANGE_CHARGE_PCT
    sebi = trade_value * SEBI_CHARGE_PCT
    stt = trade_value * STT_SELL_PCT if side in ("sell", "both") else 0.0
    return brokerage + exchange + sebi + stt


class Backtester:
    """
    Backtesting engine with realistic NSE trading costs and 3x ATR stop-loss.

    Usage
    -----
    bt = Backtester(initial_capital=100_000)
    results = bt.run_backtest(df, predictions, probabilities)
    trades_df = bt.get_trades_df()
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        position_size_pct: float = 0.05,    # max 5% of capital per trade
        stop_loss_atr_mult: float = 3.0,
        take_profit_pct: float = 0.08,       # 8% take-profit target
        min_volume_threshold: int = 100_000, # skip illiquid stocks
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.sl_atr_mult = stop_loss_atr_mult
        self.tp_pct = take_profit_pct
        self.min_volume_threshold = min_volume_threshold
        self.trades: list = []
        self.equity_curve: list = []

    # ── Position Sizing ───────────────────────────────────────────────────

    def calculate_position_size(
        self, capital: float, price: float, atr: float
    ) -> int:
        """
        Risk-based position sizing capped at *position_size_pct* of capital.

        Shares = min(
            risk_amount / stop_distance,
            capital * position_size_pct / price
        )
        """
        if price <= 0 or atr <= 0:
            return 0
        risk_amount = capital * self.position_size_pct
        stop_distance = self.sl_atr_mult * atr
        if stop_distance <= 0:
            return 0
        shares_by_risk = int(risk_amount / stop_distance)
        shares_by_capital = int((capital * self.position_size_pct) / price)
        return max(0, min(shares_by_risk, shares_by_capital))

    # ── Main Backtest ─────────────────────────────────────────────────────

    def run_backtest(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        min_confidence: float = 0.6,
    ) -> dict:
        """
        Run backtest simulation with realistic fills.

        Parameters
        ----------
        df              : OHLCV DataFrame with 'atr' column (index = dates)
        predictions     : 1-D array of 0/1 signals
        probabilities   : 1-D array of prediction probabilities
        min_confidence  : minimum probability to enter a trade

        Returns
        -------
        dict of performance metrics
        """
        self.trades = []
        capital = self.initial_capital
        self.equity_curve = [capital]
        position = None

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Align lengths
        n = min(len(df), len(predictions), len(probabilities))
        df = df.iloc[:n]
        predictions = predictions[:n]
        probabilities = probabilities[:n]

        for i in range(1, n):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            current_open = row["open"]
            current_high = row["high"]
            current_low = row["low"]
            current_close = row["close"]
            current_date = df.index[i]

            # ATR from previous bar (no look-ahead)
            atr = prev_row.get("atr", 0)
            if pd.isna(atr) or atr <= 0:
                atr = abs(current_close * 0.02)

            # ── CHECK EXITS ───────────────────────────────────────────────
            if position is not None:
                exit_reason = None
                exit_price = None

                if current_low <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                    # Apply slippage (worse fill on SL)
                    exit_price = position["stop_loss"] * (1 - SLIPPAGE_PCT)

                elif current_high >= position["take_profit"]:
                    exit_reason = "take_profit"
                    exit_price = position["take_profit"]

                if exit_reason:
                    gross = exit_price * position["shares"]
                    cost = _total_cost(gross, side="sell")
                    net_proceeds = gross - cost

                    pnl = net_proceeds - position["entry_cost"]
                    pnl_pct = (
                        pnl / position["entry_cost"] * 100
                        if position["entry_cost"] > 0
                        else 0
                    )
                    capital += net_proceeds

                    self.trades.append(
                        {
                            "entry_date": position["entry_date"],
                            "exit_date": current_date,
                            "entry_price": position["entry_price"],
                            "exit_price": round(exit_price, 2),
                            "shares": position["shares"],
                            "gross_pnl": round(
                                (exit_price - position["entry_price"])
                                * position["shares"],
                                2,
                            ),
                            "net_pnl": round(pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "exit_reason": exit_reason,
                            "confidence": position["confidence"],
                            "total_fees": round(
                                position["entry_fees"] + cost, 2
                            ),
                        }
                    )
                    position = None

            # ── CHECK ENTRIES ──────────────────────────────────────────────
            if position is None and i > 0:
                prev_pred = predictions[i - 1]
                prev_prob = probabilities[i - 1]

                # Skip illiquid (check yesterday's volume)
                prev_vol = prev_row.get("volume", 0)
                if prev_vol < self.min_volume_threshold:
                    self.equity_curve.append(capital)
                    continue

                if prev_pred == 1 and prev_prob >= min_confidence:
                    # Enter at today's open + slippage
                    entry_price = current_open * (1 + SLIPPAGE_PCT)
                    shares = self.calculate_position_size(capital, entry_price, atr)

                    if shares > 0:
                        gross = entry_price * shares
                        entry_fees = _total_cost(gross, side="buy")
                        entry_cost = gross + entry_fees

                        if entry_cost <= capital:
                            stop_loss = entry_price - (self.sl_atr_mult * atr)
                            take_profit = entry_price * (1 + self.tp_pct)
                            capital -= entry_cost

                            position = {
                                "shares": shares,
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "entry_date": current_date,
                                "entry_cost": entry_cost,
                                "entry_fees": entry_fees,
                                "confidence": prev_prob,
                            }

            pos_value = position["shares"] * current_close if position else 0
            self.equity_curve.append(capital + pos_value)

        # Close any remaining position at last close
        if position is not None:
            exit_price = df.iloc[-1]["close"]
            gross = exit_price * position["shares"]
            cost = _total_cost(gross, side="sell")
            net_proceeds = gross - cost
            pnl = net_proceeds - position["entry_cost"]
            pnl_pct = (
                pnl / position["entry_cost"] * 100
                if position["entry_cost"] > 0
                else 0
            )
            capital += net_proceeds

            self.trades.append(
                {
                    "entry_date": position["entry_date"],
                    "exit_date": df.index[-1],
                    "entry_price": position["entry_price"],
                    "exit_price": round(exit_price, 2),
                    "shares": position["shares"],
                    "gross_pnl": round(
                        (exit_price - position["entry_price"]) * position["shares"],
                        2,
                    ),
                    "net_pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "exit_reason": "end_of_data",
                    "confidence": position["confidence"],
                    "total_fees": round(position["entry_fees"] + cost, 2),
                }
            )

        return self._calculate_metrics(capital)

    # ── Metrics ───────────────────────────────────────────────────────────

    def _calculate_metrics(self, final_capital: float) -> dict:
        total_return = (
            (final_capital - self.initial_capital) / self.initial_capital * 100
        )

        if not self.trades:
            return {
                "initial_capital": self.initial_capital,
                "final_capital": round(final_capital, 2),
                "total_return_pct": round(total_return, 2),
                "total_trades": 0,
                "message": "No trades executed",
            }

        tdf = pd.DataFrame(self.trades)
        winning = tdf[tdf["net_pnl"] > 0]
        losing = tdf[tdf["net_pnl"] < 0]

        total_trades = len(tdf)
        win_count = len(winning)
        loss_count = len(losing)
        win_rate = win_count / total_trades * 100

        total_gains = winning["net_pnl"].sum() if not winning.empty else 0.0
        total_losses = abs(losing["net_pnl"].sum()) if not losing.empty else 1e-10
        profit_factor = total_gains / total_losses

        avg_win = winning["net_pnl"].mean() if not winning.empty else 0.0
        avg_loss = losing["net_pnl"].mean() if not losing.empty else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        total_fees = tdf["total_fees"].sum()

        # Equity curve metrics
        equity = pd.Series(self.equity_curve)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Calmar ratio = annualised return / |max drawdown|
        years = len(self.equity_curve) / 252
        ann_return = ((final_capital / self.initial_capital) ** (1 / max(years, 0.1)) - 1) * 100
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else float("inf")

        results = {
            "initial_capital": self.initial_capital,
            "final_capital": round(final_capital, 2),
            "total_return_pct": round(total_return, 2),
            "annualised_return_pct": round(ann_return, 2),
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "calmar_ratio": round(calmar, 2),
            "total_fees_paid": round(total_fees, 2),
            "sl_atr_mult": self.sl_atr_mult,
            "exit_reasons": tdf["exit_reason"].value_counts().to_dict(),
        }

        self._print_results(results)
        return results

    def _print_results(self, r: dict) -> None:
        print(f"\n{'═' * 60}")
        print("BACKTEST RESULTS  (Realistic NSE costs)")
        print(f"{'═' * 60}")
        print(f"Initial Capital      : ₹{r['initial_capital']:>12,.2f}")
        print(f"Final Capital        : ₹{r['final_capital']:>12,.2f}")
        print(f"Total Return         :  {r['total_return_pct']:>+8.2f}%")
        print(f"Annualised Return    :  {r['annualised_return_pct']:>+8.2f}%")
        print(f"")
        print(f"Total Trades         :  {r['total_trades']}")
        print(f"Win Rate             :  {r['win_rate']:.2f}%")
        print(f"Avg Win              : ₹{r['avg_win']:>10,.2f}")
        print(f"Avg Loss             : ₹{r['avg_loss']:>10,.2f}")
        print(f"Win/Loss Ratio       :  {r['win_loss_ratio']:.2f}x")
        print(f"Profit Factor        :  {r['profit_factor']:.2f}")
        print(f"")
        print(f"Max Drawdown         :  {r['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio         :  {r['sharpe_ratio']:.2f}")
        print(f"Calmar Ratio         :  {r['calmar_ratio']:.2f}")
        print(f"")
        print(f"Total Fees Paid      : ₹{r['total_fees_paid']:>10,.2f}")
        print(f"Stop Loss ATR Mult   :  {r['sl_atr_mult']}x")
        print(f"")
        print("Exit Reasons:")
        for reason, count in r["exit_reasons"].items():
            pct = count / r["total_trades"] * 100
            print(f"  {reason:<20s}: {count:4d}  ({pct:.1f}%)")

    # ── Helpers ───────────────────────────────────────────────────────────

    def get_trades_df(self) -> pd.DataFrame:
        """Return trade log as a DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

    def save_trades_csv(self, path: str = "trade_log.csv") -> None:
        """Save trade log to CSV for external analysis."""
        df = self.get_trades_df()
        if df.empty:
            print("No trades to save.")
            return
        df.to_csv(path, index=False)
        print(f"✓ Trade log saved → {path}  ({len(df)} trades)")


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Backtester Quick Test ===\n")
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    prices = 2000 + np.cumsum(np.random.normal(0.5, 15, n))

    df = pd.DataFrame(
        {
            "open": prices + np.random.normal(0, 5, n),
            "high": prices + abs(np.random.normal(10, 5, n)),
            "low": prices - abs(np.random.normal(10, 5, n)),
            "close": prices,
            "volume": np.random.randint(200_000, 2_000_000, n),
            "atr": np.full(n, 22.0),
        },
        index=dates,
    )

    preds = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    probs = np.where(preds == 1, np.random.uniform(0.55, 0.9, n), np.random.uniform(0.2, 0.5, n))

    bt = Backtester(initial_capital=100_000, stop_loss_atr_mult=3.0)
    results = bt.run_backtest(df, preds, probs, min_confidence=0.6)

    trades = bt.get_trades_df()
    if not trades.empty:
        print(f"\nSample Trades:")
        print(
            trades[["entry_date", "exit_date", "net_pnl", "pnl_pct", "exit_reason", "total_fees"]]
            .head(5)
            .to_string()
        )
