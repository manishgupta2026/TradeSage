"""
TradeSage Market Scanner Module
Scans NSE stocks for trading opportunities using a trained ML model.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from data_fetcher import MarketDataFetcher
from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer
import warnings

warnings.filterwarnings('ignore')


class MarketScanner:
    """Scans live market data to find trading opportunities."""

    def __init__(self, model_path='tradesage_model.pkl'):
        """
        Initialize scanner with a trained model.

        Args:
            model_path: Path to the saved .pkl model file
        """
        self.fetcher = MarketDataFetcher()
        self.engineer = FeatureEngineer()
        self.trainer = TradingModelTrainer()
        self.trainer.load_model(model_path)
        self.default_capital = 100000

    def scan_stock(self, symbol, min_confidence=0.6):
        """
        Scans a single stock for trading signal.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with signal details
        """
        result = {
            'symbol': f"{symbol}.NS",
            'status': 'error',
            'prediction': 'HOLD',
            'signal': False,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            # Fetch 6 months of data (need history for indicators)
            df = self.fetcher.fetch_stock_data(symbol, period='6mo')
            if df.empty or len(df) < 50:
                result['status'] = 'insufficient_data'
                return result

            # Calculate indicators
            df = self.engineer.add_technical_indicators(df)
            df.dropna(inplace=True)

            if df.empty:
                result['status'] = 'no_valid_data'
                return result

            # Get latest row features
            non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'symbol']
            feature_cols = [c for c in df.columns if c not in non_feature_cols]
            X_latest = df[feature_cols].iloc[[-1]].copy()
            X_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_latest.fillna(0, inplace=True)

            # Predict
            predictions, probabilities = self.trainer.predict(X_latest)
            confidence = float(probabilities[0])
            prediction = int(predictions[0])

            # Current market data
            latest = df.iloc[-1]
            current_price = float(latest['close'])
            atr = float(latest.get('atr', current_price * 0.02))

            # Calculate stop-loss and position
            stop_loss = current_price - (3.0 * atr)
            stop_loss_pct = ((current_price - stop_loss) / current_price) * 100

            # Position sizing (10% risk on default capital)
            risk_amount = self.default_capital * 0.10
            shares = int(risk_amount / (3.0 * atr)) if atr > 0 else 0
            position_value = shares * current_price

            # Build result
            is_signal = prediction == 1 and confidence >= min_confidence

            result.update({
                'status': 'success',
                'prediction': 'BUY' if prediction == 1 else 'HOLD',
                'confidence': round(confidence, 4),
                'signal': is_signal,
                'current_price': round(current_price, 2),
                'atr': round(atr, 2),
                'stop_loss_3x_atr': round(stop_loss, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'suggested_shares': shares,
                'position_value': round(position_value, 2),
                'rsi_14': round(float(latest.get('rsi_14', 0)), 1),
                'macd': round(float(latest.get('macd', 0)), 2),
                'adx': round(float(latest.get('adx', 0)), 1),
            })

        except Exception as e:
            result['status'] = f'error: {str(e)}'

        return result

    def quick_scan_nifty50(self, min_confidence=0.6, top_n=20):
        """
        Scans top N Nifty 50 stocks for opportunities.

        Args:
            min_confidence: Minimum confidence threshold
            top_n: Number of stocks to scan

        Returns:
            List of results with buy signals
        """
        symbols = self.fetcher.NIFTY_50[:top_n]
        return self.scan_multiple_stocks(symbols, min_confidence)

    def scan_multiple_stocks(self, symbols, min_confidence=0.6, show_all=False):
        """
        Scans multiple stocks for opportunities.

        Args:
            symbols: List of stock symbols
            min_confidence: Minimum confidence threshold
            show_all: If True, include HOLD signals too

        Returns:
            List of signal results
        """
        print(f"\n{'═'*60}")
        print(f"SCANNING {len(symbols)} STOCKS")
        print(f"{'═'*60}\n")

        results = []
        buy_signals = []

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Scanning {symbol}...", end='')
            result = self.scan_stock(symbol, min_confidence)
            results.append(result)

            if result['signal']:
                buy_signals.append(result)
                print(f" 🟢 BUY ({result['confidence']*100:.1f}%)")
            elif result['status'] == 'success':
                print(f" ⚪ HOLD ({result['confidence']*100:.1f}%)")
            else:
                print(f" ⚠ {result['status']}")

        # Display opportunities
        self.display_opportunities(buy_signals)

        return buy_signals if not show_all else results

    def display_opportunities(self, results):
        """
        Displays formatted trading opportunities.

        Args:
            results: List of signal dictionaries with signal=True
        """
        signals = [r for r in results if r.get('signal', False)]

        print(f"\n{'═'*60}")
        if not signals:
            print("🔍 No trading opportunities found at this time.")
            print("   This may be normal — the system is selective.")
            print(f"{'═'*60}")
            return

        print(f"🎯 TRADING OPPORTUNITIES FOUND: {len(signals)}")
        print(f"{'═'*60}")

        for result in sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True):
            print(f"\n📊 {result['symbol']}")
            print(f"   Confidence: {result['confidence']*100:.1f}% | Price: ₹{result['current_price']:,.2f}")
            print(f"   Stop Loss (3x ATR): ₹{result['stop_loss_3x_atr']:,.2f} ({result['stop_loss_pct']:.2f}% below)")
            print(f"   Suggested Position: {result['suggested_shares']} shares (₹{result['position_value']:,.0f})")
            print(f"   RSI: {result['rsi_14']} | MACD: {result['macd']} | ADX: {result['adx']}")

        print(f"\n{'═'*60}")
        print("⚠ DISCLAIMER: These are ML-generated signals, NOT investment advice.")
        print("   Always do your own research. Paper trade first!")
        print(f"{'═'*60}")


if __name__ == '__main__':
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else 'tradesage_model.pkl'

    print("=== TradeSage Market Scanner ===\n")
    try:
        scanner = MarketScanner(model_path)

        # Scan a few stocks
        results = scanner.scan_multiple_stocks(
            ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'],
            min_confidence=0.6
        )
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("  Run training first: python tradesage_main.py (Option 1)")
    except Exception as e:
        print(f"✗ Error: {e}")
