"""
TradeSage Market Scanner Module v2
Scans NSE stocks for trading opportunities using a trained ML model.
Supports market context features and ensemble models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer
import warnings

warnings.filterwarnings('ignore')


class MarketScanner:
    """Scans live market data to find trading opportunities."""

    NIFTY_50 = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
        'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'BAJFINANCE', 'HCLTECH',
        'WIPRO', 'ONGC', 'NTPC', 'POWERGRID', 'JSWSTEEL',
        'TATASTEEL', 'ADANIENT', 'ADANIPORTS', 'GRASIM', 'COALINDIA',
        'BAJAJFINSV', 'TECHM', 'HINDALCO', 'DIVISLAB', 'CIPLA',
        'EICHERMOT', 'BPCL', 'TATAMOTORS', 'DRREDDY', 'HEROMOTOCO',
        'UPL', 'APOLLOHOSP', 'SBILIFE', 'BRITANNIA', 'INDUSINDBK',
        'BAJAJ-AUTO', 'TATACONSUM', 'M&M', 'HDFCLIFE', 'LTIM'
    ]

    def __init__(self, model_path='tradesage_model.pkl', fetcher=None):
        """
        Initialize scanner with a trained model.

        Args:
            model_path: Path to the saved .pkl model file
            fetcher: Optional data fetcher instance (MarketDataFetcher or AngelDataFetcher)
        """
        self.fetcher = fetcher
        self.engineer = FeatureEngineer()
        self.trainer = TradingModelTrainer()
        self.trainer.load_model(model_path)
        self.default_capital = 100000
        self.index_df = None  # Set externally for market context

    def set_market_context(self, index_df):
        """Set Nifty50 index data for market context features."""
        self.index_df = index_df

    def scan_stock(self, symbol, df=None, min_confidence=0.6):
        """
        Scans a single stock for trading signal.

        Args:
            symbol: Stock symbol
            df: Pre-fetched DataFrame (optional, if None will fetch)
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with signal details
        """
        result = {
            'symbol': symbol,
            'status': 'error',
            'prediction': 'HOLD',
            'signal': False,
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            if df is None:
                if self.fetcher is None:
                    result['status'] = 'no_fetcher'
                    return result
                df = self.fetcher.fetch_stock_data(symbol, period='6mo')

            if df is None or df.empty or len(df) < 50:
                result['status'] = 'insufficient_data'
                return result

            # Calculate indicators with market context
            df = self.engineer.add_technical_indicators(df, index_df=self.index_df)
            df.dropna(inplace=True)

            if df.empty:
                result['status'] = 'no_valid_data'
                return result

            # Predict using latest row
            predictions, probabilities = self.trainer.predict(df.iloc[[-1]])
            confidence = float(probabilities[0])
            prediction = int(predictions[0])

            # Current market data
            latest = df.iloc[-1]
            current_price = float(latest['close'])
            atr = float(latest.get('atr', current_price * 0.02))

            # Calculate stops
            stop_loss = current_price - (3.0 * atr)
            stop_loss_pct = ((current_price - stop_loss) / current_price) * 100
            take_profit = current_price + (3.5 * atr)
            take_profit_pct = ((take_profit - current_price) / current_price) * 100

            # Position sizing
            risk_amount = self.default_capital * 0.10
            shares = int(risk_amount / (3.0 * atr)) if atr > 0 else 0
            position_value = shares * current_price

            is_signal = prediction == 1 and confidence >= min_confidence

            result.update({
                'status': 'success',
                'prediction': 'BUY' if prediction == 1 else 'HOLD',
                'confidence': round(confidence, 4),
                'signal': is_signal,
                'current_price': round(current_price, 2),
                'atr': round(atr, 2),
                'stop_loss': round(stop_loss, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit': round(take_profit, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'suggested_shares': shares,
                'position_value': round(position_value, 2),
                'rsi_14': round(float(latest.get('rsi_14', 0)), 1),
                'adx': round(float(latest.get('adx', 0)), 1),
            })

        except Exception as e:
            result['status'] = f'error: {str(e)}'

        return result

    def scan_multiple_stocks(self, symbols, min_confidence=0.6, show_all=False):
        """Scans multiple stocks for opportunities."""
        print(f"\n{'═'*60}")
        print(f"SCANNING {len(symbols)} STOCKS")
        print(f"{'═'*60}\n")

        results = []
        buy_signals = []

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Scanning {symbol}...", end='')
            result = self.scan_stock(symbol, min_confidence=min_confidence)
            results.append(result)

            if result['signal']:
                buy_signals.append(result)
                print(f" 🟢 BUY ({result['confidence']*100:.1f}%)")
            elif result['status'] == 'success':
                print(f" ⚪ HOLD ({result['confidence']*100:.1f}%)")
            else:
                print(f" ⚠ {result['status']}")

        self.display_opportunities(buy_signals)

        return buy_signals if not show_all else results

    def quick_scan_nifty50(self, min_confidence=0.6, top_n=20):
        """Scans top N Nifty 50 stocks."""
        symbols = self.NIFTY_50[:top_n]
        return self.scan_multiple_stocks(symbols, min_confidence)

    def display_opportunities(self, results):
        """Displays formatted trading opportunities."""
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
            print(f"   Stop Loss:    ₹{result['stop_loss']:,.2f} ({result['stop_loss_pct']:.2f}% below)")
            print(f"   Take Profit:  ₹{result['take_profit']:,.2f} ({result['take_profit_pct']:.2f}% above)")
            print(f"   Position: {result['suggested_shares']} shares (₹{result['position_value']:,.0f})")
            print(f"   RSI: {result['rsi_14']} | ADX: {result['adx']}")

        print(f"\n{'═'*60}")
        print("⚠ DISCLAIMER: These are ML-generated signals, NOT investment advice.")
        print("   Always do your own research. Paper trade first!")
        print(f"{'═'*60}")


if __name__ == '__main__':
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else 'tradesage_model.pkl'

    print("=== TradeSage Market Scanner v2 ===\n")
    try:
        scanner = MarketScanner(model_path)
        results = scanner.scan_multiple_stocks(
            ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'],
            min_confidence=0.6
        )
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("  Run training first.")
    except Exception as e:
        print(f"✗ Error: {e}")
