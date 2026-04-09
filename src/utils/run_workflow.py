"""
TradeSage - GitHub Actions Workflow Runner v2
Entry point for automated paper trading via GitHub Actions.
Uses the XGBoost ML model (not the old strategy executor).
"""

import os
import sys
import json
import logging
from datetime import datetime
import pytz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.paper.paper_trader import PaperTrader
from src.utils.telegram_bot import TelegramBot

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')
MARKET_OPEN = 9
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30


def is_market_hours():
    """Check if within NSE trading hours (9:15 AM - 3:30 PM IST)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def main():
    try:
        logger.info("=" * 80)
        logger.info(f"TRADESAGE WORKFLOW - {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
        logger.info("=" * 80)
        logger.info("Mode: 🔵 DRY RUN (Paper Trading)")

        # Initialize
        trader = PaperTrader(
            portfolio_file=os.path.join(PROJECT_ROOT, 'data', 'paper_portfolio.json'),
            initial_capital=50000
        )

        telegram = TelegramBot(
            token=os.getenv('TELEGRAM_BOT_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )

        # --- Load Angel One API ---
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
            from angel.angel_one_api import AngelOneAPI
            from angel.angel_data_fetcher import AngelDataFetcher
            api = AngelOneAPI(os.path.join(PROJECT_ROOT, 'config', 'angel_config.json'))
            fetcher = AngelDataFetcher(api)
        except Exception as e:
            logger.error(f"Cannot initialize Angel One API: {e}")
            telegram.send_message(f"⚠️ Bot Error: Cannot connect to Angel One: {e}")
            sys.exit(1)

        # --- Load ML Model ---
        from core.model_training import TradingModelTrainer
        from core.feature_engineering import FeatureEngineer
        from core.fundamental_analyzer import FundamentalAnalyzer

        model_path = None
        for candidate in ['tradesage_10y.pkl', 'current.pkl', 'tradesage_v2.pkl', 'tradesage_angel.pkl']:
            p = os.path.join(PROJECT_ROOT, 'models', candidate)
            if os.path.exists(p):
                model_path = p
                break
        if not model_path:
            logger.error("No model file found in models/ directory")
            telegram.send_message("⚠️ Bot Error: Model file not found.")
            sys.exit(1)

        trainer_model = TradingModelTrainer()
        trainer_model.load_model(model_path)
        engineer = FeatureEngineer()

        # --- Fetch Nifty50 for market context ---
        nifty_df = None
        try:
            nifty_df = fetcher.fetch_historical_data('^NSEI', period_days=365)
            if nifty_df is not None and len(nifty_df) > 200:
                logger.info(f"✓ Nifty50 data: {len(nifty_df)} rows")
            else:
                nifty_df = None
        except Exception:
            nifty_df = None

        # --- Step 1: Check exits on existing positions ---
        exit_msgs = []
        current_prices = {}
        prev_closes = {}

        for ticker, holding in list(trader.portfolio['holdings'].items()):
            try:
                df = fetcher.fetch_historical_data(ticker, period_days=30)
                if df is not None and not df.empty:
                    current_prices[ticker] = float(df.iloc[-1]['close'])
                    if len(df) > 1:
                        prev_closes[ticker] = float(df.iloc[-2]['close'])
                    else:
                        prev_closes[ticker] = holding['avg_price']
                else:
                    current_prices[ticker] = holding['avg_price']
                    prev_closes[ticker] = holding['avg_price']
            except Exception:
                current_prices[ticker] = holding['avg_price']
                prev_closes[ticker] = holding['avg_price']

        exit_msgs = trader.update_portfolio(current_prices)

        # --- Drawdown protection ---
        DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "2500"))
        summary = trader.get_summary(current_prices, prev_closes)
        todays_pnl = summary.get('todays_pnl', 0)

        if todays_pnl < -DAILY_LOSS_LIMIT:
            msg = f"🛑 DRAWDOWN PROTECTION - Today P&L: ₹{todays_pnl:.2f}, Limit: ₹{DAILY_LOSS_LIMIT}"
            logger.info(msg)
            telegram.send_message(msg)
            sys.exit(0)

        # --- Step 2: Scan for new ML signals ---
        watchlist_file = os.path.join(PROJECT_ROOT, 'data', 'nse_top_3000_angel.json')
        try:
            with open(watchlist_file, 'r') as f:
                watchlist = json.load(f)
        except Exception:
            watchlist = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC',
                         'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'HINDUNILVR']

        logger.info(f"\n🔍 Scanning {min(500, len(watchlist))} stocks with ML model...")

        import numpy as np
        from tqdm import tqdm

        buy_signals = []
        min_prob = 0.65

        for symbol in tqdm(watchlist[:500], desc="Scanning"):
            if symbol in trader.portfolio['holdings']:
                continue

            try:
                df = fetcher.fetch_historical_data(symbol, period_days=365)
                if df is None or len(df) < 200:
                    continue

                df = engineer.add_technical_indicators(df, index_df=nifty_df)
                df.dropna(inplace=True)
                if df.empty:
                    continue

                predictions, probabilities = trainer_model.predict(df.iloc[[-1]])
                confidence = float(probabilities[0])

                if int(predictions[0]) == 1 and confidence >= min_prob:
                    latest = df.iloc[-1]
                    price = float(latest['close'])
                    atr = float(latest.get('atr', price * 0.02))

                    buy_signals.append({
                        'ticker': symbol,
                        'price': price,
                        'action': 'BUY',
                        'confidence': confidence,
                        'sl': price - (3.0 * atr),       # Matches training max_drawdown
                        'target': price + (3.5 * atr),    # Matches backtest TP config
                        'atr_pct': (atr / price) * 100,
                    })
            except Exception:
                continue

        # Sort by confidence, get candidates for fundamental screening
        buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
        candidate_signals = buy_signals[:9]
        
        final_signals = []
        if candidate_signals:
            logger.info(f"\nEvaluating top {len(candidate_signals)} candidates fundamentally...")
            analyzer = FundamentalAnalyzer()
            for sig in candidate_signals:
                sym = sig['ticker']
                if analyzer.evaluate_candidate(sym):
                    final_signals.append(sig)
                if len(final_signals) == 3:
                    break
        
        buy_signals = final_signals

        # --- Build Telegram message ---
        msg = f"🤖 **TradeSage Scan** ({datetime.now(IST).strftime('%H:%M IST')})\n\n"

        if exit_msgs:
            msg += '**📈 Exits:**\n'
            for em in exit_msgs:
                msg += f'{em}\n'
            msg += '\n'

        if buy_signals:
            msg += '**🔍 ML Signals:**\n'
            for sig in buy_signals[:3]:
                trade_msg = trader.execute_trade(sig)
                if trade_msg:
                    msg += f'📊 {sig["ticker"]} @ ₹{sig["price"]:.0f} ({sig["confidence"]*100:.0f}%)\n'
                    msg += f'   {trade_msg}\n\n'

        # Portfolio summary
        summary = trader.get_summary(current_prices, prev_closes)
        pnl_emoji = '🟢' if summary['total_pnl'] >= 0 else '🔴'
        msg += f'────────────────\n'
        msg += f'💼 Equity: ₹{summary["equity"]:,.0f}\n'
        msg += f'{pnl_emoji} P&L: ₹{summary["total_pnl"]:,.0f} ({summary["roi"]:.1f}%)\n'
        msg += f'📊 {summary["open_positions"]} Open | {summary["closed_trades"]} Closed\n'

        logger.info(msg)

        if telegram.token and telegram.chat_id:
            telegram.send_message(msg)

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
