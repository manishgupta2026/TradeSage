"""
Live Paper Trading Script (Angel One Integration)
Simulates trading using the trained XGBoost model and genuine Angel One market data.
Checks for stop-loss and take-profit exits on existing open positions, then scans for new entries.
"""

import json
import logging
import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from tqdm import tqdm
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.angel.angel_one_api import AngelOneAPI
from src.angel.angel_data_fetcher import AngelDataFetcher
from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer
from src.core.fundamental_analyzer import FundamentalAnalyzer
from src.utils.telegram_bot import TelegramBot

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class AngelPaperTrader:
    def __init__(self, model_path=None, positions_file=None):
        # Auto-discover model file
        if model_path is None:
            for candidate in ['tradesage_10y.pkl', 'current.pkl', 'tradesage_v2.pkl', 'tradesage_angel.pkl']:
                p = os.path.join(PROJECT_ROOT, 'models', candidate)
                if os.path.exists(p):
                    model_path = p
                    break
        self.model_path = model_path or os.path.join(PROJECT_ROOT, 'models', 'tradesage_10y.pkl')
        self.positions_file = Path(positions_file or os.path.join(PROJECT_ROOT, 'data', 'positions.json'))
        
        # OPTIMIZED PARAMETERS (from backtest)
        self.default_capital = 50000  # Rs.50k
        self.position_size = 2000     # Rs.2k per trade
        self.max_positions = 25       # Max concurrent positions
        self.min_prob = 0.65          # Minimum confidence threshold
        self.stop_loss_atr = 3.0      # Stop loss ATR multiplier (matches training max_drawdown)
        self.take_profit_atr = 3.5    # Take profit ATR multiplier
        self.max_hold_days = 5        # Maximum holding period (matches forward_days)
        
        # Market context data
        self.nifty_df = None
        
        # Load APIs
        try:
            self.api = AngelOneAPI(os.path.join(PROJECT_ROOT, 'config', 'angel_config.json'))
            self.fetcher = AngelDataFetcher(self.api)
        except Exception as e:
            logger.error(f"Cannot connect to Angel One: {e}")
            raise
            
        # Load Model
        self.engineer = FeatureEngineer()
        self.trainer = TradingModelTrainer()
        if not os.path.exists(self.model_path):
            logger.error(f"Model file {self.model_path} not found! Please train the model first.")
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        self.trainer.load_model(self.model_path)
        
        # Initialize Telegram
        self.telegram = TelegramBot(
            token=self.api.api_config.get('telegram_token'),
            chat_id=self.api.api_config.get('telegram_chat_id')
        )
        
        # Fetch Nifty50 for market context
        logger.info("Fetching Nifty50 index for market context...")
        self._fetch_nifty_data()
        
        # Load positions
        self.positions = {}
        if self.positions_file.exists():
            with open(self.positions_file, 'r') as f:
                self.positions = json.load(f)
        else:
            logger.info("No existing positions file")
            self.telegram.send_message("ℹ️ *TradeSage:* Bot started. No existing positions.")
    
    def _fetch_nifty_data(self):
        """Fetch Nifty50 index data for market context features"""
        try:
            # Try Angel One first
            self.nifty_df = self.fetcher.fetch_historical_data('^NSEI', period_days=365)
            if self.nifty_df is None or len(self.nifty_df) < 200:
                # Fallback to yfinance
                import yfinance as yf
                self.nifty_df = yf.download('^NSEI', period='1y', progress=False)
                if len(self.nifty_df) > 0:
                    self.nifty_df.index = self.nifty_df.index.tz_localize(None)
                    if isinstance(self.nifty_df.columns, pd.MultiIndex):
                        self.nifty_df.columns = self.nifty_df.columns.get_level_values(0)
                    self.nifty_df.columns = [str(c).lower() for c in self.nifty_df.columns]
            
            if self.nifty_df is not None and len(self.nifty_df) > 200:
                logger.info(f"✓ Loaded Nifty50 data: {len(self.nifty_df)} rows")
            else:
                logger.warning("⚠ Could not fetch Nifty50 - market context disabled")
                self.nifty_df = None
        except Exception as e:
            logger.warning(f"⚠ Nifty50 fetch failed: {e} - market context disabled")
            self.nifty_df = None
            
    def save_positions(self):
        with open(self.positions_file, 'w') as f:
            json.dump(self.positions, f, indent=4)
            
    def check_exits(self):
        """Monitor existing positions for Stop Loss or Take Profit"""
        open_positions = {sym: pos for sym, pos in self.positions.items() if pos.get('status') == 'open'}
        
        if not open_positions:
            return
            
        logger.info(f"\n🔍 Checking {len(open_positions)} positions for exits...")
        
        for symbol, pos in open_positions.items():
            df = self.fetcher.fetch_historical_data(symbol, period_days=30)
            if df is None or df.empty:
                continue
                
            current_price = float(df.iloc[-1]['close'])
            entry_price = float(pos['entry_price'])
            stop_loss = float(pos['stop_loss'])
            take_profit = float(pos.get('take_profit', entry_price * 1.10))
            
            # Calculate days held
            entry_date = datetime.datetime.fromisoformat(pos['entry_date'])
            days_held = (datetime.datetime.now() - entry_date).days
            
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            exit_price, reason = None, None
            if current_price >= take_profit:
                exit_price, reason = current_price, f'Take Profit ({self.take_profit_atr}x ATR)'
            elif current_price <= stop_loss:
                exit_price, reason = stop_loss, f'Stop Loss ({self.stop_loss_atr}x ATR)'
            elif days_held >= self.max_hold_days:
                exit_price, reason = current_price, f'Time Exit ({self.max_hold_days} days)'
            
            if exit_price:
                logger.info(f"\n🔔 EXIT SIGNAL: {symbol} - {reason}")
                logger.info(f"   PNL: {pnl_pct:.2f}%")
                
                # Update positions
                self.positions[symbol]['status'] = 'closed'
                self.positions[symbol]['exit_price'] = exit_price
                self.positions[symbol]['exit_date'] = datetime.datetime.now().isoformat()
                self.positions[symbol]['exit_reason'] = reason
                
                # Send Telegram Notification
                res_msg = f"{pnl_pct:+.2f}% | {reason}"
                self.telegram.send_trade_alert(symbol, "EXIT", exit_price, confidence=res_msg)
                
        self.save_positions()

    def scan_for_opportunities(self, watchlist):
        """Scan watchlist for new buy signals"""
        # Check current open positions
        open_count = sum(1 for pos in self.positions.values() if pos.get('status') == 'open')
        available_slots = self.max_positions - open_count
        
        if available_slots <= 0:
            logger.info(f"\n⚠️ Max positions reached ({self.max_positions}). No new trades.")
            return
        
        logger.info(f"\n🔍 Scanning {len(watchlist)} stocks (max {available_slots} new positions)...")
        
        buy_signals = []
        
        for symbol in tqdm(watchlist, desc="Scanning"):
            # Skip if already open
            if symbol in self.positions and self.positions[symbol].get('status') == 'open':
                continue
                
            df = self.fetcher.fetch_historical_data(symbol, period_days=365)  # 1 year for market context
            if df is None or len(df) < 200:
                continue
                
            try:
                # Engineer features WITH market context
                df = self.engineer.add_technical_indicators(df, index_df=self.nifty_df)
                df.dropna(inplace=True)
                if df.empty: 
                    continue
                
                # Predict
                non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'symbol']
                feature_cols = [c for c in df.columns if c not in non_feature_cols]
                
                X_latest = df[feature_cols].iloc[[-1]].copy()
                X_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_latest.fillna(0, inplace=True)
                
                predictions, probabilities = self.trainer.predict(X_latest)
                
                confidence = float(probabilities[0])
                prediction = int(predictions[0])
                
                if prediction == 1 and confidence >= self.min_prob:
                    latest = df.iloc[-1]
                    current_price = float(latest['close'])
                    atr = float(latest.get('atr', current_price * 0.02))
                    
                    # Use optimized parameters
                    stop_loss = current_price - (self.stop_loss_atr * atr)
                    take_profit = current_price + (self.take_profit_atr * atr)
                    
                    # Fixed position size
                    shares = int(self.position_size / current_price) if current_price > 0 else 0
                    
                    buy_signals.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'atr': atr,
                        'shares': shares,
                        'position_value': shares * current_price
                    })
            except Exception as e:
                continue
                
        if buy_signals:
            logger.info(f"\n✓ Found {len(buy_signals)} buy signals")
            
            # Sort by confidence and get 3x the available slots for fundamental screening
            buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
            candidate_signals = buy_signals[:available_slots * 3]
            
            logger.info(f"\nEvaluating top {len(candidate_signals)} candidates fundamentally...")
            
            analyzer = FundamentalAnalyzer()
            final_signals = []
            
            for sig in candidate_signals:
                sym = sig['symbol']
                if analyzer.evaluate_candidate(sym):
                    final_signals.append(sig)
                if len(final_signals) == available_slots:
                    break
                    
            if not final_signals:
                logger.info("\n⚠ All candidates failed fundamental screening. No trades executed.")
                return
                
            logger.info(f"\nExecuting top {len(final_signals)} fundamentally strong trades...")
            
            for sig in final_signals:
                sym = sig['symbol']
                sl_pct = ((sig['current_price'] - sig['stop_loss']) / sig['current_price']) * 100
                tp_pct = ((sig['take_profit'] - sig['current_price']) / sig['current_price']) * 100
                
                logger.info(f"\n📊 Trading Signal: {sym}")
                logger.info(f"   Confidence: {sig['confidence']*100:.1f}%")
                logger.info(f"   Entry Price: ₹{sig['current_price']:,.2f}")
                logger.info(f"   Stop Loss: ₹{sig['stop_loss']:,.2f} (-{sl_pct:.2f}%)")
                logger.info(f"   Take Profit: ₹{sig['take_profit']:,.2f} (+{tp_pct:.2f}%)")
                logger.info(f"   Shares: {sig['shares']}")
                logger.info(f"   Position Value: ₹{sig['position_value']:,.0f}")
                logger.info("   🔵 DRY RUN - Order not placed")
                
                self.positions[sym] = {
                    'entry_price': sig['current_price'],
                    'shares': sig['shares'],
                    'stop_loss': sig['stop_loss'],
                    'take_profit': sig['take_profit'],
                    'entry_date': datetime.datetime.now().isoformat(),
                    'confidence': sig['confidence'],
                    'status': 'open'
                }
                
                # Update Telegram
                self.telegram.send_trade_alert(
                    sym, "ENTRY", sig['current_price'], 
                    confidence=sig['confidence'], 
                    sl=sig['stop_loss'], 
                    tp=sig['take_profit']
                )
            self.save_positions()
        else:
            logger.info(f"\n✓ Found 0 buy signals (threshold: {self.min_prob*100:.0f}%)")

def main():
    logger.info("="*80)
    logger.info(f"DAILY TRADING WORKFLOW - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("="*80)
    logger.info("Mode: 🔵 DRY RUN (Paper Trading)")
    
    try:
        trader = AngelPaperTrader()
        
        # Load watchlist
        watchlist_file = os.path.join(PROJECT_ROOT, 'data', 'nse_top_3000_angel.json')
        logger.info(f"\n📂 Loading symbols from {watchlist_file}...")
        try:
            with open(watchlist_file, 'r') as f:
                watchlist = json.load(f)
        except Exception:
            # Fallback to a hardcoded safe list for scanning if the file isn't around
            watchlist = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'HINDUNILVR', 'KOTAKBANK', 'L&T', 'AXISBANK', 'HCLTECH', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'TATASTEEL']
            
        # 1. Check exits
        trader.check_exits()
        
        # 2. Scan for new entries
        trader.scan_for_opportunities(watchlist[:500])  # Limit scan to 500 stocks for speed
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        
    logger.info("\n" + "="*80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()
