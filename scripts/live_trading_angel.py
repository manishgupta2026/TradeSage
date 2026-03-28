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

from data.angel_one_api import AngelOneAPI
from data.angel_data_fetcher import AngelDataFetcher
from features.technical_indicators import FeatureEngineer
from models.train_xgboost import TradingModelTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class AngelPaperTrader:
    def __init__(self, model_path=None, positions_file=None):
        self.model_path = model_path or os.path.join(PROJECT_ROOT, 'models', 'tradesage_angel.pkl')
        self.positions_file = Path(positions_file or os.path.join(PROJECT_ROOT, 'positions.json'))
        self.default_capital = 100000
        
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
        
        # Load positions
        self.positions = {}
        if self.positions_file.exists():
            with open(self.positions_file, 'r') as f:
                self.positions = json.load(f)
        else:
            logger.info("No existing positions file")
            
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
            df = self.fetcher.fetch_historical_data(symbol, period_days=10)
            if df is None or df.empty:
                continue
                
            current_price = float(df.iloc[-1]['close'])
            entry_price = float(pos['entry_price'])
            stop_loss = float(pos['stop_loss'])
            
            # 5% take profit
            take_profit = entry_price * 1.05
            
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            if current_price >= take_profit:
                logger.info(f"\n🟢 TAKE PROFIT HIT: {symbol}")
                logger.info(f"   Entry: ₹{entry_price:.2f}")
                logger.info(f"   Current: ₹{current_price:.2f}")
                logger.info(f"   Profit: {pnl_pct:.2f}%")
                logger.info("   🔵 DRY RUN - Order not placed")
                self.positions[symbol]['status'] = 'closed'
                self.positions[symbol]['exit_price'] = current_price
                self.positions[symbol]['exit_date'] = datetime.datetime.now().isoformat()
                self.positions[symbol]['exit_reason'] = 'Take profit'
                
            elif current_price <= stop_loss:
                logger.info(f"\n🔴 STOP LOSS HIT: {symbol}")
                logger.info(f"   Entry: ₹{entry_price:.2f}")
                logger.info(f"   Current: ₹{current_price:.2f}")
                logger.info(f"   Loss: {pnl_pct:.2f}%")
                logger.info("   🔵 DRY RUN - Order not placed")
                self.positions[symbol]['status'] = 'closed'
                self.positions[symbol]['exit_price'] = current_price
                self.positions[symbol]['exit_date'] = datetime.datetime.now().isoformat()
                self.positions[symbol]['exit_reason'] = 'Stop loss'
                
        self.save_positions()

    def scan_for_opportunities(self, watchlist, min_confidence=0.60):
        """Scan watchlist for new buy signals"""
        logger.info(f"\n🔍 Scanning {len(watchlist)} stocks...")
        
        buy_signals = []
        
        for symbol in tqdm(watchlist, desc="Scanning"):
            # Skip if already open
            if symbol in self.positions and self.positions[symbol].get('status') == 'open':
                continue
                
            df = self.fetcher.fetch_historical_data(symbol, period_days=180) # 6 months
            if df is None or len(df) < 50:
                continue
                
            try:
                # Engineer
                df = self.engineer.add_technical_indicators(df)
                df.dropna(inplace=True)
                if df.empty: continue
                
                # Predict
                non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'symbol']
                feature_cols = [c for c in df.columns if c not in non_feature_cols]
                
                X_latest = df[feature_cols].iloc[[-1]].copy()
                X_latest.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_latest.fillna(0, inplace=True)
                
                predictions, probabilities = self.trainer.predict(X_latest)
                
                confidence = float(probabilities[0])
                prediction = int(predictions[0])
                
                if prediction == 1 and confidence >= min_confidence:
                    latest = df.iloc[-1]
                    current_price = float(latest['close'])
                    atr = float(latest.get('atr', current_price * 0.02))
                    
                    stop_loss = current_price - (3.0 * atr)
                    risk_amount = self.default_capital * 0.10
                    shares = int(risk_amount / (3.0 * atr)) if atr > 0 else 0
                    
                    buy_signals.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'shares': shares,
                        'position_value': shares * current_price
                    })
            except Exception as e:
                continue
                
        if buy_signals:
            logger.info(f"\n✓ Found {len(buy_signals)} buy signals")
            logger.info(f"\nExecuting {len(buy_signals)} trades...")
            
            # Sort by confidence
            buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            for index, sig in enumerate(buy_signals):
                if index >= 3:  # Only take top 3 at most per scan
                    break
                    
                sym = sig['symbol']
                sl_pct = ((sig['current_price'] - sig['stop_loss']) / sig['current_price']) * 100
                logger.info(f"\n📊 Trading Signal: {sym}")
                logger.info(f"   Confidence: {sig['confidence']*100:.1f}%")
                logger.info(f"   Entry Price: ₹{sig['current_price']:,.2f}")
                logger.info(f"   Stop Loss: ₹{sig['stop_loss']:,.2f} ({sl_pct:.2f}% below)")
                logger.info(f"   Shares: {sig['shares']}")
                logger.info(f"   Position Value: ₹{sig['position_value']:,.0f}")
                logger.info("   🔵 DRY RUN - Order not placed")
                
                self.positions[sym] = {
                    'entry_price': sig['current_price'],
                    'shares': sig['shares'],
                    'stop_loss': sig['stop_loss'],
                    'entry_date': datetime.datetime.now().isoformat(),
                    'confidence': sig['confidence'],
                    'status': 'open'
                }
            self.save_positions()
        else:
            logger.info(f"\n✓ Found 0 buy signals")

def main():
    logger.info("="*80)
    logger.info(f"DAILY TRADING WORKFLOW - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("="*80)
    logger.info("Mode: 🔵 DRY RUN (Paper Trading)")
    
    try:
        trader = AngelPaperTrader()
        
        # Load watchlist
        watchlist_file = os.path.join(PROJECT_ROOT, 'data', 'nse_top_2000_angel.json')
        logger.info(f"\n📂 Loading symbols from {watchlist_file}...")
        try:
            with open(watchlist_file, 'r') as f:
                watchlist = json.load(f)
        except Exception:
            # Fallback to a hardcoded safe list for scanning if the file isn't around
            watchlist = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'HINDUNILVR', 'KOTAKBANK', 'L&T', 'AXISBANK', 'HCLTECH', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'TATASTEEL']
            
        # 1. Check exits
        trader.check_exits()
        
        # 2. Scan for new entries (limitting to 50 for quick paper-trade demo, per README)
        trader.scan_for_opportunities(watchlist[:50], min_confidence=0.60)
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        
    logger.info("\n" + "="*80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()
