"""
TradeSage Angel One Backtester
Simulates real PNL on historical predictions using the trained XGBoost model.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Config
# Use Angel One API cache (3yr OHLCV CSVs) — same data the model was trained on
DATA_DIR   = Path(PROJECT_ROOT) / 'data_cache_angel'
MODEL_PATH = Path(PROJECT_ROOT) / 'models' / 'tradesage_angel.pkl'  # Updated to new model

# Simulation Parameters
# Optimal parameters based on your data:

# Simulation Parameters - STANDARDIZED (matches training config)
INITIAL_CAPITAL = 50000.0    # 50k INR
POSITION_SIZE = 2000.0       # 2k per trade -> 25 concurrent slots
STOP_LOSS_ATR_MULT = 3.0     # Matches training max_drawdown=-0.03
TAKE_PROFIT_ATR_MULT = 3.5   # 1.17:1 R:R
MAX_HOLD_DAYS = 5            # Matches training forward_days=5
MAX_POSITIONS = int(INITIAL_CAPITAL / POSITION_SIZE)  # 25 positions
MIN_PROB = 0.65              # Higher threshold
MIN_ROWS = 1                 # lowered to 1 for debugging with short data history

# Transaction Costs (realistic Indian market costs)
BROKERAGE_PCT = 0.0003       # 0.03% brokerage
STT_GST_PCT = 0.001          # 0.1% STT + GST + stamp duty
SLIPPAGE_PCT = 0.0005        # 0.05% slippage
TOTAL_COST_PCT = BROKERAGE_PCT + STT_GST_PCT + SLIPPAGE_PCT  # 0.18% per trade

def run_backtest():
    logger.info("="*60)
    logger.info(" TRADESAGE HISTORICAL PNL BACKTESTER")
    logger.info("="*60)
    
    if not MODEL_PATH.exists():
        logger.error("[ERROR] Model not found! Please run train_angel_one.py first.")
        return

    # Load Model & Scaler
    trainer = TradingModelTrainer()
    try:
        trainer.load_model(str(MODEL_PATH))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    engineer = FeatureEngineer()
    
    # Fetch Nifty50 for market context (if model was trained with it)
    logger.info("\n Fetching Nifty50 index for market context...")
    nifty_df = None
    try:
        nifty_file = DATA_DIR.parent / 'data_cache_angel' / 'NSEI_daily.csv'
        if nifty_file.exists():
            nifty_df = pd.read_csv(nifty_file, index_col='timestamp', parse_dates=True)
            if nifty_df.index.tz is not None:
                nifty_df.index = nifty_df.index.tz_localize(None)
            nifty_df.columns = [c.lower() for c in nifty_df.columns]
            logger.info(f"  Loaded Nifty50: {len(nifty_df)} rows")
        else:
            # Fallback to yfinance
            import yfinance as yf
            nifty_df = yf.download('^NSEI', period='3y', progress=False)
            if len(nifty_df) > 0:
                nifty_df.index = nifty_df.index.tz_localize(None)
                # Handle MultiIndex columns from yfinance
                if isinstance(nifty_df.columns, pd.MultiIndex):
                    nifty_df.columns = nifty_df.columns.get_level_values(0)
                nifty_df.columns = [str(c).lower() for c in nifty_df.columns]
                logger.info(f"  Downloaded Nifty50: {len(nifty_df)} rows")
    except Exception as e:
        logger.warning(f"  Could not load Nifty50: {e}")
    
    if nifty_df is None or len(nifty_df) < 200:
        logger.warning("  Market context features disabled (no Nifty50 data)")
        nifty_df = None
    
    # Process cached stocks — use Angel One 3yr CSVs (same source as training)
    skip_keywords = ['BEES','IETF','BETA','CASE','ETF','NIFTY','SENSEX',
                     'GOLD','SILVER','LIQUID','GILT','BOND']

    data_files = [
        f for f in DATA_DIR.glob('*_daily.csv')
        if f.stat().st_size > 5000  # skip stub/empty files (lowered from 10000)
        and not any(k in f.name.upper() for k in skip_keywords)
    ]

    if not data_files:
        logger.error("[ERROR] No CSV data found in data_cache_angel/. Run train_angel_one.py first.")
        return

    logger.info(f"Loaded {len(data_files)} stocks for backtesting...")
    if nifty_df is not None:
        logger.info("  Market context: ENABLED")
    processed = 0
    trades = []

    for file in data_files:
        symbol = file.stem.replace('_daily', '')
        try:
            df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
            # Normalize timezone-aware index to tz-naive for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [c.lower().strip() for c in df.columns]
            if len(df) < MIN_ROWS:
                continue

            df_features = engineer.add_technical_indicators(df, index_df=nifty_df)
            df_features = df_features.dropna()
            if len(df_features) < MIN_ROWS:
                continue

            # Use same feature filtering as training (prepare_training_data)
            X_bt, _, _ = engineer.prepare_training_data(df_features)
            if len(X_bt) < 1:
                continue

            predictions, probabilities = trainer.predict(X_bt)
            
            # Debug: check why no trades
            if probabilities.max() >= MIN_PROB:
                print(f"  {symbol}: Potential Trade! Max Prob = {probabilities.max():.4f}")

            # Align df_features to X_bt index (prepare_training_data may drop rows)
            df_sim = df_features.loc[X_bt.index]

            # --- vectorised trade simulation ---
            closes = df_sim['close'].values
            highs  = df_sim['high'].values
            lows   = df_sim['low'].values
            atrs   = df_sim['atr'].values if 'atr' in df_sim.columns else closes * 0.02
            dates  = df_sim.index

            in_trade = False
            for i in range(len(df_sim)):
                if in_trade:
                    exit_price, reason = 0.0, ''
                    if lows[i] <= sl:
                        exit_price, reason = sl, 'Stop Loss'
                    elif highs[i] >= tp:
                        exit_price, reason = tp, 'Take Profit'
                    elif (dates[i] - entry_date).days >= MAX_HOLD_DAYS:
                        exit_price, reason = closes[i], f'Time Stop ({MAX_HOLD_DAYS}d)'

                    if exit_price > 0:
                        pnl_pct = (exit_price - entry_price) / entry_price
                        # Apply transaction costs (entry + exit)
                        pnl_pct = pnl_pct - (2 * TOTAL_COST_PCT)
                        
                        trades.append({
                            'symbol':     symbol,
                            'entry_date': entry_date,
                            'exit_date':  dates[i],
                            'entry_price': entry_price,
                            'exit_price':  exit_price,
                            'pnl_pct':    round(pnl_pct * 100, 2),
                            'pnl_rupees': round(POSITION_SIZE * pnl_pct, 2),
                            'reason':     reason,
                            'hold_days':  (dates[i] - entry_date).days,
                        })
                        in_trade = False
                        print(f"    CLOSED {symbol} PNL: {pnl_pct*100:.2f}%")

                if not in_trade and probabilities[i] >= MIN_PROB:
                    in_trade    = True
                    entry_price = closes[i]
                    entry_date  = dates[i]
                    atr_i = atrs[i] if (not np.isnan(atrs[i]) and atrs[i] > 0) else closes[i] * 0.02
                    sl = entry_price - STOP_LOSS_ATR_MULT   * atr_i
                    tp = entry_price + TAKE_PROFIT_ATR_MULT * atr_i
                    print(f"    ENTERED {symbol} @ {entry_price:.2f} (Prob: {probabilities[i]:.4f})")

        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            continue
        processed += 1
        if processed % 100 == 0:
            logger.info(f"  Processed {processed} stocks, {len(trades)} trades so far...")
            
    if not trades:
        logger.warning(f"[!] No trades were initiated across {len(data_files)} stocks during this backtest!")
        return
        
    trades_df = pd.DataFrame(trades)
    
    # Portfolio Execution Simulation
    logger.info(f"Simulating Portfolio Execution with Capital Constraints (Max {MAX_POSITIONS} Positions)...")
    
    events = []
    for idx, row in trades_df.iterrows():
        events.append((row['entry_date'], 'entry', idx, dict(row)))
        events.append((row['exit_date'], 'exit', idx, dict(row)))
        
    # Sort events: primary key date, secondary key exit before entry
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'exit' else 1))
    
    available_positions = MAX_POSITIONS
    active_trades = set()
    executed_trades = []
    
    for date, event_type, idx, row in events:
        if event_type == 'exit' and idx in active_trades:
            active_trades.remove(idx)
            available_positions += 1
        elif event_type == 'entry' and available_positions > 0 and idx not in active_trades:
            available_positions -= 1
            active_trades.add(idx)
            executed_trades.append(row)
            
    if not executed_trades:
        logger.warning("No trades were executed after applying capital constraints!")
        return
        
    trades_df = pd.DataFrame(executed_trades)
    trades_df = trades_df.sort_values('exit_date')
    
    # Calculate Metrics
    trades_df['cumulative_pnl'] = trades_df['pnl_rupees'].cumsum()
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_rupees'] > 0])
    win_rate = winning_trades / total_trades * 100
    
    avg_win = trades_df[trades_df['pnl_rupees'] > 0]['pnl_rupees'].mean()
    avg_loss = trades_df[trades_df['pnl_rupees'] <= 0]['pnl_rupees'].mean()
    gross_win = trades_df[trades_df['pnl_rupees'] > 0]['pnl_rupees'].sum()
    gross_loss = trades_df[trades_df['pnl_rupees'] <= 0]['pnl_rupees'].sum()
    if pd.isna(gross_loss) or gross_loss == 0:
        profit_factor = 999.0
    else:
        profit_factor = abs(gross_win / gross_loss)
        
    final_pnl = trades_df['pnl_rupees'].sum()
    final_capital = INITIAL_CAPITAL + final_pnl
    return_pct = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Max Drawdown
    cumulative_capital = INITIAL_CAPITAL + trades_df['cumulative_pnl']
    peak = cumulative_capital.expanding(min_periods=1).max()
    drawdown = (cumulative_capital - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    logger.info("\n" + "="*60)
    logger.info(f" BACKTEST RESULTS ({len(data_files)} Stocks)")
    logger.info("="*60)
    logger.info(f"Total Trades:       {total_trades}")
    logger.info(f"Win Rate:           {win_rate:.2f}%")
    logger.info(f"Average Winner:     Rs.{avg_win:.2f}")
    logger.info(f"Average Loser:      Rs.{avg_loss:.2f}")
    logger.info(f"Profit Factor:      {profit_factor:.2f}")
    logger.info(f"Max Drawdown:       {max_drawdown:.2f}%")
    logger.info("-"*60)
    logger.info(f"Starting Capital:   Rs.{INITIAL_CAPITAL:,.2f}")
    logger.info(f"Ending Capital:     Rs.{final_capital:,.2f}")
    logger.info(f"Net Profit:         Rs.{final_pnl:,.2f} ({return_pct:.2f}%)")
    logger.info("="*60)
    
    # Save ledger
    os.makedirs(Path(PROJECT_ROOT) / 'data', exist_ok=True)
    ledger_path = Path(PROJECT_ROOT) / 'data' / 'backtest_ledger.csv'
    trades_df.to_csv(ledger_path, index=False)
    logger.info(f"Saved full chronological trade ledger to: {ledger_path}")

if __name__ == "__main__":
    run_backtest()
