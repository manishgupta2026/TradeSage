#!/usr/bin/env python3
"""
TradeSage Scanner Service
Background service that scans NSE stocks for trading signals during market hours.
Publishes signals to Redis for SSE streaming to the dashboard.

Features:
- NSE market hours check (9:15–15:30 IST, Mon–Fri)
- Angel One API with 3 req/sec rate limiter (token bucket)
- Redis pub/sub for live signal streaming
- Circuit filter: skip stocks where price=0 or volume=0
- Model hot-swap via symlink — reloads without restart
- Retry with exponential backoff + Telegram alerts on failure
"""

import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.core.feature_engineering import FeatureEngineer
from src.core.model_training import TradingModelTrainer

# ── Logging ──
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "scanner.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tradesage.scanner")


# ══════════════════════════════════════════════════════════════
#  TOKEN BUCKET RATE LIMITER — Angel One: max 3 req/sec
# ══════════════════════════════════════════════════════════════

class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, rate: float = 3.0, capacity: float = 3.0):
        self.rate = rate          # tokens per second
        self.capacity = capacity  # max burst
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


# ══════════════════════════════════════════════════════════════
#  TELEGRAM HELPER
# ══════════════════════════════════════════════════════════════

def send_telegram(message: str):
    """Best-effort Telegram notification."""
    import requests

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    # Fallback to angel_config.json
    if not token or not chat_id:
        cfg_path = PROJECT_ROOT / "config" / "angel_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                token = token or cfg.get("telegram_token")
                chat_id = chat_id or cfg.get("telegram_chat_id")
            except Exception:
                pass

    if not token or not chat_id:
        logger.info(f"[TELEGRAM STUB] {message}")
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ══════════════════════════════════════════════════════════════
#  MARKET HOURS CHECK
# ══════════════════════════════════════════════════════════════

IST = timezone(timedelta(hours=5, minutes=30))

def is_market_open() -> bool:
    """Check if current time is within NSE market hours (9:15–15:30 IST, Mon–Fri)."""
    now = datetime.now(IST)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    if weekday >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def next_market_open() -> datetime:
    """Calculate the next market open time."""
    now = datetime.now(IST)
    target = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    # Skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


# ══════════════════════════════════════════════════════════════
#  MODEL LOADER (supports hot-swap via symlink)
# ══════════════════════════════════════════════════════════════

class ModelManager:
    """Manages model loading with hot-swap support via symlink."""

    def __init__(self):
        self.trainer = TradingModelTrainer()
        self.engineer = FeatureEngineer()
        self.model_path = None
        self._last_mtime = 0

    def load(self, model_path: str = None):
        """Load model, preferring symlink models/current.pkl."""
        search_paths = [
            PROJECT_ROOT / "models" / "tradesage_10y.pkl",
            PROJECT_ROOT / "models" / "current.pkl",
            PROJECT_ROOT / "models" / "tradesage_v2.pkl",
            PROJECT_ROOT / "models" / "tradesage_angel.pkl",
            PROJECT_ROOT / "models" / "tradesage_model.pkl",
        ]
        if model_path:
            search_paths.insert(0, Path(model_path))

        for p in search_paths:
            if p.exists():
                self.model_path = p
                self.trainer.load_model(str(p))
                self._last_mtime = os.path.getmtime(p)
                logger.info(f"Model loaded: {p}")
                return True

        logger.error("No model file found!")
        return False

    def check_reload(self) -> bool:
        """Check if model file has been updated (hot-swap)."""
        if not self.model_path or not self.model_path.exists():
            return False
        current_mtime = os.path.getmtime(self.model_path)
        if current_mtime > self._last_mtime:
            logger.info("🔄 Model file changed — hot-swapping...")
            try:
                self.trainer.load_model(str(self.model_path))
                self._last_mtime = current_mtime
                logger.info("✅ Model hot-swap complete")
                send_telegram("🔄 Model hot-swapped in scanner (no restart)")
                return True
            except Exception as e:
                logger.error(f"Hot-swap failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════

def generate_signal(symbol: str, df: pd.DataFrame, model_mgr: ModelManager) -> dict:
    """Run feature engineering + model prediction for a single stock."""
    try:
        df = model_mgr.engineer.add_technical_indicators(df)
        df.dropna(inplace=True)

        if df.empty or len(df) < 50:
            return None

        latest = df.iloc[-1]
        current_price = float(latest["close"])
        current_volume = float(latest["volume"])

        # Circuit filter: skip stocks with price=0 or volume=0
        if current_price <= 0 or current_volume <= 0:
            return None

        atr = float(latest.get("atr", current_price * 0.02))
        if atr <= 0:
            atr = current_price * 0.02

        # Predict
        preds, probs = model_mgr.trainer.predict(df.iloc[[-1]])
        prob = float(probs[0])
        pred = int(preds[0])

        if pred != 1 or prob < 0.65:
            return None

        # Calculate trade levels
        stop_loss = round(current_price - (3.0 * atr), 2)
        take_profit = round(current_price + (3.5 * atr), 2)
        risk = current_price - stop_loss
        reward = take_profit - current_price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0

        confidence = "HIGH" if prob >= 0.75 else "MEDIUM"

        signal = {
            "timestamp": datetime.now(IST).isoformat(),
            "symbol": symbol,
            "probability": round(prob, 4),
            "signal": "BUY",
            "entry_price": round(current_price, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "r_r_ratio": rr_ratio,
            "atr": round(atr, 2),
            "confidence": confidence,
        }

        return signal

    except Exception as e:
        logger.debug(f"Signal gen failed for {symbol}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  FETCH DATA WITH RATE LIMITING & RETRY
# ══════════════════════════════════════════════════════════════

def fetch_stock_data(fetcher, symbol: str, rate_limiter: TokenBucketRateLimiter, period_days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV with rate limiting and exponential backoff retry."""
    max_retries = 3

    for attempt in range(max_retries):
        rate_limiter.acquire()
        try:
            df = fetcher.fetch_historical_data(symbol, period_days=period_days)
            if df is not None and len(df) >= 200:
                return df
            return None
        except Exception as e:
            err_str = str(e).lower()
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + 0.5
                if "exceeding" in err_str or "429" in err_str:
                    wait = max(wait, 2.0)
                logger.debug(f"Retry {attempt+1}/{max_retries} for {symbol}: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"Failed to fetch {symbol} after {max_retries} retries: {e}")
                return None
    return None


# ══════════════════════════════════════════════════════════════
#  PUBLISH SIGNAL TO REDIS
# ══════════════════════════════════════════════════════════════

def publish_signal(redis_client, signal: dict):
    """Publish signal to Redis pub/sub and append to history list."""
    try:
        signal_json = json.dumps(signal)

        # Pub/sub for SSE streaming
        redis_client.publish("tradesage:signals", signal_json)

        # Append to history list (capped at 500)
        redis_client.lpush("tradesage:signals_history", signal_json)
        redis_client.ltrim("tradesage:signals_history", 0, 499)

        logger.info(f"📡 Published: {signal['symbol']} P={signal['probability']:.2f} ({signal['confidence']})")
    except Exception as e:
        logger.error(f"Redis publish failed: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN SCAN LOOP
# ══════════════════════════════════════════════════════════════

def run_scanner():
    """Main scanner loop."""
    import redis as sync_redis

    logger.info("=" * 70)
    logger.info("  TRADESAGE SCANNER SERVICE")
    logger.info("=" * 70)

    # ── Connect to Redis ──
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = None
    try:
        redis_client = sync_redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info(f"✅ Redis connected: {redis_url}")
    except Exception as e:
        logger.warning(f"⚠️  Redis not available ({e}). Signals will be logged locally only.")
        redis_client = None

    # ── Load model ──
    model_mgr = ModelManager()
    if not model_mgr.load():
        logger.error("Cannot start scanner — no model found")
        send_telegram("🚨 Scanner failed to start: no model file found")
        sys.exit(1)

    # ── Connect to Angel One ──
    try:
        from src.angel.angel_one_api import AngelOneAPI
        from src.angel.angel_data_fetcher import AngelDataFetcher

        config_path = PROJECT_ROOT / "config" / "angel_config.json"
        alt_config = PROJECT_ROOT / "config" / "angel_one_config.json"
        cfg = str(alt_config if alt_config.exists() else config_path)

        api = AngelOneAPI(cfg)
        fetcher = AngelDataFetcher(api)
        logger.info("✅ Angel One API connected")
    except Exception as e:
        logger.error(f"Angel One connection failed: {e}")
        send_telegram(f"🚨 Scanner: Angel One connection failed — {e}")
        sys.exit(1)

    # ── Load watchlist ──
    watchlist_paths = [
        PROJECT_ROOT / "data" / "nse_top_3000_angel.json",
        PROJECT_ROOT / "data" / "nse_top_500_angel.json",
        PROJECT_ROOT / "data" / "nifty500.json",
        PROJECT_ROOT / "data" / "nifty200.json",
    ]
    watchlist = []
    for wp in watchlist_paths:
        if wp.exists():
            with open(wp) as f:
                watchlist = json.load(f)
            logger.info(f"Loaded {len(watchlist)} symbols from {wp.name}")
            break

    if not watchlist:
        logger.error("No watchlist found")
        sys.exit(1)

    # ── Rate limiter: 3 req/sec ──
    rate_limiter = TokenBucketRateLimiter(rate=3.0, capacity=3.0)

    # ── Scan interval ──
    SCAN_INTERVAL_MINUTES = 15
    send_telegram(f"🟢 TradeSage Scanner started | {len(watchlist)} stocks | {SCAN_INTERVAL_MINUTES}min interval")

    # ── Main loop ──
    while True:
        try:
            # Check for model hot-swap
            model_mgr.check_reload()

            if is_market_open():
                logger.info(f"\n{'═' * 60}")
                logger.info(f"SCAN STARTING — {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
                logger.info(f"{'═' * 60}")

                scan_start = time.time()
                signal_count = 0
                high_conf_count = 0
                errors = 0

                # Save local signals for fallback
                local_signals = []

                for i, symbol in enumerate(watchlist, 1):
                    if i % 100 == 0:
                        logger.info(f"  [{i}/{len(watchlist)}] scanning...")

                    df = fetch_stock_data(fetcher, symbol, rate_limiter)
                    if df is None:
                        errors += 1
                        continue

                    signal = generate_signal(symbol, df, model_mgr)
                    if signal:
                        local_signals.append(signal)

                    # Check if market closed during scan
                    if not is_market_open():
                        logger.info("Market closed during scan — stopping early")
                        break

                elapsed = time.time() - scan_start

                # --- FUNDAMENTAL FILTERING ---
                if local_signals:
                    logger.info(f"\n--- Technical Scan Complete. Running Fundamental Filter on top candidates ---")
                    local_signals.sort(key=lambda x: x['probability'], reverse=True)
                    candidate_pool = local_signals[:10]  # Only top 10 tradable stocks
                    
                    finals = []
                    from src.core.fundamental_analyzer import FundamentalAnalyzer
                    analyzer = FundamentalAnalyzer()
                    
                    for sig in candidate_pool:
                        flags = analyzer.evaluate_candidate(sig['symbol'])
                        if flags:  # Dictionary returned on success
                            sig['fundamentals'] = flags
                            finals.append(sig)
                    
                    # Overwrite local_signals with the finalized batch
                    local_signals = finals
                    signal_count = len(local_signals)
                    high_conf_count = sum(1 for s in local_signals if s["confidence"] == "HIGH")
                    
                    # Now publish the elite survivors
                    for sig in local_signals:
                        if redis_client:
                            publish_signal(redis_client, sig)
                        logger.info(
                            f"  🟢 {sig['symbol']:>12s}  P={sig['probability']:.2f}  "
                            f"Entry=₹{sig['entry_price']:,.2f}  "
                            f"SL=₹{sig['stop_loss']:,.2f}  "
                            f"TP=₹{sig['take_profit']:,.2f}  "
                            f"R:R={sig['r_r_ratio']}  "
                            f"[TV:{sig['fundamentals'].get('tv_rating', 'N/A')} | News:{sig['fundamentals'].get('sentiment', 'N/A')}]"
                        )

                # Update last scan timestamp
                if redis_client:
                    try:
                        redis_client.set("tradesage:last_scan", datetime.now(IST).strftime("%H:%M:%S"))
                    except Exception:
                        pass

                # Save local fallback
                local_signals_path = PROJECT_ROOT / "data" / "live_signals.json"
                try:
                    with open(local_signals_path, "w") as f:
                        json.dump(local_signals, f, indent=2)
                except Exception:
                    pass

                # --- AUTONOMOUS PAPER TRADING ---
                # Initial capital setup: 50k INR | Max alloc per trade: 10k INR
                positions_path = PROJECT_ROOT / "data" / "positions.json"
                if not positions_path.exists():
                    try:
                        with open(positions_path, "w") as f:
                            json.dump({}, f)
                    except Exception:
                        pass
                
                try:
                    with open(positions_path, "r") as f:
                        positions = json.load(f)
                except Exception:
                    positions = {}
                
                new_trades_count = 0
                for sig in top_signals:
                    if sig['probability'] >= 0.75:
                        sym = sig['symbol']
                        if sym not in positions or positions[sym].get('status') != 'open':
                            entry = sig['entry_price']
                            shares = int(10000 // entry) if entry > 0 else 0
                            if shares > 0:
                                positions[sym] = {
                                    "status": "open",
                                    "entry_price": entry,
                                    "shares": shares,
                                    "stop_loss": sig['stop_loss'],
                                    "take_profit": sig['take_profit'],
                                    "entry_date": datetime.now(IST).isoformat(),
                                    "confidence": sig['confidence'],
                                    "fundamentals": sig.get('fundamentals', {})
                                }
                                new_trades_count += 1
                                logger.info(f"🤖 [AUTO-TRADE] Executed {shares} shares of {sym} at ₹{entry:.2f}")

                if new_trades_count > 0:
                    try:
                        with open(positions_path, "w") as f:
                            json.dump(positions, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save paper trades: {e}")

                logger.info(f"\n{'─' * 60}")
                logger.info(
                    f"SCAN COMPLETE — {signal_count} signals ({high_conf_count} HIGH) | "
                    f"{errors} errors | {elapsed:.1f}s"
                )
                logger.info(f"{'─' * 60}")

                if signal_count > 0:
                    summary_msg = f"📊 *TradeSage Scan Complete*\n"
                    summary_msg += f"✅ Found {signal_count} signals ({high_conf_count} HIGH)\n\n"
                    
                    # Log top 5 signals to Telegram
                    top_signals = local_signals[:5]
                    for s in top_signals:
                        sym = s['symbol']
                        entry = s['entry_price']
                        tp = s['take_profit']
                        sl = s['stop_loss']
                        tv = s.get('fundamentals', {}).get('tv_rating', 'N/A')
                        news = s.get('fundamentals', {}).get('sentiment', 'N/A')
                        prob = s['probability'] * 100
                        conf = '🔥' if prob >= 75 else '🟢'
                        
                        summary_msg += f"{conf} *{sym}* - {s['signal']}\n"
                        summary_msg += f"▸ Entry: ₹{entry:.2f} | P: {prob:.1f}%\n"
                        summary_msg += f"▸ TP: ₹{tp:.2f} | SL: ₹{sl:.2f}\n"
                        summary_msg += f"▸ TV: {tv} | News: {news}\n\n"
                        
                    summary_msg += f"⏱ Time: {elapsed:.0f}s | Errors: {errors}"
                    send_telegram(summary_msg)

                # Wait for next scan interval
                logger.info(f"Next scan in {SCAN_INTERVAL_MINUTES} minutes...")
                time.sleep(SCAN_INTERVAL_MINUTES * 60)

            else:
                next_open = next_market_open()
                wait_seconds = (next_open - datetime.now(IST)).total_seconds()
                wait_hours = wait_seconds / 3600

                logger.info(
                    f"🌙 Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M IST')} "
                    f"({wait_hours:.1f}h)"
                )

                # Sleep in chunks to allow graceful shutdown
                sleep_chunk = min(wait_seconds, 300)  # Max 5 min chunks
                time.sleep(sleep_chunk)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user")
            send_telegram("🔴 TradeSage Scanner stopped")
            break
        except Exception as e:
            logger.error(f"Scanner error: {e}", exc_info=True)
            send_telegram(f"🚨 Scanner error: {e}")
            time.sleep(60)  # Wait 1 min before retry


if __name__ == "__main__":
    run_scanner()
