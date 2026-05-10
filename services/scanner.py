#!/usr/bin/env python3
"""
TradeSage Scanner Service
Background service that scans NSE stocks for trading signals during market hours.
Publishes signals to Redis for SSE streaming to the dashboard.

Features:
- NSE market hours check (9:15–15:30 IST, Mon–Fri)
- Angel One API with 3 req/sec rate limiter (token bucket)
- AUTO SESSION RECONNECT — re-authenticates on token expiry
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
#  ANGEL ONE API MANAGER — Auto-reconnect on session expiry
# ══════════════════════════════════════════════════════════════

class AngelSessionManager:
    """
    Wraps AngelOneAPI + AngelDataFetcher with automatic session 
    reconnection when token expires. Angel One sessions expire daily.
    """

    def __init__(self):
        self.api = None
        self.fetcher = None
        self._last_connect = None
        self._connect_lock = threading.Lock()
        self._consecutive_failures = 0
        self._MAX_FAILURES_BEFORE_RECONNECT = 5

    def connect(self) -> bool:
        """Establish or re-establish Angel One connection."""
        with self._connect_lock:
            try:
                from src.angel.angel_one_api import AngelOneAPI
                from src.angel.angel_data_fetcher import AngelDataFetcher

                config_path = PROJECT_ROOT / "config" / "angel_config.json"
                alt_config = PROJECT_ROOT / "config" / "angel_one_config.json"
                cfg = str(alt_config if alt_config.exists() else config_path)

                logger.info(f"🔑 Connecting to Angel One... (config: {Path(cfg).name})")
                self.api = AngelOneAPI(cfg)
                self.fetcher = AngelDataFetcher(self.api)
                self._last_connect = datetime.now(IST)
                self._consecutive_failures = 0
                logger.info("✅ Angel One API connected successfully")
                return True

            except Exception as e:
                logger.error(f"❌ Angel One connection failed: {e}")
                send_telegram(f"🚨 Scanner: Angel One connection failed — {e}")
                return False

    def reconnect_if_needed(self) -> bool:
        """
        Check if we need to reconnect:
        1. Too many consecutive fetch failures (session likely expired)
        2. Session is older than 8 hours (proactive refresh)
        """
        needs_reconnect = False

        if self._consecutive_failures >= self._MAX_FAILURES_BEFORE_RECONNECT:
            logger.warning(
                f"⚠️ {self._consecutive_failures} consecutive failures — "
                f"session likely expired. Reconnecting..."
            )
            needs_reconnect = True

        if self._last_connect:
            age_hours = (datetime.now(IST) - self._last_connect).total_seconds() / 3600
            if age_hours >= 8:
                logger.info(f"🔄 Session is {age_hours:.1f}h old — proactive reconnect")
                needs_reconnect = True

        if needs_reconnect:
            send_telegram("🔄 Scanner: Reconnecting Angel One session...")
            return self.connect()

        return True

    def record_success(self):
        """Record a successful API call."""
        self._consecutive_failures = 0

    def record_failure(self):
        """Record a failed API call."""
        self._consecutive_failures += 1

    @property
    def is_connected(self) -> bool:
        return self.api is not None and self.fetcher is not None


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

        logger.info("🔍 Searching for model file...")
        for p in search_paths:
            logger.info(f"  Checking: {p} — {'EXISTS' if p.exists() else 'not found'}")
            if p.exists():
                self.model_path = p
                self.trainer.load_model(str(p))
                self._last_mtime = os.path.getmtime(p)
                logger.info(f"✅ Model loaded: {p} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
                return True

        logger.error("❌ No model file found in any search path!")
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
        df = model_mgr.engineer.add_technical_indicators(df, symbol=symbol)
        df.dropna(inplace=True)

        if df.empty or len(df) < 10:
            return None

        latest = df.iloc[-1]
        current_price = float(latest["close"])
        current_volume = float(latest["volume"])

        # Circuit filter: skip stocks with price=0 or volume=0
        if current_price <= 0 or current_volume <= 0:
            return None

        # Quality filter: skip penny stocks and illiquid instruments
        MIN_STOCK_PRICE = 50   # ₹50 minimum to avoid penny stock manipulation
        MIN_VOLUME = 100000    # 1 lakh minimum daily volume for liquidity
        if current_price < MIN_STOCK_PRICE:
            return None
        if current_volume < MIN_VOLUME:
            return None

        atr = float(latest.get("atr", current_price * 0.02))
        if atr <= 0:
            atr = current_price * 0.02

        # Predict
        preds, probs = model_mgr.trainer.predict(df.iloc[[-1]])
        prob = float(probs[0])
        pred = int(preds[0])

        if pred != 1 or prob < 0.55:
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
#  FETCH DATA WITH RATE LIMITING, RETRY & SESSION RECOVERY
# ══════════════════════════════════════════════════════════════

def fetch_stock_data(
    angel_mgr: AngelSessionManager,
    symbol: str,
    rate_limiter: TokenBucketRateLimiter,
    period_days: int = 365,
) -> pd.DataFrame:
    """Fetch OHLCV with rate limiting, retry, and session recovery."""
    max_retries = 3

    for attempt in range(max_retries):
        rate_limiter.acquire()
        try:
            df = angel_mgr.fetcher.fetch_historical_data(symbol, period_days=period_days)
            if df is not None and len(df) >= 200:
                angel_mgr.record_success()
                return df
            return None
        except Exception as e:
            err_str = str(e).lower()
            angel_mgr.record_failure()

            # Check for session expiry indicators
            if any(kw in err_str for kw in [
                "invalid token", "session expired", "unauthorized",
                "jwt", "token", "login", "auth"
            ]):
                logger.warning(f"🔑 Session likely expired during {symbol} fetch. Will reconnect...")
                if angel_mgr.reconnect_if_needed():
                    continue  # Retry with new session

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


def fetch_stock_data_cached(
    angel_mgr: AngelSessionManager,
    symbol: str,
    rate_limiter: TokenBucketRateLimiter,
    redis_client=None,
    period_days: int = 365,
) -> pd.DataFrame:
    """Fetch OHLCV with Redis cache (15-min TTL). Daily candles don't change intraday."""
    cache_key = f"tradesage:ohlc:{symbol}"

    # Try Redis cache first
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                df = pd.read_json(cached)
                if df is not None and len(df) >= 200:
                    return df
        except Exception:
            pass  # Cache miss or corrupt — fetch fresh

    # Cache miss — fetch from API
    df = fetch_stock_data(angel_mgr, symbol, rate_limiter, period_days)

    # Store in cache with 15-min TTL
    if df is not None and redis_client:
        try:
            redis_client.setex(cache_key, 900, df.to_json())
        except Exception:
            pass  # Non-critical

    return df


def _scan_single_stock(symbol, angel_mgr, rate_limiter, model_mgr, ltp_cache, redis_client=None):
    """Fetch data + generate signal for one stock. Thread-safe worker."""
    try:
        df = fetch_stock_data_cached(angel_mgr, symbol, rate_limiter, redis_client)
        if df is None:
            return symbol, None, True  # error

        # Inject live LTP from pre-fetched cache
        ltp = ltp_cache.get(symbol)
        if ltp and ltp > 0:
            df.iloc[-1, df.columns.get_loc('close')] = ltp

        signal = generate_signal(symbol, df, model_mgr)
        return symbol, signal, False
    except Exception as e:
        logger.debug(f"Worker error for {symbol}: {e}")
        return symbol, None, True


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
    logger.info("  TRADESAGE SCANNER SERVICE v2")
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

    # ── Connect to Angel One (with auto-reconnect) ──
    angel_mgr = AngelSessionManager()
    if not angel_mgr.connect():
        # Don't exit — we'll retry on the next scan cycle
        logger.error("⚠️ Initial Angel One connection failed — will retry on next scan cycle")
        send_telegram("⚠️ Scanner: Initial Angel One connection failed — will retry")

    # ── Load watchlists (tiered scanning) ──
    # Priority tier: Top 500 stocks — scanned every cycle (15 min)
    # Extended tier: Full 3000 stocks — scanned every 4th cycle (60 min)
    priority_watchlist = []
    full_watchlist = []

    # Load full watchlist
    full_paths = [
        PROJECT_ROOT / "data" / "nse_top_3000_angel.json",
        PROJECT_ROOT / "data" / "nse_top_2000_angel.json",
        PROJECT_ROOT / "data" / "nse_1200.json",
    ]
    for wp in full_paths:
        if wp.exists():
            with open(wp) as f:
                full_watchlist = json.load(f)
            logger.info(f"Loaded FULL watchlist: {len(full_watchlist)} symbols from {wp.name}")
            break

    # Load priority watchlist
    priority_paths = [
        PROJECT_ROOT / "data" / "nse_top_500_angel.json",
        PROJECT_ROOT / "data" / "nifty500.json",
        PROJECT_ROOT / "data" / "nifty200.json",
    ]
    for wp in priority_paths:
        if wp.exists():
            with open(wp) as f:
                priority_watchlist = json.load(f)
            logger.info(f"Loaded PRIORITY watchlist: {len(priority_watchlist)} symbols from {wp.name}")
            break

    # Fallback: if no priority list, use full list
    if not priority_watchlist:
        priority_watchlist = full_watchlist
    if not full_watchlist:
        full_watchlist = priority_watchlist

    if not full_watchlist and not priority_watchlist:
        logger.error("No watchlist found")
        sys.exit(1)

    # ── Rate limiter: 3 req/sec ──
    rate_limiter = TokenBucketRateLimiter(rate=3.0, capacity=3.0)

    # ── Scan config ──
    SCAN_INTERVAL_MINUTES = 15
    PARALLEL_WORKERS = 6  # Threads for parallel fetch (rate limiter gates actual API calls)
    scan_cycle_count = 0

    send_telegram(
        f"🟢 TradeSage Scanner v3 started\\n"
        f"⚡ Priority: {len(priority_watchlist)} stocks (every 15 min)\\n"
        f"📊 Full: {len(full_watchlist)} stocks (every 60 min)\\n"
        f"🔀 Parallel workers: {PARALLEL_WORKERS}\\n"
        f"💾 Redis OHLC caching: ON"
    )

    # ── Main loop ──
    while True:
        try:
            # Check for model hot-swap
            model_mgr.check_reload()

            # Check for manual scan trigger
            force_scan = False
            if redis_client:
                try:
                    if redis_client.get("tradesage:force_scan") == "1":
                        force_scan = True
                        redis_client.delete("tradesage:force_scan")
                        logger.info("⚡ Force scan triggered via Redis!")
                except Exception:
                    pass

            if is_market_open() or force_scan:
                # ── Pre-scan: ensure Angel One session is fresh ──
                if not angel_mgr.is_connected:
                    logger.info("🔑 Angel One not connected — attempting connection...")
                    if redis_client: redis_client.publish("tradesage:signals", "Attempting Angel One connection...")
                    if not angel_mgr.connect():
                        logger.error("Angel One connection failed — skipping this scan cycle")
                        if redis_client: redis_client.publish("tradesage:signals", "Angel One connection failed — skipping scan")
                        time.sleep(60)
                        continue

                # Proactive reconnect if session is old
                angel_mgr.reconnect_if_needed()

                # ── Tiered watchlist selection ──
                scan_cycle_count += 1
                is_full_scan = (scan_cycle_count % 4 == 0)  # Every 4th cycle = full scan
                current_watchlist = full_watchlist if is_full_scan else priority_watchlist
                scan_tier = "FULL" if is_full_scan else "PRIORITY"

                msg = f"SCAN STARTING [{scan_tier}] — {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')} — {len(current_watchlist)} stocks"
                logger.info(f"\n{'═' * 60}")
                logger.info(msg)
                logger.info(f"{'═' * 60}")
                if redis_client: redis_client.publish("tradesage:signals", f"Started {scan_tier} scan of {len(current_watchlist)} stocks...")

                scan_start = time.time()
                signal_count = 0
                high_conf_count = 0
                errors = 0
                successful_fetches = 0

                # Save local signals for fallback
                local_signals = []

                # ── Phase 1: Parallel historical data fetch + signal generation ──
                # Pre-fetch LTPs in batches for injection (done in parallel with historical fetch)
                ltp_cache = {}

                completed = 0
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                    futures = {
                        executor.submit(
                            _scan_single_stock, sym, angel_mgr, rate_limiter,
                            model_mgr, ltp_cache, redis_client
                        ): sym
                        for sym in current_watchlist
                    }

                    for future in as_completed(futures):
                        try:
                            symbol, signal, is_error = future.result(timeout=60)
                        except Exception as e:
                            logger.debug(f"Future error: {e}")
                            errors += 1
                            completed += 1
                            continue

                        completed += 1

                        if is_error:
                            errors += 1
                        else:
                            successful_fetches += 1
                            if signal:
                                local_signals.append(signal)

                        # Progress logging
                        if completed % 100 == 0:
                            progress_msg = f"[{completed}/{len(current_watchlist)}] scanning... (signals: {len(local_signals)}, errors: {errors})"
                            logger.info(progress_msg)
                            if redis_client: redis_client.publish("tradesage:signals", progress_msg)

                        # Mid-scan session recovery
                        if angel_mgr._consecutive_failures >= angel_mgr._MAX_FAILURES_BEFORE_RECONNECT:
                            logger.warning("🔄 Too many failures mid-scan — reconnecting session...")
                            angel_mgr.connect()

                # ── Phase 2: Batch LTP injection for signal accuracy ──
                # Re-fetch LTPs for stocks that generated signals (most accurate pricing)
                if local_signals and angel_mgr.is_connected:
                    signal_symbols = [s['symbol'] for s in local_signals]
                    logger.info(f"📡 Batch-fetching LTPs for {len(signal_symbols)} signal stocks...")
                    for sym in signal_symbols:
                        try:
                            rate_limiter.acquire()
                            ltp = angel_mgr.api.get_ltp(sym)
                            if ltp and ltp > 0:
                                # Update the signal's entry price with live LTP
                                for sig in local_signals:
                                    if sig['symbol'] == sym:
                                        sig['entry_price'] = round(ltp, 2)
                                        # Recalculate SL/TP based on new price
                                        atr = sig.get('atr', ltp * 0.02)
                                        sig['stop_loss'] = round(ltp - (3.0 * atr), 2)
                                        sig['take_profit'] = round(ltp + (3.5 * atr), 2)
                                        break
                        except Exception as e:
                            logger.debug(f"LTP batch fetch failed for {sym}: {e}")

                elapsed = time.time() - scan_start

                # Log fetch success rate
                total_attempted = successful_fetches + errors
                if total_attempted > 0:
                    success_rate = (successful_fetches / total_attempted) * 100
                    logger.info(f"📊 Fetch success rate: {success_rate:.1f}% ({successful_fetches}/{total_attempted})")

                    # If success rate is very low, the session is probably dead
                    if success_rate < 10 and total_attempted > 50:
                        logger.error("🚨 Fetch success rate critically low — forcing reconnection")
                        angel_mgr.connect()
                        send_telegram(f"⚠️ Scanner: Low fetch success rate ({success_rate:.0f}%). Reconnected session.")

                # --- FUNDAMENTAL FILTERING ---
                if local_signals:
                    logger.info(f"\n--- Technical Scan Complete. Running Fundamental Filter on top candidates ---")
                    local_signals.sort(key=lambda x: x['probability'], reverse=True)
                    candidate_pool = local_signals[:10]  # Only top 10 tradable stocks
                    
                    finals = []
                    try:
                        from src.core.fundamental_analyzer import FundamentalAnalyzer
                        analyzer = FundamentalAnalyzer()
                        
                        for sig in candidate_pool:
                            try:
                                flags = analyzer.evaluate_candidate(sig['symbol'])
                                if flags:  # Dictionary returned on success
                                    sig['fundamentals'] = flags
                                    finals.append(sig)
                            except Exception as e:
                                # Don't block signal on fundamental analysis failure
                                logger.debug(f"Fundamental analysis failed for {sig['symbol']}: {e}")
                                finals.append(sig)
                    except ImportError:
                        logger.warning("FundamentalAnalyzer not available — skipping fundamental filter")
                        finals = candidate_pool
                    
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
                            f"[TV:{sig.get('fundamentals', {}).get('tv_rating', 'N/A')} | "
                            f"News:{sig.get('fundamentals', {}).get('sentiment', 'N/A')}]"
                        )

                # Update last scan timestamp
                if redis_client:
                    try:
                        redis_client.set("tradesage:last_scan", datetime.now(IST).strftime("%H:%M:%S"))
                        redis_client.set("tradesage:scan_stats", json.dumps({
                            "timestamp": datetime.now(IST).isoformat(),
                            "signals": signal_count,
                            "high_conf": high_conf_count,
                            "errors": errors,
                            "fetched": successful_fetches,
                            "elapsed": round(elapsed, 1),
                            "tier": scan_tier,
                            "cycle": scan_cycle_count,
                            "watchlist_size": len(current_watchlist),
                        }))
                    except Exception:
                        pass

                # Save local fallback
                local_signals_path = PROJECT_ROOT / "data" / "live_signals.json"
                try:
                    with open(local_signals_path, "w") as f:
                        json.dump(local_signals, f, indent=2)
                except Exception:
                    pass

                # --- AUTONOMOUS PAPER TRADING & SIDEWAYS DETECTION ---
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
                    
                # 1. Active Position Monitoring (Sideways & TSL)
                positions_modified = False
                for sym, p in positions.items():
                    if p.get('status') == 'open':
                        try:
                            # Handle datetime parsing carefully
                            dt_str = p.get('entry_date', datetime.now(IST).isoformat())
                            dt_clean = dt_str.split('.')[0].split('+')[0]
                            entry_date = datetime.strptime(dt_clean, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=IST)
                            days_held = (datetime.now(IST) - entry_date).days
                            
                            df = angel_mgr.fetcher.fetch_historical_data(sym, period_days=20)
                            if df is not None and not df.empty and len(df) >= 14:
                                # Calculate ATR(14)
                                df['prev_close'] = df['close'].shift(1)
                                df['tr'] = df[['high', 'low', 'prev_close']].apply(
                                    lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']) if not pd.isna(x['prev_close']) else 0, abs(x['low'] - x['prev_close']) if not pd.isna(x['prev_close']) else 0), axis=1
                                )
                                atr_14 = df['tr'].rolling(window=14).mean().iloc[-1]
                                current_close = df.iloc[-1]['close']
                                entry_price = p.get('entry_price', current_close)
                                
                                # Trailing Stop-Loss Logic (3x ATR)
                                if atr_14 and atr_14 > 0 and not pd.isna(atr_14):
                                    current_sl = p.get('stop_loss', 0)
                                    tsl_price = current_close - (3 * atr_14)
                                    
                                    # Activate TSL if it's higher than current SL and we are in profit
                                    if current_close > entry_price and tsl_price > current_sl:
                                        p['stop_loss'] = round(tsl_price, 2)
                                        positions_modified = True
                                        msg = f"🛡️ *TSL Updated*: {sym} SL raised to ₹{tsl_price:,.2f} (3x ATR)"
                                        logger.info(msg.replace('*', ''))
                                        send_telegram(msg)
                                        
                                # Sideways Market Logic
                                if days_held >= 10:
                                    last_14 = df.tail(14)
                                    max_close = last_14['close'].max()
                                    min_close = last_14['close'].min()
                                    
                                    if entry_price > 0 and min_close > 0:
                                        pnl_pct = ((current_close - entry_price) / entry_price) * 100
                                        price_range_pct = (max_close - min_close) / min_close
                                        
                                        is_sideways = (pnl_pct > 0 and price_range_pct < 0.05)
                                        if is_sideways != p.get('sideways_suggestion', False):
                                            p['sideways_suggestion'] = is_sideways
                                            positions_modified = True
                                            if is_sideways:
                                                msg = f"💤 *Sideways Market*: {sym} is in profit ({pnl_pct:.1f}%) but flat. Consider taking profit."
                                                logger.info(msg.replace('*', ''))
                                                send_telegram(msg)
                        except Exception as e:
                            logger.warning(f"Error checking position monitors for {sym}: {e}")
                # Calculate deployed capital
                deployed_capital = 0
                for p in positions.values():
                    if p.get('status') == 'open':
                        deployed_capital += p.get('entry_price', 0) * p.get('shares', 0)
                available_cash = 50000 - deployed_capital
                
                new_trades_count = 0
                for sig in local_signals:
                    if sig['probability'] >= 0.75:
                        if available_cash <= 2000:
                            logger.info(f"Skipping {sig['symbol']} - Insufficient capital (₹{available_cash:,.2f})")
                            continue
                            
                        sym = sig['symbol']
                        if sym not in positions or positions[sym].get('status') != 'open':
                            entry = sig['entry_price']
                            sl = sig['stop_loss']
                            
                            # ATR-Based Risk Allocation (Target exactly ₹1000 risk per trade)
                            risk_per_trade = 1000
                            distance = entry - sl
                            if distance <= 0:
                                distance = entry * 0.05  # Fallback to 5% SL distance
                                
                            target_shares = int(risk_per_trade // distance)
                            required_capital = target_shares * entry
                            
                            # Cap allocation to ₹10,000 max per trade AND available cash
                            max_capital = min(10000, available_cash)
                            if required_capital > max_capital:
                                required_capital = max_capital
                                target_shares = int(required_capital // entry)
                                
                            shares = target_shares if entry > 0 else 0
                            
                            if shares > 0:
                                available_cash -= (shares * entry) # Deduct for next iterations
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

                if new_trades_count > 0 or positions_modified:
                    try:
                        with open(positions_path, "w") as f:
                            json.dump(positions, f, indent=2)
                    except Exception as e:
                        logger.error(f"Failed to save paper trades: {e}")

                logger.info(f"\n{'─' * 60}")
                comp_msg = f"SCAN COMPLETE [{scan_tier}] — {signal_count} signals ({high_conf_count} HIGH) | {errors} errors | {elapsed:.1f}s"
                logger.info(comp_msg)
                logger.info(f"{'─' * 60}")
                if redis_client: redis_client.publish("tradesage:signals", comp_msg)

                if signal_count > 0:
                    summary_msg = f"📊 *TradeSage Scan Complete* [{scan_tier}]\n"
                    summary_msg += f"🕐 {datetime.now(IST).strftime('%d %b %Y, %I:%M %p IST')}\n"
                    summary_msg += f"⚡ Scanned {successful_fetches} stocks in {elapsed:.0f}s\n"
                    summary_msg += f"✅ Found *{signal_count} signals* ({high_conf_count} HIGH)\n"
                    summary_msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    
                    top_signals = local_signals[:5]
                    for idx, s in enumerate(top_signals, 1):
                        sym = s['symbol']
                        entry = s['entry_price']
                        tp = s['take_profit']
                        sl = s['stop_loss']
                        rr = s.get('r_r_ratio', 0)
                        prob = s['probability'] * 100
                        conf = '🔥' if prob >= 75 else '🟢'
                        
                        # Fundamental data
                        fund = s.get('fundamentals', {})
                        tv = fund.get('tv_rating', 'N/A')
                        news = fund.get('sentiment', 'N/A')
                        pe = fund.get('pe_ratio', 'N/A')
                        
                        summary_msg += f"{conf} *#{idx} {sym}* — {s['signal']}\n"
                        summary_msg += f"   📈 Entry: ₹{entry:,.2f} | Prob: {prob:.0f}%\n"
                        summary_msg += f"   🎯 TP: ₹{tp:,.2f} | 🛑 SL: ₹{sl:,.2f}\n"
                        summary_msg += f"   ⚖️ R:R = {rr} | P/E: {pe}\n"
                        summary_msg += f"   📊 TV: {tv} | 📰 News: {news}\n\n"
                    
                    if new_trades_count > 0:
                        summary_msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        summary_msg += f"🤖 *Auto Paper Trades: {new_trades_count} new positions opened*\n\n"
                        for sig in local_signals:
                            if sig['probability'] >= 0.75:
                                sym = sig['symbol']
                                if sym in positions and positions[sym].get('status') == 'open':
                                    shares = positions[sym].get('shares', 0)
                                    cost = shares * sig['entry_price']
                                    summary_msg += f"   💰 {sym}: {shares} shares @ ₹{sig['entry_price']:,.2f} (₹{cost:,.0f})\n"
                    
                    summary_msg += f"\n⏱ Scan time: {elapsed:.0f}s | Stocks scanned: {successful_fetches} | Errors: {errors}"
                    send_telegram(summary_msg)
                else:
                    # Notify that scan completed with no signals
                    no_sig_msg = (
                        f"📊 *TradeSage Scan Complete*\n"
                        f"🕐 {datetime.now(IST).strftime('%d %b %Y, %I:%M %p IST')}\n\n"
                        f"⚪ No qualifying signals this cycle\n"
                        f"📉 Scanned {successful_fetches} stocks in {elapsed:.0f}s\n"
                        f"🔄 Next scan in {SCAN_INTERVAL_MINUTES} min"
                    )
                    send_telegram(no_sig_msg)
                    logger.info("No qualifying signals this scan cycle")

                # Wait for next scan interval
                logger.info(f"Next scan in {SCAN_INTERVAL_MINUTES} minutes...")
                wait_seconds = SCAN_INTERVAL_MINUTES * 60
                
                # Sleep in 5-second chunks to allow interruption by force_scan
                for _ in range(int(wait_seconds / 5)):
                    if redis_client and redis_client.get("tradesage:force_scan") == "1":
                        break
                    time.sleep(5)

            else:
                # ── Daily Equity Snapshot (runs once per day after market close) ──
                try:
                    equity_path = PROJECT_ROOT / "data" / "equity_history.json"
                    today_str = datetime.now(IST).strftime("%Y-%m-%d")
                    
                    # Load existing history
                    history = []
                    if equity_path.exists():
                        with open(equity_path) as f:
                            history = json.load(f)
                    
                    # Check if we already have a snapshot for today
                    already_snapped = any(h.get("date") == today_str for h in history)
                    
                    if not already_snapped:
                        # Calculate portfolio value
                        positions_path = PROJECT_ROOT / "data" / "positions.json"
                        positions = {}
                        if positions_path.exists():
                            with open(positions_path) as f:
                                positions = json.load(f)
                        
                        deployed = 0
                        unrealized_pnl = 0
                        realized_pnl = 0
                        
                        for sym, p in positions.items():
                            if p.get("status") == "open":
                                entry_price = p.get("entry_price", 0)
                                shares = p.get("shares", 0)
                                deployed += entry_price * shares
                                # Try to get live price for unrealized P&L
                                try:
                                    if angel_mgr.is_connected:
                                        ltp = angel_mgr.fetcher.get_ltp(sym)
                                        if ltp and ltp > 0:
                                            unrealized_pnl += (ltp - entry_price) * shares
                                except Exception:
                                    pass
                            elif p.get("status") == "closed":
                                ep = p.get("entry_price", 0)
                                xp = p.get("exit_price", 0)
                                sh = p.get("shares", 0)
                                if ep > 0 and xp > 0 and sh > 0:
                                    realized_pnl += (xp - ep) * sh
                        
                        initial_capital = 50000
                        available_cash = initial_capital + realized_pnl - deployed
                        total_value = available_cash + deployed + unrealized_pnl
                        
                        # Fetch Nifty 50 value
                        nifty_value = None
                        try:
                            if angel_mgr.is_connected:
                                nifty_ltp = angel_mgr.fetcher.get_ltp("NIFTY")
                                if nifty_ltp and nifty_ltp > 0:
                                    nifty_value = round(nifty_ltp, 2)
                        except Exception as e:
                            logger.warning(f"Could not fetch Nifty 50 for snapshot: {e}")
                        
                        # If we couldn't get Nifty from API, try yfinance as fallback
                        if nifty_value is None:
                            try:
                                import yfinance as yf
                                nifty = yf.Ticker("^NSEI")
                                hist = nifty.history(period="1d")
                                if not hist.empty:
                                    nifty_value = round(hist['Close'].iloc[-1], 2)
                            except Exception as e:
                                logger.warning(f"yfinance Nifty fallback failed: {e}")
                        
                        snapshot = {
                            "date": today_str,
                            "portfolio_value": round(total_value, 2),
                            "deployed_capital": round(deployed, 2),
                            "available_cash": round(available_cash, 2),
                            "realized_pnl": round(realized_pnl, 2),
                            "unrealized_pnl": round(unrealized_pnl, 2),
                            "nifty_50": nifty_value,
                            "active_positions": sum(1 for p in positions.values() if p.get("status") == "open"),
                        }
                        
                        history.append(snapshot)
                        
                        with open(equity_path, "w") as f:
                            json.dump(history, f, indent=2)
                        
                        logger.info(f"📸 Daily equity snapshot saved: ₹{total_value:,.2f} | Nifty: {nifty_value}")
                        send_telegram(
                            f"📸 *Daily Portfolio Snapshot*\n"
                            f"📅 {today_str}\n"
                            f"💰 Portfolio Value: ₹{total_value:,.2f}\n"
                            f"📈 Nifty 50: {nifty_value or 'N/A'}\n"
                            f"💵 Cash: ₹{available_cash:,.2f}\n"
                            f"🔓 Deployed: ₹{deployed:,.2f}\n"
                            f"📊 Realized P&L: ₹{realized_pnl:,.2f}\n"
                            f"📊 Unrealized P&L: ₹{unrealized_pnl:,.2f}"
                        )
                except Exception as e:
                    logger.warning(f"Equity snapshot error: {e}")

                next_open = next_market_open()
                wait_seconds = (next_open - datetime.now(IST)).total_seconds()
                wait_hours = wait_seconds / 3600

                logger.info(
                    f"🌙 Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M IST')} "
                    f"({wait_hours:.1f}h)"
                )

                # Sleep in chunks to allow graceful shutdown and force scans
                sleep_chunk = min(wait_seconds, 300)  # Max 5 min chunks
                
                for _ in range(int(sleep_chunk / 5)):
                    if redis_client and redis_client.get("tradesage:force_scan") == "1":
                        break
                    time.sleep(5)

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
