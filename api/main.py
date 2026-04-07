"""
TradeSage API — FastAPI Backend
Serves the dashboard, signals, status, portfolio, config, and training endpoints.
SSE streaming via Redis pub/sub for live signal push.
"""

import asyncio
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as aioredis
from sse_starlette.sse import EventSourceResponse

# ── Project paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tradesage.api")

# ── Redis ──
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── App ──
app = FastAPI(
    title="TradeSage API",
    version="1.0.0",
    description="AI-powered NSE swing trading dashboard API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──
redis_pool: Optional[aioredis.Redis] = None
scanner_process: Optional[subprocess.Popen] = None


# ══════════════════════════════════════════════════════════════
#  STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    global redis_pool
    try:
        redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_pool.ping()
        logger.info(f"✅ Connected to Redis: {REDIS_URL}")
    except Exception as e:
        logger.warning(f"⚠️  Redis not available ({e}). Running in local-only mode.")
        redis_pool = None


@app.on_event("shutdown")
async def shutdown():
    global redis_pool
    if redis_pool:
        await redis_pool.close()


# ══════════════════════════════════════════════════════════════
#  SERVE FRONTEND
# ══════════════════════════════════════════════════════════════

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the Stitch-generated index.html dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return FileResponse(index_path, media_type="text/html")


# Serve any other static assets in frontend/
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ══════════════════════════════════════════════════════════════
#  GET /api/signals — latest signals from Redis
# ══════════════════════════════════════════════════════════════

@app.get("/api/signals")
async def get_signals(min_prob: float = 0.65):
    """Return latest 100 signals, optionally filtered by minimum probability."""
    signals = []

    # Try Redis first
    if redis_pool:
        try:
            raw_list = await redis_pool.lrange("tradesage:signals_history", 0, 99)
            for raw in raw_list:
                try:
                    sig = json.loads(raw)
                    if sig.get("probability", 0) >= min_prob:
                        signals.append(sig)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Redis read error: {e}")

    # Fallback: check local signals file
    if not signals:
        local_signals_path = PROJECT_ROOT / "data" / "live_signals.json"
        if local_signals_path.exists():
            try:
                with open(local_signals_path) as f:
                    all_sigs = json.load(f)
                signals = [s for s in all_sigs if s.get("probability", 0) >= min_prob][:100]
            except Exception:
                pass

    return signals


# ══════════════════════════════════════════════════════════════
#  GET /api/signals/live — SSE stream via Redis pub/sub
# ══════════════════════════════════════════════════════════════

@app.get("/api/signals/live")
async def signals_live(request: Request):
    """Server-Sent Events stream for live signal updates."""

    async def event_generator():
        # Heartbeat to keep connection alive
        if not redis_pool:
            yield {"event": "message", "data": json.dumps({"info": "Redis not available — polling mode"})}
            while True:
                if await request.is_disconnected():
                    break
                yield {"event": "heartbeat", "data": ":heartbeat"}
                await asyncio.sleep(15)
            return

        pubsub = redis_pool.pubsub()
        await pubsub.subscribe("tradesage:signals")
        logger.info("SSE client connected to tradesage:signals")

        try:
            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    yield {"event": "message", "data": message["data"]}
                else:
                    # Send heartbeat every ~15s to keep connection alive
                    yield {"event": "heartbeat", "data": ":heartbeat"}
                    await asyncio.sleep(5)
        finally:
            await pubsub.unsubscribe("tradesage:signals")
            await pubsub.close()

    return EventSourceResponse(event_generator())


# ══════════════════════════════════════════════════════════════
#  GET /api/status — system health
# ══════════════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    """Return system status: market hours, last scan, model AUC, etc."""
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour, minute = now.hour, now.minute
    market_minutes = hour * 60 + minute

    # NSE market hours: 9:15 – 15:30 IST, Mon-Fri
    market_open = (
        weekday < 5
        and 9 * 60 + 15 <= market_minutes <= 15 * 60 + 30
    )

    # Read model report for AUC
    model_auc = None
    model_version = None
    report_paths = [
        PROJECT_ROOT / "models" / "tradesage_v2_report.json",
        PROJECT_ROOT / "models" / "tradesage_angel_report.json",
        PROJECT_ROOT / "models" / "tradesage_model_report.json",
    ]
    for rp in report_paths:
        if rp.exists():
            try:
                with open(rp) as f:
                    report = json.load(f)
                model_auc = report.get("test_metrics", {}).get("auc_score")
                model_version = report.get("version", rp.stem)
                break
            except Exception:
                continue

    # Last scan timestamp from Redis
    last_scan = None
    scanner_healthy = False
    if redis_pool:
        try:
            last_scan = await redis_pool.get("tradesage:last_scan")
            scanner_healthy = last_scan is not None
        except Exception:
            pass

    return {
        "market_open": market_open,
        "last_scan": last_scan or now.strftime("%H:%M:%S"),
        "model_auc": model_auc,
        "model_version": model_version,
        "scanner_healthy": scanner_healthy,
        "server_time": now.isoformat(),
    }


# ══════════════════════════════════════════════════════════════
#  GET /api/portfolio — active positions from ledger
# ══════════════════════════════════════════════════════════════

@app.get("/api/portfolio")
async def get_portfolio():
    """Read active positions from backtest_ledger.csv or positions.json."""
    active = []
    closed = []
    total_pnl = 0.0

    # Try positions.json first (live paper trading state)
    positions_path = PROJECT_ROOT / "positions.json"
    if positions_path.exists():
        try:
            with open(positions_path) as f:
                positions = json.load(f)
            for sym, pos in positions.items():
                if pos.get("status") == "open":
                    active.append({
                        "symbol": sym,
                        "entry_price": pos.get("entry_price"),
                        "shares": pos.get("shares"),
                        "stop_loss": pos.get("stop_loss"),
                        "take_profit": pos.get("take_profit"),
                        "entry_date": pos.get("entry_date"),
                        "confidence": pos.get("confidence"),
                    })
                else:
                    closed.append(pos)
        except Exception:
            pass

    # Also read backtest_ledger.csv for historical win rate
    ledger_path = PROJECT_ROOT / "data" / "backtest_ledger.csv"
    wins, total_trades = 0, 0
    if ledger_path.exists():
        try:
            with open(ledger_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_trades += 1
                    pnl = float(row.get("pnl_pct", 0) or 0)
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
        except Exception:
            pass

    win_rate = wins / total_trades if total_trades > 0 else 0

    return {
        "active_count": len(active),
        "active_positions": active,
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
    }


# ══════════════════════════════════════════════════════════════
#  POST /api/config/angelone — save & restart scanner
# ══════════════════════════════════════════════════════════════

class AngelOneConfig(BaseModel):
    client_id: str
    password: str
    totp_secret: str


@app.post("/api/config/angelone")
async def save_angel_config(config: AngelOneConfig):
    """Validate and save Angel One config, restart scanner subprocess."""
    if not config.client_id or not config.password or not config.totp_secret:
        raise HTTPException(status_code=400, detail="All fields are required")

    # Save to config file
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "angel_one_config.json"

    config_data = {
        "client_id": config.client_id,
        "password": config.password,
        "totp_token": config.totp_secret,
        "updated_at": datetime.now().isoformat(),
    }

    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Angel One config saved to {config_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    # Send Telegram notification
    _send_telegram(f"🔑 Angel One config updated at {datetime.now().strftime('%H:%M:%S')}")

    # Restart scanner subprocess
    global scanner_process
    if scanner_process and scanner_process.poll() is None:
        scanner_process.terminate()
        scanner_process.wait(timeout=10)
        logger.info("Scanner process terminated for restart")

    try:
        scanner_process = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "services" / "scanner.py")],
            cwd=str(PROJECT_ROOT),
        )
        logger.info(f"Scanner restarted (PID: {scanner_process.pid})")
    except Exception as e:
        logger.error(f"Failed to restart scanner: {e}")

    return {"status": "ok", "message": "Config saved. Scanner restarting..."}


# ══════════════════════════════════════════════════════════════
#  GET /api/training/status — last retrain info
# ══════════════════════════════════════════════════════════════

@app.get("/api/training/status")
async def training_status():
    """Return info about the last model retrain."""
    result = {
        "last_retrain": None,
        "current_auc": None,
        "next_retrain_eta": None,
        "samples_trained_on": None,
    }

    # Scan for the most recent report
    models_dir = PROJECT_ROOT / "models"
    reports = sorted(models_dir.glob("*_report.json"), key=os.path.getmtime, reverse=True)

    if reports:
        try:
            with open(reports[0]) as f:
                report = json.load(f)
            result["last_retrain"] = report.get("training_date")
            result["current_auc"] = report.get("test_metrics", {}).get("auc_score")
            result["samples_trained_on"] = report.get("total_samples")
        except Exception:
            pass

    # Estimate next retrain: 16:30 IST today or tomorrow
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    retrain_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
    if now >= retrain_time:
        retrain_time += timedelta(days=1)
    # Skip weekends
    while retrain_time.weekday() >= 5:
        retrain_time += timedelta(days=1)
    result["next_retrain_eta"] = retrain_time.isoformat()

    return result


# ══════════════════════════════════════════════════════════════
#  POST /api/training/trigger — manual retrain
# ══════════════════════════════════════════════════════════════

@app.post("/api/training/trigger")
async def trigger_retrain():
    """Manually kick off a retraining job."""
    retrainer_path = PROJECT_ROOT / "services" / "retrainer.py"
    if not retrainer_path.exists():
        raise HTTPException(status_code=404, detail="Retrainer script not found")

    try:
        proc = subprocess.Popen(
            [sys.executable, str(retrainer_path), "--manual"],
            cwd=str(PROJECT_ROOT),
        )
        logger.info(f"Retrain triggered manually (PID: {proc.pid})")
        _send_telegram("🔄 Manual retrain triggered from dashboard")
        return {"status": "ok", "pid": proc.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _send_telegram(message: str):
    """Best-effort Telegram notification."""
    import requests as req

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
        req.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ══════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
