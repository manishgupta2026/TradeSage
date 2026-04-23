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
    version="2.0.0",
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
#  HELPER: Find latest model report
# ══════════════════════════════════════════════════════════════

def _find_latest_report() -> dict:
    """Find and load the most recent model report JSON."""
    models_dir = PROJECT_ROOT / "models"

    # Try specific known report files first (in priority order)
    # current_report.json is always updated by the retrainer after daily training
    # tradesage_10y_report.json is the primary report (also updated by retrainer)
    priority_reports = [
        models_dir / "current_report.json",
        models_dir / "tradesage_10y_report.json",
        models_dir / "tradesage_v2_report.json",
        models_dir / "tradesage_angel_report.json",
        models_dir / "tradesage_model_report.json",
    ]

    for rp in priority_reports:
        if rp.exists():
            try:
                with open(rp) as f:
                    report = json.load(f)
                report["_source_file"] = rp.name
                return report
            except Exception:
                continue

    # Fallback: pick the most recently modified *_report.json
    try:
        all_reports = sorted(
            models_dir.glob("*_report.json"),
            key=os.path.getmtime,
            reverse=True,
        )
        if all_reports:
            with open(all_reports[0]) as f:
                report = json.load(f)
            report["_source_file"] = all_reports[0].name
            return report
    except Exception:
        pass

    return {}


def _fin_scalar(field: str, raw) -> Optional[float]:
    """Parse one financial value; invalid/missing → None. Logs rejections (no silent bad data)."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        logger.warning("portfolio: %s rejected (boolean)", field)
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        if v != v or v in (float("inf"), float("-inf")):
            logger.warning("portfolio: %s rejected (non-finite)", field)
            return None
        return v
    if isinstance(raw, str):
        try:
            v = float(raw.strip())
            if v != v or v in (float("inf"), float("-inf")):
                logger.warning("portfolio: %s rejected (non-finite string)", field)
                return None
            return v
        except ValueError:
            logger.warning("portfolio: %s rejected (not numeric): %r", field, raw)
            return None
    logger.warning("portfolio: %s rejected (bad type %s)", field, type(raw).__name__)
    return None


def _ledger_field(row: dict, key: str) -> Optional[float]:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    return _fin_scalar(key, raw)


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

    # Read model report for AUC + version
    report = _find_latest_report()
    model_auc = report.get("test_metrics", {}).get("auc_score")
    model_version = report.get("version", report.get("_source_file", "").replace("_report.json", ""))

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
        "model_version": model_version or "tradesage_10y",
        "scanner_healthy": scanner_healthy,
        "server_time": now.isoformat(),
    }


# ══════════════════════════════════════════════════════════════
#  POST /api/scan — Trigger manual scan
# ══════════════════════════════════════════════════════════════

@app.post("/api/scan")
async def trigger_scan():
    """Trigger a manual scan immediately."""
    if redis_pool:
        try:
            await redis_pool.set("tradesage:force_scan", "1")
            return {"status": "success", "message": "Scan triggered"}
        except Exception as e:
            return {"status": "error", "message": f"Redis error: {e}"}
    return {"status": "error", "message": "Redis not available"}



# ══════════════════════════════════════════════════════════════
#  GET /api/model/metrics — full model report for dashboard
# ══════════════════════════════════════════════════════════════

@app.get("/api/model/metrics")
async def get_model_metrics():
    """Return comprehensive model metrics from the latest training report."""
    report = _find_latest_report()

    if not report:
        return {
            "available": False,
            "message": "No training report found",
        }

    test_metrics = report.get("test_metrics", {})
    val_metrics = report.get("val_metrics", {})
    params = report.get("parameters", {})

    return {
        "available": True,
        "training_date": report.get("training_date"),
        "version": report.get("version"),
        "data_source": report.get("data_source"),
        "stocks_trained": report.get("stocks_trained"),
        "total_samples": report.get("total_samples"),
        "features": report.get("features"),
        "ensemble_mode": report.get("ensemble_mode"),
        "market_context": report.get("market_context"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "parameters": {
            "forward_days": params.get("forward_days"),
            "gain_threshold": params.get("gain_threshold"),
            "max_drawdown": params.get("max_drawdown"),
        },
        "test_metrics": {
            "auc": test_metrics.get("auc_score"),
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "f1": test_metrics.get("f1"),
            "win_rate": test_metrics.get("predicted_win_rate"),
        },
        "val_metrics": {
            "auc": val_metrics.get("auc_score"),
            "precision": val_metrics.get("precision"),
            "recall": val_metrics.get("recall"),
            "f1": val_metrics.get("f1"),
            "win_rate": val_metrics.get("predicted_win_rate"),
        },
        "top_features": report.get("top_features", [])[:10],
    }


# ══════════════════════════════════════════════════════════════
#  GET /api/portfolio — active positions from ledger
# ══════════════════════════════════════════════════════════════

@app.get("/api/portfolio")
async def get_portfolio():
    """Read active positions from backtest_ledger.csv or positions.json."""
    active = []
    closed = []
    ledger_pnl = 0.0

    # Try positions.json first (live paper trading state from persistent data volume)
    positions_path = PROJECT_ROOT / "data" / "positions.json"
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
                    c = dict(pos)
                    c["symbol"] = sym
                    closed.append(c)
        except Exception as e:
            logger.warning("portfolio: failed to read positions.json: %s", e)

    # Authoritative total_pnl (Option A): ONLY validated backtest_ledger.csv — never positions.json (no mixing).
    ledger_path = PROJECT_ROOT / "data" / "backtest_ledger.csv"
    wins, total_trades = 0, 0
    ledger_rows_excluded_invalid = 0
    if ledger_path.exists():
        try:
            with open(ledger_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rupees = _ledger_field(row, "pnl_rupees")
                    pnl_pct = _ledger_field(row, "pnl_pct")
                    if rupees is None or pnl_pct is None:
                        ledger_rows_excluded_invalid += 1
                        logger.warning(
                            "portfolio: skipping ledger row (invalid numbers) symbol=%s",
                            row.get("symbol"),
                        )
                        continue
                    total_trades += 1
                    ledger_pnl += rupees
                    if pnl_pct > 0:
                        wins += 1
        except Exception as e:
            logger.warning("portfolio: ledger read failed: %s", e)

    total_pnl = round(ledger_pnl, 2)
    total_pnl_source = "backtest_ledger"

    # Enrich closed rows (positions.json): informational only — not used for total_pnl.
    live_closed_pnl = 0.0
    for c in closed:
        ep = _fin_scalar("entry_price", c.get("entry_price"))
        xp = _fin_scalar("exit_price", c.get("exit_price"))
        sh = _fin_scalar("shares", c.get("shares"))
        if ep is None or xp is None or sh is None:
            c["pnl_error"] = "invalid or missing entry_price, exit_price, or shares"
            logger.warning(
                "portfolio: closed row %s excluded from live_closed_total_pnl (missing/invalid fields)",
                c.get("symbol"),
            )
            continue
        if ep < 0 or xp < 0 or sh <= 0:
            c["pnl_error"] = "entry_price, exit_price, and shares must be positive for P&L"
            logger.warning(
                "portfolio: closed row %s excluded from live_closed_total_pnl (non-positive values)",
                c.get("symbol"),
            )
            continue
        realized = (xp - ep) * sh
        c["realized_pnl_rupees"] = round(realized, 2)
        c["pnl_pct_realized"] = round(((xp - ep) / ep) * 100.0, 4) if ep > 0 else None
        live_closed_pnl += realized

    invalid_trades_count = sum(1 for c in closed if c.get("pnl_error"))
    excluded_trades_count = ledger_rows_excluded_invalid + invalid_trades_count

    # Use model report win rate as primary (more representative of current model)
    # Backtest ledger win rate is secondary (historical trades, may be stale)
    ledger_win_rate = wins / total_trades if total_trades > 0 else 0
    report = _find_latest_report()
    model_win_rate = report.get("test_metrics", {}).get("predicted_win_rate")

    return {
        "active_count": len(active),
        "active_positions": active,
        "closed_count": len(closed),
        "closed_positions": closed,
        "ledger_total_pnl": round(ledger_pnl, 2),
        "live_closed_total_pnl": round(live_closed_pnl, 2),
        "total_pnl": total_pnl,
        "total_pnl_source": total_pnl_source,
        "total_pnl_definition": (
            "total_pnl is the sum of pnl_rupees for validated rows only in "
            "data/backtest_ledger.csv. Live closed trades (positions.json) appear in "
            "live_closed_total_pnl and closed_positions, not in total_pnl."
        ),
        "ledger_rows_excluded_invalid": ledger_rows_excluded_invalid,
        "invalid_trades_count": invalid_trades_count,
        "excluded_trades_count": excluded_trades_count,
        "win_rate": round(model_win_rate, 4) if model_win_rate else round(ledger_win_rate, 4),
        "ledger_win_rate": round(ledger_win_rate, 4) if total_trades > 0 else None,
        "model_win_rate": round(model_win_rate, 4) if model_win_rate else None,
        "total_trades": total_trades,
    }


# ══════════════════════════════════════════════════════════════
#  POST /api/config/angelone — save & restart scanner
# ══════════════════════════════════════════════════════════════

class AngelOneConfig(BaseModel):
    client_id: str
    password: str
    totp_secret: str
    api_key: Optional[str] = None


@app.post("/api/config/angelone")
async def save_angel_config(config: AngelOneConfig):
    """Validate and save Angel One config, restart scanner subprocess."""
    if not config.client_id or not config.password or not config.totp_secret:
        raise HTTPException(status_code=400, detail="All fields are required")

    # Save to config file
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "angel_one_config.json"
    main_cfg = config_dir / "angel_config.json"

    existing: dict = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except Exception as e:
            logger.warning("angel config: could not read existing %s: %s", config_path, e)

    new_key = (config.api_key or "").strip() or None
    api_key = new_key
    if not api_key and isinstance(existing.get("api_key"), str) and existing["api_key"].strip():
        api_key = existing["api_key"].strip()
    if not api_key and main_cfg.exists():
        try:
            with open(main_cfg) as f:
                mk = json.load(f).get("api_key")
            if isinstance(mk, str) and mk.strip():
                api_key = mk.strip()
        except Exception as e:
            logger.warning("angel config: could not read angel_config.json: %s", e)
    if not api_key:
        api_key = os.environ.get("ANGEL_API_KEY")
        if api_key:
            api_key = api_key.strip() or None

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "api_key is required and was not found: send api_key in the request body, "
                "or set it in config/angel_config.json, an existing angel_one_config.json, "
                "or environment variable ANGEL_API_KEY"
            ),
        )

    config_data = {
        "client_id": config.client_id,
        "password": config.password,
        "totp_token": config.totp_secret,
        "updated_at": datetime.now().isoformat(),
        "api_key": api_key,
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

    report = _find_latest_report()
    if report:
        result["last_retrain"] = report.get("training_date")
        result["current_auc"] = report.get("test_metrics", {}).get("auc_score")
        result["samples_trained_on"] = report.get("total_samples")

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
