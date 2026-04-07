#!/usr/bin/env python3
"""
TradeSage Daily Retrainer
Runs daily at 16:30 IST (after market close), Mon–Fri.
1. Fetches today's OHLCV for all stocks via Angel One
2. Appends to existing data_cache_angel/ CSVs
3. Runs training pipeline
4. Compares new AUC vs current — hot-swaps model via symlink if improved
5. Sends Telegram notification on success/degradation

Usage:
    python services/retrainer.py                # Runs on schedule (16:30 IST)
    python services/retrainer.py --manual       # Run immediately (skip schedule wait)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Logging ──
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "retrainer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tradesage.retrainer")

IST = timezone(timedelta(hours=5, minutes=30))


# ══════════════════════════════════════════════════════════════
#  TELEGRAM HELPER
# ══════════════════════════════════════════════════════════════

def send_telegram(message: str):
    """Best-effort Telegram notification."""
    import requests

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

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
#  GET CURRENT MODEL AUC
# ══════════════════════════════════════════════════════════════

def get_current_auc() -> float:
    """Read the current model's AUC from the most recent report."""
    models_dir = PROJECT_ROOT / "models"
    reports = sorted(models_dir.glob("*_report.json"), key=os.path.getmtime, reverse=True)

    for rp in reports:
        try:
            with open(rp) as f:
                report = json.load(f)
            auc = report.get("test_metrics", {}).get("auc_score", 0)
            if auc > 0:
                logger.info(f"Current model AUC: {auc:.4f} (from {rp.name})")
                return auc
        except Exception:
            continue

    logger.warning("No existing model report found — treating as fresh train")
    return 0.0


# ══════════════════════════════════════════════════════════════
#  FETCH TODAY'S DATA
# ══════════════════════════════════════════════════════════════

def fetch_todays_data():
    """Fetch today's OHLCV data and append to cache CSVs."""
    logger.info("📥 Fetching today's market data...")

    try:
        from src.angel.angel_one_api import AngelOneAPI
        from src.angel.angel_data_fetcher import AngelDataFetcher
        import pandas as pd

        config_path = PROJECT_ROOT / "config" / "angel_config.json"
        alt_config = PROJECT_ROOT / "config" / "angel_one_config.json"
        cfg = str(alt_config if alt_config.exists() else config_path)

        api = AngelOneAPI(cfg)
        fetcher = AngelDataFetcher(api)

        # Load watchlist
        watchlist_path = PROJECT_ROOT / "data" / "nse_top_3000_angel.json"
        if not watchlist_path.exists():
            watchlist_path = PROJECT_ROOT / "data" / "nse_top_500_angel.json"

        with open(watchlist_path) as f:
            symbols = json.load(f)

        logger.info(f"Fetching data for {len(symbols)} symbols...")

        cache_dir = PROJECT_ROOT / "data_cache_angel"
        cache_dir.mkdir(exist_ok=True)
        updated = 0
        failed = 0

        # Rate limit: 3 req/sec
        for i, symbol in enumerate(symbols, 1):
            if i % 3 == 0:
                time.sleep(1.1)

            try:
                df = fetcher.fetch_historical_data(symbol, period_days=30)
                if df is not None and len(df) > 0:
                    cache_file = cache_dir / f"{symbol}_daily.csv"

                    if cache_file.exists():
                        existing = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        combined = pd.concat([existing, df])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined.sort_index(inplace=True)
                        combined.to_csv(cache_file)
                    else:
                        df.to_csv(cache_file)

                    updated += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            if i % 100 == 0:
                logger.info(f"  [{i}/{len(symbols)}] updated={updated} failed={failed}")

        logger.info(f"✅ Data fetch complete: {updated} updated, {failed} failed")
        return updated

    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        send_telegram(f"⚠️ Retrainer: data fetch failed — {e}")
        return 0


# ══════════════════════════════════════════════════════════════
#  RUN TRAINING
# ══════════════════════════════════════════════════════════════

def run_training() -> dict:
    """Execute the training pipeline and return the report."""
    timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M")
    model_filename = f"tradesage_retrain_{timestamp}.pkl"
    model_path = str(PROJECT_ROOT / "models" / model_filename)

    logger.info(f"🤖 Starting training → {model_filename}")

    train_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--source", "cache",
        "--model-path", model_path,
        "--forward-days", "5",
        "--threshold", "0.04",
        "--max-drawdown", "-0.03",
        "--ensemble",
    ]

    try:
        result = subprocess.run(
            train_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            logger.error(f"Training failed (exit code {result.returncode})")
            logger.error(f"STDERR: {result.stderr[-500:]}")
            send_telegram(f"🚨 Retrain FAILED (exit code {result.returncode})")
            return None

        logger.info("Training subprocess completed")

        # Read the report
        report_path = model_path.replace(".pkl", "_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            report["model_path"] = model_path
            return report
        else:
            logger.error("No training report generated")
            return None

    except subprocess.TimeoutExpired:
        logger.error("Training timed out (1 hour)")
        send_telegram("🚨 Retrain timed out after 1 hour")
        return None
    except Exception as e:
        logger.error(f"Training error: {e}")
        send_telegram(f"🚨 Retrain error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  MODEL HOT-SWAP VIA SYMLINK
# ══════════════════════════════════════════════════════════════

def hot_swap_model(new_model_path: str):
    """Update the models/current.pkl symlink to point to the new model."""
    current_link = PROJECT_ROOT / "models" / "current.pkl"

    try:
        # Remove existing symlink/file
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()

        # On Windows, copy instead of symlink (symlinks require admin)
        if sys.platform == "win32":
            import shutil
            shutil.copy2(new_model_path, str(current_link))
            logger.info(f"Model copied to {current_link}")
        else:
            # Unix: create symlink
            current_link.symlink_to(Path(new_model_path).resolve())
            logger.info(f"Symlink updated: {current_link} → {new_model_path}")

        # Also copy the report
        report_src = new_model_path.replace(".pkl", "_report.json")
        report_dst = str(PROJECT_ROOT / "models" / "current_report.json")
        if os.path.exists(report_src):
            import shutil
            shutil.copy2(report_src, report_dst)

    except Exception as e:
        logger.error(f"Hot-swap failed: {e}")
        send_telegram(f"⚠️ Model hot-swap failed: {e}")


# ══════════════════════════════════════════════════════════════
#  RETRAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def retrain_pipeline():
    """Full retrain pipeline: fetch → train → compare → swap."""
    logger.info("=" * 70)
    logger.info("  TRADESAGE DAILY RETRAINER")
    logger.info(f"  {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info("=" * 70)

    start_time = time.time()

    # 1. Get current AUC baseline
    current_auc = get_current_auc()

    # 2. Fetch today's data
    updated_count = fetch_todays_data()
    if updated_count == 0:
        logger.warning("No data updated — skipping retrain")
        send_telegram("⚠️ Retrainer: No new data to train on — skipping")
        return

    # 3. Run training
    report = run_training()
    if report is None:
        return

    new_auc = report.get("test_metrics", {}).get("auc_score", 0)
    new_model_path = report.get("model_path", "")
    total_samples = report.get("total_samples", 0)
    elapsed = time.time() - start_time

    logger.info(f"\n{'─' * 60}")
    logger.info(f"  Current AUC: {current_auc:.4f}")
    logger.info(f"  New AUC:     {new_auc:.4f}")
    logger.info(f"  Samples:     {total_samples:,}")
    logger.info(f"  Time:        {elapsed / 60:.1f} minutes")
    logger.info(f"{'─' * 60}")

    # 4. Compare and decide
    DEGRADATION_THRESHOLD = 0.02

    if new_auc >= current_auc - DEGRADATION_THRESHOLD:
        # Accept new model
        logger.info("✅ New model accepted — hot-swapping...")
        hot_swap_model(new_model_path)

        deployed = "YES" if new_auc >= current_auc - DEGRADATION_THRESHOLD else "MARGINAL"
        send_telegram(
            f"✅ *Retrained Successfully*\n"
            f"AUC: {new_auc:.4f} (was {current_auc:.4f})\n"
            f"Samples: {total_samples:,}\n"
            f"Deployed: {deployed}\n"
            f"Time: {elapsed / 60:.1f}m"
        )
    else:
        # Model degraded — keep old
        logger.warning(
            f"⚠️ Model degraded: {current_auc:.4f} → {new_auc:.4f} "
            f"(drop > {DEGRADATION_THRESHOLD}). Keeping previous model."
        )
        send_telegram(
            f"⚠️ *Retrain Degraded*\n"
            f"Old AUC: {current_auc:.4f} → New: {new_auc:.4f}\n"
            f"Keeping previous model.\n"
            f"Investigate data quality or label drift."
        )

        # Clean up the rejected model to save space
        try:
            if os.path.exists(new_model_path):
                os.remove(new_model_path)
                report_path = new_model_path.replace(".pkl", "_report.json")
                if os.path.exists(report_path):
                    os.remove(report_path)
                logger.info("Rejected model files cleaned up")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
#  SCHEDULER
# ══════════════════════════════════════════════════════════════

def wait_for_retrain_window():
    """Wait until 16:30 IST on a weekday."""
    now = datetime.now(IST)
    target = now.replace(hour=16, minute=30, second=0, microsecond=0)

    if now >= target:
        target += timedelta(days=1)

    # Skip weekends
    while target.weekday() >= 5:
        target += timedelta(days=1)

    wait_seconds = (target - datetime.now(IST)).total_seconds()
    if wait_seconds > 0:
        logger.info(
            f"⏰ Next retrain: {target.strftime('%Y-%m-%d %H:%M IST')} "
            f"({wait_seconds / 3600:.1f}h from now)"
        )
        # Sleep in chunks for graceful shutdown
        while wait_seconds > 0:
            chunk = min(wait_seconds, 300)
            time.sleep(chunk)
            wait_seconds -= chunk


def run_scheduled():
    """Run retrainer on schedule (16:30 IST, Mon-Fri)."""
    logger.info("TradeSage Retrainer — scheduled mode")
    send_telegram("📅 TradeSage Retrainer started (scheduled mode)")

    while True:
        try:
            wait_for_retrain_window()
            retrain_pipeline()
        except KeyboardInterrupt:
            logger.info("Retrainer stopped by user")
            break
        except Exception as e:
            logger.error(f"Retrainer error: {e}", exc_info=True)
            send_telegram(f"🚨 Retrainer crash: {e}")
            time.sleep(300)  # Wait 5 min before retry


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeSage Daily Retrainer")
    parser.add_argument("--manual", action="store_true", help="Run immediately (skip schedule)")
    args = parser.parse_args()

    if args.manual:
        logger.info("Manual retrain triggered")
        retrain_pipeline()
    else:
        run_scheduled()
