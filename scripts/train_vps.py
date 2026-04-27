#!/usr/bin/env python3
"""
TradeSage VPS Retrain Wrapper

Orchestrates:
  1. Fetch 20yr data via yfinance (skip if fresh)
  2. Train model on yfinance data
  3. AUC gate: deploy or reject
  4. Telegram notification
  5. Full logging to logs/retrain_YYYY-MM-DD.log

Usage:
    python scripts/train_vps.py            # Full pipeline
    python scripts/train_vps.py --skip-fetch  # Skip data fetch, train only
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ──
MODEL_PATH     = PROJECT_ROOT / 'models' / 'tradesage_10y.pkl'
REPORT_PATH    = PROJECT_ROOT / 'models' / 'tradesage_10y_report.json'
CURRENT_MODEL  = PROJECT_ROOT / 'models' / 'current.pkl'
LOG_DIR        = PROJECT_ROOT / 'logs'

# ── Logging ──
LOG_DIR.mkdir(exist_ok=True, parents=True)
log_file = LOG_DIR / f"retrain_{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('train_vps')


def send_telegram(message):
    """Send Telegram notification using .env or env vars."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / '.env')
    except ImportError:
        pass

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        # Try angel_config.json as fallback
        config_path = PROJECT_ROOT / 'config' / 'angel_config.json'
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                token = token or cfg.get('telegram_token')
                chat_id = chat_id or cfg.get('telegram_chat_id')
            except Exception:
                pass

    if not token or not chat_id:
        logger.warning(f"[TELEGRAM STUB] {message}")
        return

    import requests
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


def run_step(cmd, description):
    """Run a subprocess step with logging."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  {description}")
    logger.info(f"  CMD: {' '.join(cmd)}")
    logger.info(f"{'='*70}\n")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,  # Stream to stdout/log
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Step failed with exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='TradeSage VPS Retrain Pipeline')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip yfinance data fetch step')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='Gain threshold (default: 0.02)')
    parser.add_argument('--forward-days', type=int, default=5,
                        help='Forward days (default: 5)')
    parser.add_argument('--max-drawdown', type=float, default=-0.99,
                        help='Max drawdown filter (default: -0.99 = disabled)')
    parser.add_argument('--max-rows-per-stock', type=int, default=2500,
                        help='Max rows per stock (default: 2500 for 10yr data)')
    parser.add_argument('--max-stocks', type=int, default=500,
                        help='Max stocks to train on (default: 500, prevents OOM on 4GB VPS)')
    args = parser.parse_args()

    start_time = datetime.now()
    python = sys.executable  # Use same Python interpreter

    logger.info("=" * 70)
    logger.info("  TRADESAGE VPS RETRAIN PIPELINE")
    logger.info(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Log: {log_file}")
    logger.info("=" * 70)

    # ── Step 1: Fetch data ──
    if not args.skip_fetch:
        ok = run_step(
            [python, str(PROJECT_ROOT / 'scripts' / 'fetch_yfinance_10y.py'), '--years', '10'],
            "STEP 1: Fetch/Update 10yr yfinance data (Rolling Window)"
        )
        if not ok:
            send_telegram("🚨 *TradeSage Retrain FAILED* — yfinance fetch error")
            sys.exit(1)
    else:
        logger.info("\n⏭️  Skipping data fetch (--skip-fetch)")

    # ── Step 2: Train model ──
    train_cmd = [
        python, str(PROJECT_ROOT / 'scripts' / 'train.py'),
        '--source', 'yfinance',
        '--model-path', str(MODEL_PATH),
        '--threshold', str(args.threshold),
        '--forward-days', str(args.forward_days),
        '--max-drawdown', str(args.max_drawdown),
        '--max-rows-per-stock', str(args.max_rows_per_stock),
        '--max-stocks', str(args.max_stocks),
        '--ensemble',
    ]

    ok = run_step(train_cmd, "STEP 2: Train model on 10yr yfinance data")
    if not ok:
        send_telegram("🚨 *TradeSage Retrain FAILED* — Training error")
        sys.exit(1)

    # ── Step 3: Read AUC from report ──
    if not REPORT_PATH.exists():
        logger.error(f"Report not found: {REPORT_PATH}")
        send_telegram("🚨 *TradeSage Retrain FAILED* — No report generated")
        sys.exit(1)

    with open(REPORT_PATH) as f:
        report = json.load(f)

    test_auc = report.get('test_metrics', {}).get('auc_score', 0)
    val_auc = report.get('val_metrics', {}).get('auc_score', 0)
    precision = report.get('test_metrics', {}).get('precision', 0)
    win_rate = report.get('test_metrics', {}).get('predicted_win_rate', 0)
    stocks = report.get('stocks_trained', 0)
    elapsed = report.get('elapsed_seconds', 0)
    n_features = report.get('features', 0)
    top_feats = report.get('top_features', [])
    fund_feat_count = sum(1 for f in top_feats if f.get('feature', '').startswith('fund_'))

    logger.info(f"\n{'='*70}")
    logger.info(f"  AUC GATE EVALUATION")
    logger.info(f"{'='*70}")
    logger.info(f"  Test  AUC:  {test_auc:.4f}")
    logger.info(f"  Val   AUC:  {val_auc:.4f}")
    logger.info(f"  Precision:  {precision:.4f}")
    logger.info(f"  Win Rate:   {win_rate*100:.1f}%")
    logger.info(f"  Stocks:     {stocks}")
    logger.info(f"  Features:   {n_features}")

    # Format top features for Telegram
    top_feats_str = ""
    if top_feats:
        for i, f in enumerate(top_feats[:5], 1):
            name = f.get('feature', '?')
            imp = f.get('importance', 0)
            icon = '🔬' if name.startswith('fund_') else '📊'
            top_feats_str += f"   {icon} {i}. {name} ({imp:.4f})\n"

    # ── Step 4: AUC Gate ──
    if test_auc >= 0.70:
        # DEPLOY
        logger.info(f"\n  ✅ AUC {test_auc:.4f} >= 0.70 — DEPLOYING")

        # Copy model to current.pkl (Windows doesn't support symlinks easily)
        shutil.copy2(str(MODEL_PATH), str(CURRENT_MODEL))
        logger.info(f"  Copied {MODEL_PATH.name} → {CURRENT_MODEL.name}")

        # Copy report to current_report.json so the API/frontend always loads fresh data
        CURRENT_REPORT = PROJECT_ROOT / 'models' / 'current_report.json'
        shutil.copy2(str(REPORT_PATH), str(CURRENT_REPORT))
        logger.info(f"  Copied {REPORT_PATH.name} → {CURRENT_REPORT.name}")

        msg = (
            f"✅ *TradeSage Retrain Complete*\n"
            f"🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Test AUC: *{test_auc:.4f}*\n"
            f"🎯 Precision: {precision:.4f}\n"
            f"💰 Win Rate: {win_rate*100:.1f}%\n"
            f"📈 Stocks: {stocks} | Features: {n_features}\n"
            f"⏱️ Time: {elapsed/60:.0f} min\n\n"
        )
        if top_feats_str:
            msg += f"🏆 *Top 5 Features:*\n{top_feats_str}\n"
        msg += f"✅ *Deployed: YES* → `current.pkl`"
        send_telegram(msg)

    elif test_auc >= 0.65:
        # BELOW GATE — don't deploy
        logger.info(f"\n  ⚠️ AUC {test_auc:.4f} >= 0.65 but < 0.70 — NOT DEPLOYING")

        msg = (
            f"⚠️ *TradeSage Retrain*\n"
            f"🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Test AUC: *{test_auc:.4f}*\n"
            f"🎯 Precision: {precision:.4f}\n"
            f"📈 Stocks: {stocks} | Features: {n_features}\n\n"
        )
        if top_feats_str:
            msg += f"🏆 *Top 5 Features:*\n{top_feats_str}\n"
        msg += (
            f"⚠️ *Below 0.70 gate — Not deployed*\n"
            f"Old model remains active."
        )
        send_telegram(msg)

    else:
        # INVESTIGATE
        logger.info(f"\n  🚨 AUC {test_auc:.4f} < 0.65 — INVESTIGATE LABELS")

        msg = (
            f"🚨 *TradeSage Retrain — LOW AUC*\n"
            f"🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Test AUC: *{test_auc:.4f}*\n"
            f"📈 Stocks: {stocks} | Features: {n_features}\n\n"
            f"🚨 *Investigate label quality*\n"
            f"Possible issues: threshold, forward\\_days, data quality"
        )
        send_telegram(msg)

    elapsed_total = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"  Total time: {elapsed_total/60:.1f} minutes")
    logger.info(f"  Log saved: {log_file}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
