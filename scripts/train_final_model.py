"""
TradeSage - Full Training Pipeline (scripts/train_final_model.py)

Orchestrates the complete ML pipeline:
  1. Fetch 10y NSE data via yfinance
  2. Feature engineering (80+ indicators)
  3. Walk-forward validation
  4. Optional Optuna hyperparameter tuning
  5. Final model training
  6. Backtest with realistic NSE costs
  7. Save model to models/

Usage:
    python scripts/train_final_model.py
    python scripts/train_final_model.py --tune         # enable Optuna
    python scripts/train_final_model.py --universe nifty200
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tradesage.data.fetch_nse_data import fetch_nse_data, load_symbol_universe
from tradesage.features.technical_indicators import FeatureEngineer
from tradesage.models.train_xgboost import TradingModelTrainer
from tradesage.backtest.simulate_trades import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_UNIVERSE = "data/nifty200.json"
DEFAULT_YEARS = 10
DEFAULT_FORWARD_DAYS = 10
DEFAULT_GAIN_THRESHOLD = 0.04     # 4% swing gain
DEFAULT_MODEL_PATH = "models/tradesage_model.pkl"
DEFAULT_MIN_CONFIDENCE = 0.60


def load_symbols(universe_name: str) -> list:
    """Load symbol list; fall back to nifty500 if requested file not found."""
    candidates = [
        f"data/{universe_name}.json",
        f"data/nifty{universe_name}.json",
        universe_name,
    ]
    for path in candidates:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            symbols = load_symbol_universe(str(full_path))
            logger.info(f"Loaded {len(symbols)} symbols from {full_path}")
            return symbols
    raise FileNotFoundError(
        f"Symbol universe '{universe_name}' not found. "
        f"Tried: {candidates}"
    )


def main(args: argparse.Namespace) -> None:
    t0 = time.monotonic()

    logger.info("=" * 70)
    logger.info("TRADESAGE — FULL TRAINING PIPELINE")
    logger.info("=" * 70)

    # ── 1. Load symbol universe ───────────────────────────────────────────
    symbols = load_symbols(args.universe)
    logger.info(f"Universe: {len(symbols)} stocks")

    # ── 2. Fetch data ─────────────────────────────────────────────────────
    stock_data, failed = fetch_nse_data(
        symbols,
        years=args.years,
        max_workers=args.workers,
        cache_dir=str(PROJECT_ROOT / "data_cache_yfinance"),
    )
    if len(stock_data) < 10:
        logger.error("Too few stocks fetched — aborting.")
        sys.exit(1)
    logger.info(
        f"Data ready: {len(stock_data)} stocks  |  {len(failed)} failed"
    )

    # ── 3. Feature engineering ────────────────────────────────────────────
    engineer = FeatureEngineer()
    all_frames = []

    logger.info("Engineering features…")
    ok, skipped = 0, 0
    for symbol, df in stock_data.items():
        try:
            df_feat = engineer.add_technical_indicators(df)
            df_feat = engineer.create_target_variable(
                df_feat,
                forward_days=args.forward_days,
                gain_threshold=args.gain_threshold,
            )
            df_feat["symbol"] = symbol
            all_frames.append(df_feat)
            ok += 1
        except Exception as exc:
            logger.debug(f"  Skipped {symbol}: {exc}")
            skipped += 1

    if not all_frames:
        logger.error("No usable data after feature engineering — aborting.")
        sys.exit(1)
    logger.info(f"Feature engineering: {ok} ok  |  {skipped} skipped")

    # ── 4. Per-stock 80/20 chronological split ────────────────────────────
    #  Splitting per-stock (not globally) prevents market-regime leakage.
    #  (Global split → val = only recent bear market → biased AUC)
    train_parts, val_parts = [], []
    for df in all_frames:
        n = len(df)
        sp = int(n * 0.8)
        train_parts.append(df.iloc[:sp])
        val_parts.append(df.iloc[sp:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    X_train, y_train, feature_cols = engineer.prepare_training_data(train_df)
    X_val, y_val, _ = engineer.prepare_training_data(val_df)

    # Align val columns
    for col in set(feature_cols) - set(X_val.columns):
        X_val[col] = 0
    X_val = X_val[feature_cols]

    logger.info(
        f"Train: {len(X_train):,} rows  |  Val: {len(X_val):,} rows  |  "
        f"Features: {len(feature_cols)}"
    )

    # ── 5. Walk-forward validation ─────────────────────────────────────────
    trainer = TradingModelTrainer()
    all_data = pd.concat(all_frames, ignore_index=True).sort_index()
    wf_results = trainer.evaluate_walk_forward(
        all_data, feature_cols, n_splits=5
    )
    mean_wf_auc = np.mean([r["auc"] for r in wf_results])
    logger.info(f"Walk-forward mean AUC: {mean_wf_auc:.4f}")

    # ── 6. Optional Optuna tuning ─────────────────────────────────────────
    best_params: dict = {}
    if args.tune:
        logger.info(f"Running Optuna ({args.n_trials} trials)…")
        best_params = trainer.tune_hyperparams(
            X_train, y_train, n_trials=args.n_trials
        )

    # ── 7. Final model training ────────────────────────────────────────────
    trainer.train_model(X_train, y_train, X_val, y_val, params=best_params or None)

    # ── 8. Evaluate on val set ────────────────────────────────────────────
    metrics = trainer.evaluate_model(X_val, y_val)
    trainer.get_feature_importance(top_n=20)

    # ── 9. Backtest on first stock that has enough data ────────────────────
    logger.info("\nRunning backtest on sample stock…")
    sample_sym = next(iter(stock_data))
    sample_df_raw = stock_data[sample_sym]
    try:
        sample_feat = engineer.add_technical_indicators(sample_df_raw)
        # Add ATR column for backtester
        if "atr" not in sample_feat.columns:
            from ta.volatility import average_true_range
            sample_feat["atr"] = average_true_range(
                sample_feat["high"], sample_feat["low"], sample_feat["close"]
            )

        X_bt, _, _ = engineer.prepare_training_data(
            engineer.create_target_variable(
                sample_feat, forward_days=args.forward_days,
                gain_threshold=args.gain_threshold
            )
        )
        for col in set(feature_cols) - set(X_bt.columns):
            X_bt[col] = 0
        X_bt = X_bt[feature_cols]

        preds, probs = trainer.predict(X_bt)
        bt = Backtester(initial_capital=100_000)
        bt.run_backtest(
            sample_feat.loc[X_bt.index],
            preds,
            probs,
            min_confidence=DEFAULT_MIN_CONFIDENCE,
        )
    except Exception as exc:
        logger.warning(f"Backtest skipped: {exc}")

    # ── 10. Save model ─────────────────────────────────────────────────────
    model_path = PROJECT_ROOT / args.model_path
    trainer.save_model(str(model_path))

    elapsed = time.monotonic() - t0
    logger.info(f"\n✓ Pipeline complete in {elapsed:.0f}s")
    logger.info(f"  AUC:         {metrics.get('auc_score', 'N/A')}")
    logger.info(f"  Win Rate:    {metrics.get('predicted_win_rate', 0)*100:.1f}%")
    logger.info(f"  Model path:  {model_path}")

    auc = metrics.get("auc_score", 0)
    if auc < 0.70:
        logger.warning(
            f"AUC {auc:.4f} below 0.70 threshold. Consider:\n"
            f"  • Adding more training stocks\n"
            f"  • Running Optuna tuning (--tune)\n"
            f"  • Adjusting gain_threshold"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TradeSage full training pipeline"
    )
    parser.add_argument(
        "--universe",
        default="nifty200",
        help="Symbol universe file (without .json). Default: nifty200",
    )
    parser.add_argument(
        "--years", type=int, default=DEFAULT_YEARS,
        help="Years of historical data (default: 10)"
    )
    parser.add_argument(
        "--forward-days", type=int, default=DEFAULT_FORWARD_DAYS,
        dest="forward_days",
        help="Target look-ahead window in trading days (default: 10)"
    )
    parser.add_argument(
        "--gain-threshold", type=float, default=DEFAULT_GAIN_THRESHOLD,
        dest="gain_threshold",
        help="Minimum gain for positive label, e.g. 0.04 = 4%% (default: 0.04)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning before final training"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, dest="n_trials",
        help="Optuna trial count (default: 100)"
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH, dest="model_path",
        help=f"Output model path (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Parallel download threads (default: 10)"
    )

    main(parser.parse_args())
