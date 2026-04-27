"""
TradeSage Local Training Pipeline — Full Technical + Fundamental
1. Download yfinance cache from VPS
2. Download fundamentals cache from VPS
3. Batch-scrape any missing fundamentals via yfinance fallback
4. Train model with BOTH technical (99) + fundamental (8) features
5. Push trained model + report back to VPS
6. Restart scanner to hot-swap
"""
import paramiko
import sys
import os
import time
import json
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HOST = "64.227.139.165"
USER = "root"
PASS = "maniS@12345H"
VPS_DIR = "/root/TradeSage"
LOCAL_ROOT = Path(r"z:\Trade AI")
LOCAL_CACHE = LOCAL_ROOT / "data_cache_yfinance"
FUND_CACHE_DIR = LOCAL_ROOT / "data_cache" / "fundamentals"
FUND_CACHE_FILE = FUND_CACHE_DIR / "fundamentals_cache.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('local_train')


def run_ssh(ssh, cmd, timeout=30):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        stdout.channel.settimeout(timeout)
        return stdout.read().decode('utf-8', 'replace') + stderr.read().decode('utf-8', 'replace')
    except Exception as e:
        return f"ERROR: {e}"


def step1_download_data(ssh, sftp):
    """Download yfinance CSVs from VPS to local machine."""
    print("=" * 60)
    print("  STEP 1: Downloading yfinance data from VPS")
    print("=" * 60)

    LOCAL_CACHE.mkdir(exist_ok=True, parents=True)

    remote_dir = f"{VPS_DIR}/data_cache_yfinance"
    remote_files = [f for f in sftp.listdir(remote_dir) if f.endswith('.csv')]
    print(f"  Found {len(remote_files)} CSV files on VPS")

    local_files = set(f.name for f in LOCAL_CACHE.glob('*.csv'))
    to_download = [f for f in remote_files if f not in local_files]
    print(f"  Already have {len(local_files)} locally, need {len(to_download)} more")

    downloaded = 0
    for i, fname in enumerate(to_download):
        try:
            sftp.get(f"{remote_dir}/{fname}", str(LOCAL_CACHE / fname))
            downloaded += 1
            if (i + 1) % 200 == 0:
                print(f"    [{i+1}/{len(to_download)}] downloaded...")
        except Exception:
            pass

    total = len(list(LOCAL_CACHE.glob('*.csv')))
    print(f"  ✅ Download complete: {downloaded} new, {total} total local CSVs\n")
    return total


def step2_download_fundamentals(ssh, sftp):
    """Download fundamentals cache from VPS."""
    print("=" * 60)
    print("  STEP 2: Downloading fundamentals cache from VPS")
    print("=" * 60)

    FUND_CACHE_DIR.mkdir(exist_ok=True, parents=True)

    remote_fund = f"{VPS_DIR}/data_cache/fundamentals/fundamentals_cache.json"
    try:
        sftp.get(remote_fund, str(FUND_CACHE_FILE))
        with open(FUND_CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  ✅ Downloaded {len(cache)} stock fundamentals from VPS")
        return cache
    except Exception as e:
        print(f"  ⚠️  No fundamentals cache on VPS: {e}")
        return {}


def step3_fetch_missing_fundamentals(fund_cache, symbols):
    """Use yfinance to fill in fundamentals for stocks missing from cache."""
    print("\n" + "=" * 60)
    print("  STEP 3: Fetching missing fundamentals via yfinance")
    print("=" * 60)

    cached_symbols = set(fund_cache.keys())
    missing = [s for s in symbols if s not in cached_symbols]
    print(f"  Have fundamentals for {len(cached_symbols)} stocks")
    print(f"  Missing: {len(missing)} — will fetch via yfinance")

    if not missing:
        print("  ✅ All fundamentals already cached!\n")
        return fund_cache

    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠️  yfinance not installed, skipping")
        return fund_cache

    fetched = 0
    failed = 0
    for i, sym in enumerate(missing):
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            info = ticker.info

            if info and info.get('regularMarketPrice'):
                fund_data = {
                    'pe_ratio': info.get('trailingPE') or info.get('forwardPE') or 0,
                    'roe': (info.get('returnOnEquity') or 0) * 100 if info.get('returnOnEquity') else 0,
                    'roce': 0,  # Not directly available from yfinance
                    'debt_to_equity': info.get('debtToEquity') or 0,
                    'promoter_holding': 0,  # Not available from yfinance
                    'fii_holding': 0,
                    'dii_holding': 0,
                    'dividend_yield': (info.get('dividendYield') or 0) * 100,
                    'market_cap': info.get('marketCap', 0),
                }
                fund_cache[sym] = {**fund_data, '_ts': time.time()}
                fetched += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(missing)}] fetched={fetched} failed={failed}")
            time.sleep(0.5)

        time.sleep(0.3)  # Rate limit

    # Save updated cache
    FUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(FUND_CACHE_FILE, 'w') as f:
        json.dump(fund_cache, f, indent=2)

    print(f"  ✅ Fundamentals: {fetched} new via yfinance, {failed} failed")
    print(f"  Total in cache: {len(fund_cache)}\n")
    return fund_cache


def step4_train_with_fundamentals(fund_cache):
    """Train model with both technical + fundamental features."""
    print("=" * 60)
    print("  STEP 4: Training model (Technical + Fundamental)")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    # Add project root to path
    sys.path.insert(0, str(LOCAL_ROOT))
    from src.core.feature_engineering import FeatureEngineer
    from src.core.model_training import TradingModelTrainer

    # Load stock data
    print("  Loading yfinance CSVs...")
    csv_files = sorted(LOCAL_CACHE.glob('*_daily.csv'))  # All stocks

    stock_data = {}
    for csv_file in tqdm(csv_files, desc="Loading CSVs"):
        symbol = csv_file.stem.replace('_daily', '')
        try:
            df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.columns = [str(c).lower().strip() for c in df.columns]
            if len(df) >= 200 and all(c in df.columns for c in ['open', 'high', 'low', 'close', 'volume']):
                stock_data[symbol] = df
        except Exception:
            pass

    print(f"  Loaded {len(stock_data)} stocks")

    # Set up feature engineer with fundamentals cache
    engineer = FeatureEngineer()

    # Pre-load the fundamentals cache into the engineer
    clean_cache = {}
    for sym, entry in fund_cache.items():
        clean_cache[sym] = {k: v for k, v in entry.items() if not k.startswith('_')}
    engineer.set_fundamentals_cache(clean_cache)

    # Load market context
    nifty_df = None
    try:
        import yfinance as yf
        print("  Fetching Nifty50 for market context...")
        nifty = yf.download('^NSEI', period='3y', progress=False)
        if len(nifty) > 200:
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            nifty.columns = [str(c).lower() for c in nifty.columns]
            if nifty.index.tz is not None:
                nifty.index = nifty.index.tz_localize(None)
            nifty_df = nifty
            print(f"  ✓ Nifty50: {len(nifty_df)} rows")
    except Exception:
        print("  ⚠ Nifty50 not available")

    # Feature engineering with fundamentals
    print(f"\n  Engineering features for {len(stock_data)} stocks...")
    all_data = []
    fund_hit = 0
    fund_miss = 0

    for symbol, df in tqdm(stock_data.items(), desc="Features+Fundamentals"):
        try:
            if len(df) > 2500:
                df = df.iloc[-2500:]

            # Pass symbol to inject fundamental features!
            df_feat = engineer.add_technical_indicators(df, index_df=nifty_df, symbol=symbol)

            # Check if fundamentals were injected
            if 'fund_pe' in df_feat.columns and df_feat['fund_pe'].iloc[0] != 0:
                fund_hit += 1
            else:
                fund_miss += 1

            df_final = engineer.create_target_variable(
                df_feat,
                forward_days=5,
                gain_threshold=0.02,
                max_drawdown=-0.99,
            )
            df_final['symbol'] = symbol
            all_data.append(df_final)
        except Exception as e:
            if len(all_data) < 3:
                print(f"    Failed {symbol}: {e}")

    print(f"  Processed: {len(all_data)} stocks")
    print(f"  Fundamentals injected: {fund_hit} stocks | Missing: {fund_miss}")

    if not all_data:
        print("❌ No valid data!")
        return False

    # Combine and split
    combined = pd.concat(all_data).sort_index()
    X, y, feature_names = engineer.prepare_training_data(combined)

    # Check for fund_ features
    fund_features = [f for f in feature_names if f.startswith('fund_')]
    print(f"\n  📊 Total features: {len(feature_names)}")
    print(f"  🔬 Fundamental features: {fund_features}")

    dates = X.index.unique().sort_values()
    n_dates = len(dates)
    val_start = dates[int(n_dates * 0.90)]
    test_start = dates[int(n_dates * 0.95)]

    train_mask = X.index < val_start
    val_mask = (X.index >= val_start) & (X.index < test_start)
    test_mask = X.index >= test_start

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Positive rate: {y_train.mean()*100:.1f}%")

    # Train
    print("\n  🤖 Training model (XGBoost + LightGBM + CatBoost ensemble)...")
    trainer = TradingModelTrainer()
    trainer.train_model(X_train, y_train, X_val, y_val, use_ensemble=True)

    # Evaluate
    print("\n  📈 Evaluation...")
    print("\n  --- VALIDATION ---")
    val_metrics = trainer.evaluate_model(X_val, y_val)
    print("\n  --- TEST SET ---")
    test_metrics = trainer.evaluate_model(X_test, y_test)

    # Feature importance
    importance_df = trainer.get_feature_importance(top_n=25)

    # Save
    model_path = str(LOCAL_ROOT / "models" / "tradesage_10y.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)

    # Report
    report_path = model_path.replace('.pkl', '_report.json')

    def safe_metric(m, key):
        v = m.get(key, 0)
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return v

    report = {
        'training_date': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'version': 'unified-v2-fundamental',
        'data_source': 'yfinance',
        'stocks_trained': len(all_data),
        'total_samples': len(X),
        'features': len(feature_names),
        'fundamental_features': fund_features,
        'fundamental_stocks_hit': fund_hit,
        'fundamental_stocks_miss': fund_miss,
        'ensemble_mode': True,
        'market_context': nifty_df is not None,
        'split': {
            'train_until': str(val_start.date()),
            'val_until': str(test_start.date()),
        },
        'val_metrics': {k: safe_metric(val_metrics, k) for k in
                        ['auc_score', 'precision', 'recall', 'f1', 'profit_score', 'predicted_win_rate']},
        'test_metrics': {k: safe_metric(test_metrics, k) for k in
                         ['auc_score', 'precision', 'recall', 'f1', 'profit_score', 'predicted_win_rate']},
        'top_features': importance_df.head(25).to_dict('records'),
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    test_auc = test_metrics.get('auc_score', 0)
    print(f"\n  ✅ Model saved: {model_path}")
    print(f"  📊 Test AUC: {test_auc:.4f}")
    print(f"  📊 Features: {len(feature_names)} ({len(fund_features)} fundamental)")
    return True


def step5_push_to_vps():
    """Push trained model + report + fundamentals cache to VPS."""
    print("\n" + "=" * 60)
    print("  STEP 5: Pushing model to VPS")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=60, banner_timeout=60)
    sftp = ssh.open_sftp()

    files_to_push = {
        str(LOCAL_ROOT / "models" / "tradesage_10y.pkl"): f"{VPS_DIR}/models/tradesage_10y.pkl",
        str(LOCAL_ROOT / "models" / "tradesage_10y_report.json"): f"{VPS_DIR}/models/tradesage_10y_report.json",
        str(FUND_CACHE_FILE): f"{VPS_DIR}/data_cache/fundamentals/fundamentals_cache.json",
    }

    for local_path, remote_path in files_to_push.items():
        if Path(local_path).exists():
            size = Path(local_path).stat().st_size / 1024
            print(f"  Uploading {Path(local_path).name} ({size:.0f} KB)...")
            sftp.put(local_path, remote_path)
        else:
            print(f"  ⚠️  {Path(local_path).name} not found")

    # Copy to current.pkl
    print("  Copying to current.pkl...")
    run_ssh(ssh, f"cp {VPS_DIR}/models/tradesage_10y.pkl {VPS_DIR}/models/current.pkl")
    run_ssh(ssh, f"cp {VPS_DIR}/models/tradesage_10y_report.json {VPS_DIR}/models/current_report.json")

    sftp.close()

    # Restart scanner
    print("  Restarting scanner...")
    print(run_ssh(ssh, "docker restart tradesage-scanner 2>&1"))
    time.sleep(5)
    print("  Scanner logs:")
    print(run_ssh(ssh, "docker logs tradesage-scanner --tail 10 2>&1"))

    # Send Telegram
    report_path = LOCAL_ROOT / "models" / "tradesage_10y_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        auc = report.get('test_metrics', {}).get('auc_score', 0)
        precision = report.get('test_metrics', {}).get('precision', 0)
        win_rate = report.get('test_metrics', {}).get('predicted_win_rate', 0)
        n_features = report.get('features', 0)
        stocks = report.get('stocks_trained', 0)
        fund_feats = report.get('fundamental_features', [])
        fund_hit = report.get('fundamental_stocks_hit', 0)

        # Format top features
        top_feats = report.get('top_features', [])
        top_str = ""
        for i, f in enumerate(top_feats[:7], 1):
            name = f.get('feature', '?')
            imp = f.get('importance', 0)
            icon = '🔬' if name.startswith('fund_') else '📊'
            top_str += f"   {icon} {i}. {name} ({imp:.4f})\\n"

        msg = (
            f"✅ *TradeSage v2 Model Deployed*\\n"
            f"🖥️ Trained locally → pushed to VPS\\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n\\n"
            f"📊 Test AUC: *{auc:.4f}*\\n"
            f"🎯 Precision: {precision:.4f}\\n"
            f"💰 Win Rate: {win_rate*100:.1f}%\\n"
            f"📈 Stocks: {stocks} | Features: {n_features}\\n"
            f"🔬 Fundamental: {len(fund_feats)} features, {fund_hit} stocks\\n\\n"
            f"🏆 *Top Features:*\\n{top_str}\\n"
            f"✅ *Deployed to VPS*"
        )
        escaped_msg = msg.replace("'", "'\\''")
        run_ssh(ssh, f"""docker exec tradesage-scanner python -c "
import json, os, requests
cfg = json.load(open('/app/config/angel_config.json'))
token = cfg.get('telegram_token'); chat_id = cfg.get('telegram_chat_id')
if token and chat_id:
    requests.post(f'https://api.telegram.org/bot{{token}}/sendMessage',
                   json={{'chat_id': chat_id, 'text': '{escaped_msg}', 'parse_mode': 'Markdown'}}, timeout=10)
    print('Telegram sent')
" """)

    ssh.close()
    print("  ✅ Deployment complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TradeSage Full Local Training Pipeline')
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading data from VPS")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Skip yfinance fundamental fetch")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just push")
    parser.add_argument("--skip-push", action="store_true", help="Train only, don't push to VPS")
    args = parser.parse_args()

    start_time = time.time()
    fund_cache = {}

    if not args.skip_download:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(HOST, username=USER, password=PASS, timeout=60, banner_timeout=60)
        sftp = ssh.open_sftp()

        step1_download_data(ssh, sftp)
        fund_cache = step2_download_fundamentals(ssh, sftp)

        sftp.close()
        ssh.close()
    else:
        # Load local fundamentals cache if it exists
        if FUND_CACHE_FILE.exists():
            with open(FUND_CACHE_FILE) as f:
                fund_cache = json.load(f)
            print(f"  Loaded {len(fund_cache)} fundamentals from local cache")

    # Get stock symbols from local CSVs
    symbols = [f.stem.replace('_daily', '') for f in sorted(LOCAL_CACHE.glob('*_daily.csv'))]

    if not args.skip_fundamentals and not args.skip_train:
        fund_cache = step3_fetch_missing_fundamentals(fund_cache, symbols)

    if not args.skip_train:
        ok = step4_train_with_fundamentals(fund_cache)
        if not ok:
            print("❌ Training failed!")
            sys.exit(1)

    if not args.skip_push:
        step5_push_to_vps()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ALL DONE! Total time: {elapsed/60:.1f} minutes")
    print(f"  Technical + Fundamental features → model → VPS")
    print(f"{'=' * 60}")
