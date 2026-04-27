import schedule
import time
import subprocess
import logging
from datetime import datetime
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('auto_train')

def run_training():
    logger.info("="*60)
    logger.info("Starting Daily TradeSage Training Pipeline")
    logger.info("="*60)
    try:
        # Run the training script, skipping download since we'll fetch via yfinance
        result = subprocess.run(
            [sys.executable, "train_local_push_vps.py", "--skip-download"],
            cwd=".",
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✅ Training and VPS deployment completed successfully!")
        else:
            logger.error("❌ Training failed!")
            logger.error(result.stderr)
    except Exception as e:
        logger.error(f"❌ Error running training script: {e}")

# Schedule the job every day at 18:00 (6:00 PM) IST
schedule.every().day.at("18:00").do(run_training)

if __name__ == "__main__":
    logger.info("TradeSage Auto-Training Scheduler Started.")
    logger.info("The training pipeline will run daily at 18:00 (6:00 PM).")
    logger.info("Press Ctrl+C to exit.")
    
    # Optional: Run immediately on startup for testing? 
    # run_training()

    while True:
        schedule.run_pending()
        time.sleep(60)
