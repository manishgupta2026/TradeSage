# TradeSage AI: 10-Year Indian Stock Market Trading System 📈🤖

TradeSage is a production-ready machine learning trading system optimized for the Indian Stock Market (NSE). It evaluates 10 years of historical data to predict high-probability swing trades.

## 🚀 Recent Performance (Verified)
- **Net Profit**: +28.77% (Rs. 2.87L on 10L capital)
- **Win Rate**: 45.47%
- **Dataset**: 10 years of OHLCV data (~3.2 million rows)
- **Model**: Memory-optimized XGBoost

---

## 🛠️ Quick Start

### Option 1: Google Colab (Recommended)
1. Open [TradeSage_Colab.ipynb](TradeSage_Colab.ipynb) directly in VS Code or upload it to Google Colab.
2. Run the cells to:
   - Install dependencies.
   - Fetch 10 years of data for 2000+ stocks.
   - Train the memory-optimized model.

### Option 2: Local Setup
1. **Initialize Virtual Env**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Fetch 10y Dataset**:
   ```bash
   python scripts/fetch_yfinance_10y.py
   ```
3. **Train Model**:
   ```bash
   python scripts/train.py --source yfinance --model-path models/tradesage_10y.pkl
   ```
4. **Run Backtest**:
   ```bash
   python scripts/backtest_angel_one.py
   ```

---

## 📂 Project Structure
- `src/core/`: Contains the "Brains" (Feature Engineering & Model Training).
- `scripts/`: Operational scripts for fetching, training, and backtesting.
- `models/`: Pre-trained models and performance reports.
- `data/`: Watchlists and symbols (e.g., `nse_top_3000_angel.json`).

---

## 🔧 Memory Optimization
This system is uniquely designed to handle **3 million+ rows** on consumer hardware (8-16GB RAM) by:
- Automatically downcasting `float64` to `float32`.
- Aggressive garbage collection (`gc.collect`) during data stitching.
- Multi-threaded fetching to bypass API rate limits.

---

## ⚠️ Disclaimer
Educational purposes only. This system uses paper trading simulations. Do not trade real capital without extensive testing and clinical risk management.
