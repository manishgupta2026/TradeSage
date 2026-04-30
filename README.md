# TradeSage AI: Autonomous 24/7 Live Trading Engine 📈🤖

TradeSage is a professional-grade, autonomous machine learning trading system optimized for the Indian Stock Market (NSE). It combines 10 years of historical intelligence with real-time Angel One API execution, AI-driven news sentiment, and TradingView conviction filtering.

---

## 🔥 V3 Production Features
The system is now fully stabilized for live paper-trading on a production VPS:

- **Live Angel One P&L Tracking**: Integrates directly with the Angel One ticker for millisecond-accurate Mark-to-Market (MTM) P&L updates.
- **AI Conviction Engine**:
  - **FinBERT News Sentiment**: Real-time news scraping and AI-driven sentiment analysis (±0.1 strict bullish/bearish classification).
  - **TradingView Consensus**: Aggregated "Strong Buy" signals from 26 technical indicators.
- **Penny Stock Protection**: Strict algorithmic filters ignore any asset under **₹50** or with low liquidity (**<1L volume**).
- **Accurate Entry Sync**: Fetches real-time LTP at the exact second of entry to eliminate stale "yesterday-close" pricing errors.
- **24/7 Autonomous Operation**: Containerized Docker stack running a background scanner, real-time FastAPI backend, and daily auto-training.

---

## 🚀 Performance & Architecture
- **Verified Strategy**: +28.77% Net Profit (Rs. 2.87L on 10L simulated capital).
- **Big Data**: Trained on **3.2 million+ rows** of 10-year OHLCV data.
- **Model**: Optimized XGBoost with daily rolling-window retraining.
- **Infrastructure**: Hosted on DigitalOcean VPS with DuckDNS (SSL) and Nginx reverse proxy.

---

## 📂 System Components
- **`services/scanner.py`**: The "Heart" — scans 3000+ stocks every 15 mins.
- **`api/main.py`**: The "Bridge" — streams real-time signals via SSE and calculates live P&L.
- **`src/core/fundamental_analyzer.py`**: The "Brain" — computes AI sentiment and conviction scores.
- **`frontend/`**: The "Face" — High-fidelity, dark-themed dashboard for real-time portfolio monitoring.

---

## 🛠️ Operational Commands (VPS)

### Start Everything
```bash
docker-compose up -d --build
```

### Manual Market Scan
```bash
# Triggers an immediate technical and fundamental scan
python scripts/manual_scan.py
```

### View Live Logs
```bash
docker logs -f tradesage-scanner
```

---

## ⚠️ Disclaimer
Educational purposes only. This system uses paper trading simulations. Do not trade real capital without extensive testing and clinical risk management. TradeSage is not a financial advisor.
