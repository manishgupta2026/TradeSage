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

## ⚡ V3.1 — Performance & Intelligence Upgrades

### Scanner Performance Overhaul
- **Parallel Scanning**: 6-thread `ThreadPoolExecutor` pipeline — scans stocks simultaneously while honoring the 3 req/sec Angel One rate limit. Reduces full scan time from ~17 min to ~5 min.
- **Tiered Smart Scanning**: Top 500 priority stocks scanned every 15 min; full 3000+ stocks scanned every hour (4th cycle). Eliminates wasted compute on low-priority instruments.
- **Redis OHLC Caching**: Daily candle data cached with 15-min TTL — subsequent scans hit cache instead of API, cutting redundant network calls by ~80%.
- **Batched LTP Injection**: Live prices fetched in a dedicated batch pass after signal generation, ensuring the most accurate entry prices on final signals.

### Autonomous Risk Management
- **ATR-Based Trailing Stop-Loss (3x ATR)**: The scanner dynamically calculates 14-day ATR and ratchets stop-losses upward when positions are in profit. Stop-losses never move down — only up.
- **Sideways Market Detection**: Alerts via Telegram when a stock has been held 10+ days with <5% price range, suggesting capital reallocation.

### Telegram Notifications
- **Manual Trade Alerts**: Instant Telegram push for every Buy/Sell executed from the dashboard.
- **Scanner Alerts**: Real-time Telegram messages for TSL adjustments, sideways detection, and daily portfolio snapshots.
- **Daily Equity Snapshot**: At market close, saves portfolio value + Nifty 50 close to `data/equity_history.json` and sends a detailed summary to Telegram.

### Portfolio Analytics Dashboard
- **Equity Curve**: Interactive line chart tracking portfolio value over time with Nifty 50 benchmark overlay (normalized).
- **Capital Allocation Donut**: Visual breakdown of cash vs. deployed capital per stock.
- **Realized P&L Bar Chart**: Green/red bar chart showing profit/loss for every closed trade.
- **Chart.js Integration**: All charts powered by Chart.js with INR formatting and dark-theme styling.

---

## 🚀 Performance & Architecture
- **Verified Strategy**: +28.77% Net Profit (Rs. 2.87L on 10L simulated capital).
- **Big Data**: Trained on **3.2 million+ rows** of 10-year OHLCV data.
- **Model**: Optimized XGBoost + LightGBM + CatBoost ensemble with 99 technical + 8 fundamental features.
- **Infrastructure**: Hosted on DigitalOcean VPS with DuckDNS (SSL) and Nginx reverse proxy.

---

## 📂 System Components
- **`services/scanner.py`**: The "Heart" — parallel scans 3000+ stocks with tiered scheduling and Redis caching.
- **`api/main.py`**: The "Bridge" — streams real-time signals via SSE, calculates live P&L, and serves equity history.
- **`src/core/fundamental_analyzer.py`**: The "Brain" — computes AI sentiment and conviction scores.
- **`frontend/`**: The "Face" — High-fidelity, dark-themed dashboard with Chart.js analytics and responsive design.

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

### Check Redis Cache
```bash
docker exec tradesage-redis redis-cli keys "tradesage:ohlc:*" | wc -l
```

---

## ⚠️ Disclaimer
Educational purposes only. This system uses paper trading simulations. Do not trade real capital without extensive testing and clinical risk management. TradeSage is not a financial advisor.
