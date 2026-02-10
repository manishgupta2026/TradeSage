# TradeSage AI - Automated Trading Bot 🤖📈

An AI-powered algorithmic trading system for the Indian stock market (NSE) with 799 strategy combinations, dual-source data validation, and paper trading capabilities.

## Features

- **799 Trading Strategies**: Extracted from professional trading books using AI
- **Dual-Source Validation**: Cross-checks prices between TradingView and Yahoo Finance (>99.9% accuracy)
- **Paper Trading**: Simulate trades for 30 days before going live
- **Telegram Integration**: Real-time alerts and portfolio updates
- **Angel One Broker**: Ready for live trading integration
- **Risk Management**: Automatic position sizing and stop-loss calculation

## Tech Stack

- Python 3.10+
- TradingView (tvDatafeed) + Yahoo Finance (yfinance)
- Telegram Bot API
- Angel One SmartAPI
- pandas, pandas_ta, numpy

## Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo>
cd "Trade AI"
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Paper Trading
```bash
.\venv\Scripts\python main.py --mode=paper --init-capital=100000
```

## Commands (Telegram)

- `/start` - Initialize bot
- `/scan` - Run market scan
- `/portfolio` - View paper trading portfolio
- `/status` - Check market status

## Project Structure

```
Trade AI/
├── src/
│   ├── engine/         # Trading engine (Scanner, DataManager, Indicators)
│   ├── paper/          # Paper trading module
│   ├── bot/            # Telegram bot
│   ├── broker/         # Angel One integration
│   └── extraction/     # Strategy extraction (Phase 1)
├── data/
│   └── strategies/     # 799 extracted strategies (JSON)
├── scripts/            # Utility scripts
└── main.py             # Entry point
```

## Deployment

See `implementation_plan_phase5.md` (coming soon) for cloud deployment instructions.

## License

MIT

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk. Never trade with money you cannot afford to lose.
