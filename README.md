# TradeSage AI - Intelligent Swing Trading Bot 🤖📈

TradeSage AI is a fully automated, **serverless swing trading bot** designed for the Indian Stock Market (NSE). It combines a rigorous **Technical Strategy Funnel** with **AI-Based Sentiment Analysis** to identify high-probability trade setups.

## 🚀 Key Features

*   **1200+ Stock Universe:** Scans Nifty 500, Microcaps, and High-Beta stocks daily.
*   **Funnel Strategy:**
    1.  **Trend Filter:** EMA 200 checks for long-term uptrends.
    2.  **Momentum Filter:** RSI (14) ensures the stock is not overbought/oversold.
    3.  **Pattern Recognition:** Detects Candlestick patterns (Engulfing, Hammer, Morning Star).
*   **🧠 AI "Brain" (New!):**
    *   Fetches real-time news via **Google News RSS**.
    *   Uses **Llama-3-70b (Groq)** to score sentiment (-1 to +1).
    *   **Safety Net:** Rejects technically good trades if breaking news is negative.
*   **Infrastructure:**
    *   **Data:** **Angel One SmartAPI** (Official Data) + *yfinance* (Fallback).
    *   **Execution:** **GitHub Actions** (Runs automatically at 09:25 AM & 02:55 PM IST).
    *   **Paper Trading:** Persists portfolio (Holdings, P&L) across runs using GitHub Cache.
*   **Reporting:** Sends detailed **Telegram Alerts** with:
    *   Entry Price, Stop Loss, Target.
    *   AI Sentiment Reason.
    *   Portfolio Summary (Equity, ROI, Holdings Breakdown).

## 🛠 Tech Stack

*   **Language:** Python 3.10
*   **Broker:** Angel One (SmartAPI)
*   **LLM:** Groq (Llama-3)
*   **Deployment:** GitHub Actions (Cron + Workflow Dispatch)
*   **Libraries:** `pandas-ta`, `feedparser`, `smartapi-python`, `python-telegram-bot`

## ⚙️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/manishgupta2026/TradeSage.git
    cd TradeSage
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Secrets (GitHub Settings -> Secrets):**
    *   `ANGEL_CLIENT_ID`, `ANGEL_MPIN`, `ANGEL_TOTP_SECRET`, `ANGEL_MARKET_KEY` (Broker)
    *   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (Alerts)
    *   `GROQ_API_KEY` (AI Sentiment)

4.  **Run Manually (Local):**
    ```bash
    python main.py
    ```
    *(Note: Local runs require `.env` file with the above keys)*

## 📊 Workflow

1.  **Trigger:** Scheduled Cron or Manual Dispatch.
2.  **Restore:** Loads `paper_portfolio.json` & `angel_scrip_master.json` from Cache.
3.  **Data Fetch:** Downloads daily candles for 1200 stocks via Angel One.
4.  **Scan:** Applies Technical Funnel Filters.
5.  **AI Analysis:** Checks news for shortlisted candidates.
6.  **Trade:** Executes Paper Trades (Buy/Sell/SL Hit/Target Hit).
7.  **Report:** Sends Telegram summary.
8.  **Save:** Caches updated portfolio state.

## ⚠️ Disclaimer

This project is for **educational purposes only**. The "Paper Trading" mode simulates trades with fake money. Do not use this for live trading with real capital unless you fully understand the risks. The authors are not responsible for any financial losses.

---
*Built with ❤️ by TradeSage Team*
