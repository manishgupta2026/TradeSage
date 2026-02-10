from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import os
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, cache_dir="data/market_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # Initialize TVDatafeed in guest mode (nologin)
        # Note: Guest mode has limits, but works for limited symbols.
        self.tv = TvDatafeed()

    def fetch_data(self, ticker: str, exchange="NSE", interval=Interval.in_daily, n_bars=300, use_cache=True) -> pd.DataFrame:
        """Fetches historical data from TradingView."""
        cache_path = os.path.join(self.cache_dir, f"{ticker}_{exchange}_daily.parquet")

        if use_cache and os.path.exists(cache_path):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mtime < timedelta(hours=24):
                # print(f"Loading {ticker} from cache...") # Silece log for cleaner output
                df = pd.read_parquet(cache_path)
                return df

        import time
        import random
        
        # print(f"Fetching {ticker} from {exchange} (TradingView) - LIVE...") 
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid rate limits
                time.sleep(random.uniform(0.5, 2.0))
                
                df = self.tv.get_hist(symbol=ticker, exchange=exchange, interval=interval, n_bars=n_bars)
                
                if df is not None and not df.empty:
                    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                    if 'symbol' in df.columns: df.drop(columns=['symbol'], inplace=True)
                    df.to_parquet(cache_path)
                    return df
                else:
                    # If data is empty, it might be an invalid symbol or temporary issue.
                    # Don't retry immediately for empty data unless we suspect connection.
                    if attempt == max_retries - 1:
                        return pd.DataFrame()
            
            except Exception as e:
                # print(f"Error fetching {ticker} (Attempt {attempt+1}/{max_retries}): {e}")
                if "Connection" in str(e) or "timeout" in str(e).lower():
                    # Re-initialize on connection error
                    try:
                        self.tv = TvDatafeed()
                    except:
                        pass
                    time.sleep(2 * (attempt + 1)) # Backoff
                else:
                    # If it's not a connection error (e.g. symbol not found), break
                    break
        
        return pd.DataFrame()

    def verify_price(self, ticker: str, current_price: float) -> dict:
        """Cross-checks price with Yahoo Finance for maximum accuracy."""
        try:
            import yfinance as yf
            ns_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
            
            # Fast fetch of just 1 day
            yf_data = yf.download(ns_ticker, period="1d", interval="1m", progress=False)
            
            if not yf_data.empty:
                yf_price = float(yf_data.iloc[-1]['Close'])
                diff = abs(current_price - yf_price)
                pct_diff = (diff / yf_price) * 100
                
                is_accurate = pct_diff < 0.5 # 0.5% tolerance
                
                return {
                    "is_accurate": is_accurate,
                    "yf_price": yf_price,
                    "diff_pct": round(pct_diff, 2),
                    "source_match": True
                }
            return {"is_accurate": True, "source_match": False, "note": "YF Data Unavailable"}
            
        except Exception as e:
            print(f"Validation Error {ticker}: {e}")
            return {"is_accurate": True, "source_match": False, "note": "Validation Failed"}

    def fetch_fundamentals(self, ticker: str) -> dict:
        """Fetches key financial metrics using yfinance."""
        try:
            import yfinance as yf
            # yfinance expects .NS for NSE
            ns_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
            stock = yf.Ticker(ns_ticker)
            info = stock.info
            
            return {
                "pe_ratio": info.get("trailingPE", None),
                "market_cap": info.get("marketCap", None),
                "roe": info.get("returnOnEquity", None),
                "sector": info.get("sector", "Unknown"),
                "high_52": info.get("fiftyTwoWeekHigh", 0),
                "low_52": info.get("fiftyTwoWeekLow", 0)
            }
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return {}

    def get_market_status(self):
        """Checks if Indian market is currently open (9:15-15:30 IST)."""
        now = datetime.now() 
        day = now.weekday()
        if day >= 5: return False
        current_time = now.time()
        start_time = datetime.strptime("09:15", "%H:%M").time()
        end_time = datetime.strptime("15:30", "%H:%M").time()
        return start_time <= current_time <= end_time
