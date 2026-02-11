from src.broker.angel import AngelOneManager
import pandas as pd
import os
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, cache_dir="data/market_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize Angel One
        self.angel = AngelOneManager()
        if self.angel.login():
            self.token_map = self.angel.get_token_map()
            self.use_angel = True
        else:
            print("⚠️ Angel One Login Failed. Falling back to yfinance (unreliable).")
            self.use_angel = False
            self.token_map = {}

    def fetch_data(self, ticker: str, exchange="NSE", interval="1d", n_bars=300, use_cache=True) -> pd.DataFrame:
        """Fetches historical data using Angel One (Primary) or yfinance (Fallback)."""
        cache_path = os.path.join(self.cache_dir, f"{ticker}_{exchange}_daily.parquet")

        if use_cache and os.path.exists(cache_path):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mtime < timedelta(hours=24):
                try:
                    df = pd.read_parquet(cache_path)
                    return df
                except:
                    pass 

        # Try Angel One first
        if self.use_angel:
            clean_ticker = ticker.replace(".NS", "")
            token = self.token_map.get(clean_ticker)
            
            if token:
                # print(f"Fetching {clean_ticker} (Token: {token}) via Angel One...")
                data = self.angel.get_historical_data(clean_ticker, token)
                if data:
                    try:
                        # Columns: [timestamp, open, high, low, close, volume]
                        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        
                        # Ensure numeric
                        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        for c in cols: df[c] = pd.to_numeric(df[c])
                        
                        df.to_parquet(cache_path)
                        return df
                    except Exception as e:
                        print(f"Error parsing Angel data for {ticker}: {e}")

        # Fallback to yfinance
        import yfinance as yf
        ns_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        
        try:
            df = yf.download(ns_ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.droplevel(1)
                df.to_parquet(cache_path)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            # print(f"Error fetching {ticker}: {e}")
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
