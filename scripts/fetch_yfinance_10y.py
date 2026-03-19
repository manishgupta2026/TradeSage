import os
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

def fetch_yfinance_data(symbols, years=10, max_workers=10, cache_dir='data_cache_yfinance'):
    """
    Fetches historical daily data for Indian stocks using yfinance.
    Saves them to CSV files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # yfinance uses '.NS' suffix for NSE stocks
    yf_symbols = [f"{s}.NS" for s in symbols]
    
    results = {}
    failed = []
    
    period = f"{years}y"
    
    def fetch_single(symbol):
        base_symbol = symbol.replace('.NS', '')
        cache_file = os.path.join(cache_dir, f"{base_symbol}_daily.csv")
        
        # Check if already cached
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
                if len(df) > 200:
                    return base_symbol, df
            except:
                pass
                
        try:
            ticker = yf.Ticker(symbol)
            # Fetch data
            df = ticker.history(period=period)
            
            if df.empty or len(df) < 200:
                return base_symbol, None
                
            # Clean up columns to match our feature engineer format
            df.reset_index(inplace=True)
            # yfinance returns Date or Datetime
            time_col = 'Date' if 'Date' in df.columns else 'Datetime'
            df.rename(columns={
                time_col: 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Save to cache
            df.to_csv(cache_file)
            return base_symbol, df
            
        except Exception as e:
            return base_symbol, None

    print(f"📥 Fetching {years} years of data for {len(symbols)} stocks from yfinance...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_single, s): s for s in yf_symbols}
        
        for future in tqdm(as_completed(future_to_symbol), total=len(yf_symbols), desc="Downloading"):
            symbol, df = future.result()
            if df is not None:
                results[symbol] = df
            else:
                failed.append(symbol)
                
    print(f"✓ Successfully fetched data for {len(results)} stocks")
    print(f"⚠ Failed to fetch {len(failed)} stocks")
    
    return results

if __name__ == "__main__":
    # Example: you can load your existing top 3000 NSE JSON list
    json_path = os.path.join('data', 'nse_top_3000_angel.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            symbols = json.load(f)
    else:
        # Fallback to a small list for testing
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        
    print(f"Loaded {len(symbols)} symbols. Starting fetch...")
    
    # Fetch 10 years of data
    stock_data = fetch_yfinance_data(symbols, years=10, max_workers=10)
    
    print("\nData fetching complete. You can now run the standard training pipeline on 'data_cache_yfinance'.")
