"""
Angel One Data Fetcher Module
Handles fetching instruments, historical candle data, and managing local cache.
"""

import os
import json
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.angel_one_api import AngelOneAPI

logger = logging.getLogger(__name__)

class AngelDataFetcher:
    """Historical Data Fetcher using Angel One SmartAPI"""
    
    def __init__(self, api_client: AngelOneAPI, cache_dir='data_cache_angel'):
        self.api = api_client.get_api()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.instrument_cache_file = self.cache_dir / 'instruments.json'
        self.instrument_list = []
        self.symbol_to_token = {}

    def get_instruments(self, force_fetch=False):
        """Fetch all instruments from Angel One and cache them"""
        if self.instrument_cache_file.exists() and not force_fetch:
            file_time = datetime.fromtimestamp(self.instrument_cache_file.stat().st_mtime)
            # Use cache if it's less than 24h old
            if datetime.now() - file_time < timedelta(days=1):
                logger.info("Loading instruments from cache...")
                with open(self.instrument_cache_file, 'r', encoding='utf-8') as f:
                    self.instrument_list = json.load(f)
                    self._build_token_map()
                return self.instrument_list
                
        logger.info("Fetching fresh instruments list from Angel One...")
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                self.instrument_list = response.json()
                with open(self.instrument_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.instrument_list, f)
                self._build_token_map()
                logger.info(f"✓ Downloaded {len(self.instrument_list)} instruments")
                return self.instrument_list
            else:
                logger.error(f"Failed to fetch instrument list: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return []
            
    def _build_token_map(self):
        """Build mapping of trading symbol to instrument token for NSE Equities"""
        for instr in self.instrument_list:
            # -EQ distinguishes Equities from F&O symbols
            if instr.get('exch_seg') == 'NSE' and instr.get('symbol').endswith('-EQ'):
                base_symbol = instr.get('symbol').replace('-EQ', '')
                self.symbol_to_token[base_symbol] = instr.get('token')

    def get_top_nse_stocks(self, count=500, save_path='data/nse_top_500_angel.json'):
        """Extract NSE equities and save to file"""
        if not self.instrument_list:
            self.get_instruments()
            
        logger.info(f"FETCHING TOP {count} NSE STOCKS FROM ANGEL ONE")
        
        # In a real scenario, you'd filter by market cap or volume.
        # Here we fetch valid NSE equity symbols we just parsed.
        # We also filter out bonds, ETFs (by checking standard stock names patterns generally)
        symbols = list(self.symbol_to_token.keys())
        
        # Optional: You can hardcode or seed a quality list, but here we just take the first 'count'
        # To avoid junk, we could optionally sort them or use NIFTY constituents, but sticking to 
        # a broad NSE sweep for the required volume:
        selected_symbols = sorted(symbols)[:count]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(selected_symbols, f, indent=4)
            
        logger.info(f"✓ Retrieved {len(selected_symbols)} stocks")
        logger.info(f"First 20: {', '.join(selected_symbols[:20])}...")
        logger.info(f"✓ Saved to: {save_path}")
        return selected_symbols

    def fetch_historical_data(self, symbol, period_days=730):
        """Fetch daily historical candles for a specific symbol"""
        if not self.symbol_to_token:
            self.get_instruments()
            
        token = self.symbol_to_token.get(symbol)
        
        if not token:
            logger.warning(f"Symbol {symbol} not found in NSE Equities index.")
            return None
            
        # Check cache first
        cache_file = self.cache_dir / f"{symbol}_daily.csv"
        if cache_file.exists():
            # If fetched within 24 hours, use cache to save API limits
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time < timedelta(hours=24):
                try:
                    df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
                    if len(df) > 200:
                        return df
                except Exception:
                    pass

        # Time calculations
        to_date = datetime.now()
        from_date = to_date - timedelta(days=period_days)
        
        historicParam = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": "ONE_DAY",
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"), 
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                candle_data = self.api.getCandleData(historicParam)
                
                if candle_data.get('status') and candle_data.get('data'):
                    # Data format: [timestamp, open, high, low, close, volume]
                    df = pd.DataFrame(candle_data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                    df.dropna(inplace=True)
                    
                    # Save to cache
                    df.to_csv(cache_file)
                    
                    return df
                elif not candle_data.get('status'):
                    # Handle rate limit from JSON structure if available
                    import time
                    if "exceeding access rate" in str(candle_data):
                        time.sleep(1 + attempt * 2)
                        continue
                
                # If valid but empty or other status gracefully fail
                return None
                    
            except Exception as e:
                import time
                # The exception itself throws "exceeding access rate"
                if "exceeding access rate" in str(e).lower() or "429" in str(e):
                    # Exponential backoff on rate limits
                    time.sleep(1.5 + attempt * 2)
                    continue
                else:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    return None
                    
        return None

    def fetch_multiple_symbols(self, symbols, period_days=730, max_workers=3):
        """Fetch multiple symbols in parallel"""
        if not self.symbol_to_token:
            self.get_instruments()
            
        logger.info(f"📥 Fetching data for {len(symbols)} stocks from Angel One...")
        
        results = {}
        failed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_historical_data, symbol, period_days): symbol 
                for symbol in symbols
            }
            
            with tqdm(total=len(symbols), desc="Downloading", unit="stock") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        # Minimum requirement: Need 200 days to calculate 200 SMA
                        if df is not None and len(df) > 200:
                            results[symbol] = df
                        else:
                            failed.append(symbol)
                    except Exception as e:
                        failed.append(symbol)
                    pbar.update(1)
                    
        logger.info(f"✓ Successfully fetched: {len(results)} stocks")
        if failed:
            logger.warning(f"⚠ Failed to fetch: {len(failed)} stocks")
            
        return results, failed
