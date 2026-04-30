"""
Angel One API Wrapper Module
Handles authentication, TOTP generation, session management,
and real-time LTP (Last Traded Price) fetching.
"""

import json
import os
import pyotp
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from SmartApi import SmartConnect

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AngelOneAPI:
    """Wrapper for Angel One SmartConnect API"""
    
    def __init__(self, config_path="angel_config.json"):
        self.config_path = config_path
        self.api_key = None
        self.client_id = None
        self.password = None
        self.totp_secret = None
        self.smartApi = None
        self.auth_token = None
        self.api_config = None
        self._symbol_to_token = {}   # Cached instrument token map
        self._token_map_loaded = False
        
        self.load_credentials()
        self.connect()
        
    def load_credentials(self):
        """Load credentials from JSON config"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found. Please create one based on angel_config_template.json.")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        self.api_config = config
            
        self.api_key = config.get('api_key')
        self.client_id = config.get('client_id')
        self.password = config.get('password')
        self.totp_secret = config.get('totp_token')
        
        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            raise ValueError("Missing credentials in config file. Ensure api_key, client_id, password, and totp_token are present.")
            
    def connect(self):
        """Establish connection with SmartApi"""
        try:
            self.smartApi = SmartConnect(api_key=self.api_key)
            
            # Generate TOTP code
            totp = pyotp.TOTP(self.totp_secret).now()
            
            # Authenticate
            data = self.smartApi.generateSession(self.client_id, self.password, totp)
            
            if data['status'] == False:
                logger.error(f"Login failed: {data['message']}")
                raise Exception(f"Login failed: {data['message']}")
                
            self.auth_token = data['data']['jwtToken']
            self.feed_token = self.smartApi.getfeedToken()
            
            logger.info("✅ Successfully connected to Angel One API")
            logger.info(f"   Client ID: {self.client_id}")
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise
            
    def get_api(self):
        """Return the authenticated SmartConnect instance"""
        if self.smartApi is None:
            self.connect()
        return self.smartApi

    # ──────────────────────────────────────────────
    #  INSTRUMENT TOKEN MAP (for LTP lookups)
    # ──────────────────────────────────────────────

    def _ensure_token_map(self):
        """Load instrument master and build symbol→token map (cached)."""
        if self._token_map_loaded:
            return

        # Try loading from a local cache file first
        cache_dir = Path(self.config_path).resolve().parent.parent / "data_cache_angel"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "instruments.json"

        instruments = []
        if cache_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age < timedelta(days=1):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        instruments = json.load(f)
                except Exception:
                    pass

        if not instruments:
            try:
                url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    instruments = resp.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(instruments, f)
                    logger.info(f"Downloaded {len(instruments)} instruments from Angel One")
            except Exception as e:
                logger.warning(f"Failed to download instrument master: {e}")

        for instr in instruments:
            if instr.get("exch_seg") == "NSE" and instr.get("symbol", "").endswith("-EQ"):
                base = instr["symbol"].replace("-EQ", "")
                self._symbol_to_token[base] = instr.get("token")

        self._token_map_loaded = True
        logger.info(f"Instrument token map: {len(self._symbol_to_token)} NSE equities")

    def get_ltp(self, symbol: str) -> float:
        """
        Fetch real-time Last Traded Price for a single NSE equity symbol.
        Returns the LTP as a float, or None on failure.
        
        Example: get_ltp("RELIANCE") → 2854.50
        """
        self._ensure_token_map()
        token = self._symbol_to_token.get(symbol)
        if not token:
            logger.debug(f"LTP: symbol {symbol} not found in instrument map")
            return None

        try:
            data = self.smartApi.ltpData("NSE", f"{symbol}-EQ", token)
            if data and data.get("status") and data.get("data"):
                ltp = data["data"].get("ltp")
                if ltp is not None:
                    return float(ltp)
            logger.debug(f"LTP response for {symbol}: {data}")
            return None
        except Exception as e:
            logger.warning(f"LTP fetch failed for {symbol}: {e}")
            return None

    def get_ltp_batch(self, symbols: list) -> dict:
        """
        Fetch LTP for multiple symbols. Returns {symbol: ltp_float}.
        Symbols that fail are excluded from the result.
        """
        result = {}
        for sym in symbols:
            ltp = self.get_ltp(sym)
            if ltp is not None:
                result[sym] = ltp
        return result

    def logout(self):
        """Terminate the session securely"""
        if self.smartApi:
            try:
                logout_response = self.smartApi.terminateSession(self.client_id)
                logger.info("Session terminated successfully")
                return logout_response
            except Exception as e:
                logger.error(f"Logout failed: {str(e)}")

# Allow testing the file directly
if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'angel_config.json'
    
    try:
        api = AngelOneAPI(config_file)
        print("\nAPI connection test successful!")
        
        # Test LTP
        test_symbols = ["RELIANCE", "TCS", "INFY"]
        ltps = api.get_ltp_batch(test_symbols)
        print(f"\nLTP results:")
        for sym, price in ltps.items():
            print(f"  {sym}: ₹{price:.2f}")
            
    except FileNotFoundError:
        print(f"\n❌ Error: '{config_file}' not found.")
        print("Please copy 'angel_config_template.json' to 'angel_config.json' and fill in your details.")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
