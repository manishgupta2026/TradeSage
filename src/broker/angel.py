import os
from SmartApi import SmartConnect
import pyotp
from dotenv import load_dotenv

load_dotenv()

class AngelOneManager:
    def __init__(self):
        # Clean keys just in case
        self.api_key = os.getenv("ANGEL_MARKET_KEY", "").strip()
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "").strip()
        self.password = os.getenv("ANGEL_MPIN", "").strip()
        self.token = os.getenv("ANGEL_TOTP_SECRET", "").strip()
        
        self.smartApi = SmartConnect(api_key=self.api_key)

    def login(self):
        print(f"Attempting login for Client ID: {self.client_id} ...")
        
        if not self.password or not self.token:
            print("❌ Missing MPIN or TOTP Secret. Cannot login.")
            return False

        try:
            # Generate TOTP
            try:
                totp = pyotp.TOTP(self.token).now()
            except Exception as e:
                 print(f"❌ Error generating TOTP (Invalid Secret?): {e}")
                 return False
                 
            # Login
            data = self.smartApi.generateSession(self.client_id, self.password, totp)
            
            if data['status']:
                print(f"✅ Angel One Login SUCCESSFUL!")
                return True
            else:
                print(f"❌ Login Failed: {data['message']}")
                return False
        except Exception as e:
            print(f"❌ Login Error: {e}")
            return False

    def get_token_map(self):
        """Downloads/Loads Scrip Master and returns Symbol->Token map for NSE Equity."""
        import requests
        import json
        
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        cache_path = "data/angel_scrip_master.json"
        
        # Download if missing or old (>24h)
        download = True
        if os.path.exists(cache_path):
            from datetime import datetime
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - mtime).days < 1:
                download = False
        
        if download:
            print("Downloading Angel One Scrip Master...")
            try:
                r = requests.get(url)
                with open(cache_path, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                print(f"❌ Failed to download Scrip Master: {e}")
                if not os.path.exists(cache_path): return {}

        # Parse
        print("Loading Scrip Master...")
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            # Filter for NSE Equity
            token_map = {}
            for item in data:
                if item['exch_seg'] == 'NSE' and item['symbol'].endswith('-EQ'):
                    # Symbol in scrip master: "RELIANCE-EQ" -> we want "RELIANCE"
                    sym = item['name'] 
                    token = item['token']
                    token_map[sym] = token
            print(f"Mapped {len(token_map)} NSE Equity Tokens.")
            return token_map
        except Exception as e:
            print(f"❌ Error parsing Scrip Master: {e}")
            return {}

    def get_historical_data(self, symbol, token, interval="ONE_DAY", days=365):
        """Fetches historical candles."""
        from datetime import datetime, timedelta
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Format: yyyy-mm-dd HH:MM
        from_str = from_date.strftime("%Y-%m-%d %H:%M")
        to_str = to_date.strftime("%Y-%m-%d %H:%M")
        
        try:
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_str,
                "todate": to_str
            }
            data = self.smartApi.getCandleData(params)
            
            if data['status'] and data['data']:
                return data['data'] # [[timestamp, open, high, low, close, vol], ...]
            else:
                # print(f"Error fetching {symbol}: {data.get('message', 'Unknown')}")
                return None
        except Exception as e:
            print(f"Exception fetching {symbol}: {e}")
            return None

if __name__ == "__main__":
    angel = AngelOneManager()
    angel.login()
