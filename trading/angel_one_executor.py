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
        """Fetches historical candles with rate limiting and retry logic."""
        import time
        from datetime import datetime, timedelta

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # Format: yyyy-mm-dd HH:MM
        from_str = from_date.strftime("%Y-%m-%d %H:%M")
        to_str = to_date.strftime("%Y-%m-%d %H:%M")

        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_str,
            "todate": to_str
        }

        max_retries = 3
        base_delay = 1  # 1 second base delay

        for attempt in range(max_retries):
            try:
                # Rate limiting: Angel One limit is ~3 req/sec
                time.sleep(0.35)

                data = self.smartApi.getCandleData(params)

                if data and data.get('status') and data.get('data'):
                    return data['data'] # [[timestamp, open, high, low, close, vol], ...]
                else:
                    msg = data.get('message', 'Unknown Error') if data else 'Empty Response'
                    if "Too Many Requests" in msg or "Rate Limit" in msg:
                        raise ValueError("Rate Limit Exceeded")

                    return None

            except Exception as e:
                print(f"⚠️ Exception fetching {symbol} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s...
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"Sleeping {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                else:
                    self._send_telegram_alert(f"CRITICAL: Failed to fetch data for {symbol} after {max_retries} attempts. Error: {e}")
                    return None

    def execute_order(self, symbol, token, quantity, action, order_type="MARKET", price=0):
        """Executes a real trade on Angel One with safety checks."""
        import time
        from datetime import datetime

        # 1. Market Hours Check (9:15 AM - 3:30 PM IST)
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if now < market_open or now > market_close:
            msg = f"Skipping order for {symbol}: Outside market hours."
            print(msg)
            return {"status": False, "message": msg}

        # 2. Position Size Check (assuming max 5% rule handled in calling logic, but we sanity check quantity)
        if quantity <= 0:
            return {"status": False, "message": "Invalid quantity."}

        try:
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": f"{symbol}-EQ",
                "symboltoken": token,
                "transactiontype": action, # "BUY" or "SELL"
                "exchange": "NSE",
                "ordertype": order_type,
                "producttype": "DELIVERY", # Swing trading
                "duration": "DAY",
                "price": price,
                "squareoff": "0",
                "stoploss": "0",
                "quantity": quantity
            }

            # Rate limiting
            time.sleep(0.35)

            orderId = self.smartApi.placeOrder(orderparams)

            if orderId:
                msg = f"✅ SUCCESS: {action} {quantity} shares of {symbol}. Order ID: {orderId}"
                print(msg)
                self._send_telegram_alert(msg)
                return {"status": True, "order_id": orderId}
            else:
                msg = f"❌ ORDER REJECTED: {action} {symbol}. Check funds/margins."
                print(msg)
                self._send_telegram_alert(msg)
                return {"status": False, "message": "Order Rejected"}

        except Exception as e:
            msg = f"❌ API ERROR executing {action} for {symbol}: {e}"
            print(msg)
            self._send_telegram_alert(f"CRITICAL: {msg}")
            return {"status": False, "message": str(e)}

    def _send_telegram_alert(self, message):
        """Sends a critical alert to Telegram."""
        import requests
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not token or not chat_id:
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": f"🤖 TradeSage Alert:\n{message}"}

        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")

if __name__ == "__main__":
    angel = AngelOneManager()
    angel.login()
