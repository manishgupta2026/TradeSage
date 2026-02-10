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
                print(f"   Auth Token: {data['data']['jwtToken'][:10]}...")
                return True
            else:
                print(f"❌ Login Failed: {data['message']}")
                print(f"   Error Code: {data['errorcode']}")
                return False
        except Exception as e:
            print(f"❌ Login Error: {e}")
            return False

if __name__ == "__main__":
    angel = AngelOneManager()
    angel.login()
