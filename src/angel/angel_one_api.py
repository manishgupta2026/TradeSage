"""
Angel One API Wrapper Module
Handles authentication, TOTP generation, and session management.
"""

import json
import os
import pyotp
import logging
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
        
        self.load_credentials()
        self.connect()
        
    def load_credentials(self):
        """Load credentials from JSON config"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found. Please create one based on angel_config_template.json.")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
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
    except FileNotFoundError:
        print(f"\n❌ Error: '{config_file}' not found.")
        print("Please copy 'angel_config_template.json' to 'angel_config.json' and fill in your details.")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
