import requests
import logging

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials missing. Alerts will only be logged locally.")

    def send_message(self, message):
        """Send a basic message to Telegram."""
        if not self.token or not self.chat_id:
            logger.info(f"[TELEGRAM STUB]: {message}")
            return
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def send_trade_alert(self, symbol, action, price, confidence=None, sl=None, tp=None):
        """Format and send a structured trade alert."""
        if action.upper() == "ENTRY":
            msg = f"🟢 *{symbol}* - BUY SIGNAL\n"
            msg += f"Price: ₹{price:,.2f}\n"
            if confidence is not None:
                msg += f"Confidence: {confidence*100:.1f}%\n"
            if sl is not None:
                msg += f"Stop Loss: ₹{sl:,.2f}\n"
            if tp is not None:
                msg += f"Take Profit: ₹{tp:,.2f}\n"
            msg += f"Strategy: 20-Year Optimized (15d Max Hold, 4.5x ATR TP)"
        else:
            msg = f"🔴 *{symbol}* - SELL ALARM\n"
            msg += f"Exit Price: ₹{price:,.2f}\n"
            msg += f"Reason: {confidence}" # Using confidence param for reason in EXIT
            
        self.send_message(msg)
