"""
TradeSage - Angel One Trade Executor (trading/angel_one_executor.py)

Handles:
  - Angel One SmartAPI authentication (TOTP-based)
  - API rate limiting (3 req/s)
  - Market hours check (NSE: 9:15–15:30 IST)
  - Position sizing (max 5% capital per stock)
  - Order placement & rejection handling
  - Retry logic (3 attempts with exponential backoff)
"""

import logging
import time
from datetime import datetime, time as dtime
from functools import wraps
from typing import Optional

import pyotp

logger = logging.getLogger(__name__)

# ── NSE Market Hours (IST) ────────────────────────────────────────────────────
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
LAST_ENTRY_TIME = dtime(15, 20)  # no new entries after 3:20 PM (intraday gap risk)

# ── Rate limit (Angel One allows ~3 req/s) ────────────────────────────────────
_RATE_LIMIT_INTERVAL = 0.35      # seconds between requests
_last_request_time: float = 0.0

# ── Retry settings ────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0           # seconds (doubles each retry)


def _rate_limit() -> None:
    """Block until the rate-limit interval has elapsed since the last call."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _RATE_LIMIT_INTERVAL:
        time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def retry_with_backoff(func):
    """Decorator: retry *func* up to MAX_RETRIES times with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[{func.__name__}] Attempt {attempt}/{MAX_RETRIES} failed: "
                    f"{exc}. Retrying in {delay:.1f}s…"
                )
                time.sleep(delay)
        logger.error(f"[{func.__name__}] All {MAX_RETRIES} attempts failed.")
        raise last_exc
    return wrapper


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Return True if NSE is currently open (Mon–Fri, 9:15–15:30 IST)."""
    if now is None:
        now = datetime.now()
    if now.weekday() >= 5:       # Saturday=5, Sunday=6
        return False
    t = now.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def can_enter_new_trade(now: Optional[datetime] = None) -> bool:
    """Return True if it is still safe to enter a new trade (before 3:20 PM)."""
    if not is_market_open(now):
        return False
    if now is None:
        now = datetime.now()
    return now.time() <= LAST_ENTRY_TIME


class AngelOneExecutor:
    """
    Wraps the Angel One SmartAPI for order execution.

    Parameters
    ----------
    client_id   : Angel One client ID
    mpin        : MPIN (4-digit)
    totp_secret : Base32 secret for TOTP 2FA
    api_key     : Angel One API key
    max_position_pct : max fraction of portfolio per stock (default 5%)
    """

    def __init__(
        self,
        client_id: str,
        mpin: str,
        totp_secret: str,
        api_key: str,
        max_position_pct: float = 0.05,
    ):
        self.client_id = client_id
        self.mpin = mpin
        self.totp_secret = totp_secret
        self.api_key = api_key
        self.max_position_pct = max_position_pct
        self._api = None
        self._session_token: Optional[str] = None

    # ── Authentication ────────────────────────────────────────────────────

    @retry_with_backoff
    def login(self) -> bool:
        """
        Authenticate with Angel One SmartAPI using TOTP.

        Returns True on success.
        """
        try:
            from smartapi import SmartConnect  # type: ignore
        except ImportError:
            raise ImportError(
                "smartapi-python not installed. Run: pip install smartapi-python"
            )

        totp = pyotp.TOTP(self.totp_secret).now()
        _rate_limit()
        api = SmartConnect(api_key=self.api_key)
        resp = api.generateSession(self.client_id, self.mpin, totp)

        if resp.get("status"):
            self._api = api
            self._session_token = resp["data"]["jwtToken"]
            logger.info("✓ Angel One login successful")
            return True

        raise RuntimeError(f"Angel One login failed: {resp.get('message', 'unknown')}")

    def _ensure_logged_in(self) -> None:
        if self._api is None or self._session_token is None:
            self.login()

    # ── Position Sizing ───────────────────────────────────────────────────

    def calculate_position_size(
        self, portfolio_value: float, price: float, atr: float
    ) -> int:
        """
        Risk-based sizing: risk at most *max_position_pct* of portfolio.
        Additional hard cap: no more than 5% of portfolio by value.
        """
        if price <= 0 or atr <= 0:
            return 0
        risk_amount = portfolio_value * self.max_position_pct
        shares_by_risk = int(risk_amount / (3.0 * atr))       # 3x ATR SL
        shares_by_value = int((portfolio_value * self.max_position_pct) / price)
        return max(0, min(shares_by_risk, shares_by_value))

    # ── Order Execution ───────────────────────────────────────────────────

    @retry_with_backoff
    def place_order(
        self,
        symbol: str,
        token: str,
        qty: int,
        order_type: str = "BUY",
        product: str = "DELIVERY",  # "INTRADAY" or "DELIVERY"
        price: float = 0.0,         # 0 = market order
        exchange: str = "NSE",
    ) -> Optional[str]:
        """
        Place a market or limit order on Angel One.

        Returns order ID string on success, None on failure.
        """
        self._ensure_logged_in()

        if not can_enter_new_trade():
            logger.warning(
                f"⏰ Market closed or past entry deadline — skipping {symbol}"
            )
            return None

        _rate_limit()

        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": symbol,
            "symboltoken": token,
            "transactiontype": order_type.upper(),
            "exchange": exchange,
            "ordertype": "MARKET" if price == 0 else "LIMIT",
            "producttype": product,
            "duration": "DAY",
            "price": str(price),
            "quantity": str(qty),
        }

        try:
            resp = self._api.placeOrder(order_params)
        except Exception as exc:
            raise RuntimeError(f"Order placement exception for {symbol}: {exc}") from exc

        if resp.get("status"):
            order_id = resp["data"]["orderid"]
            logger.info(f"✓ Order placed: {order_type} {qty} {symbol} → ID={order_id}")
            return order_id

        message = resp.get("message", "unknown rejection")
        error_code = resp.get("errorcode", "")
        raise RuntimeError(
            f"Order rejected for {symbol}: [{error_code}] {message}"
        )

    @retry_with_backoff
    def get_portfolio(self) -> list:
        """Fetch current holdings from Angel One."""
        self._ensure_logged_in()
        _rate_limit()
        resp = self._api.holding()
        if resp.get("status"):
            return resp.get("data", []) or []
        logger.warning(f"Could not fetch portfolio: {resp.get('message')}")
        return []

    @retry_with_backoff
    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        """Fetch Last Traded Price for a symbol."""
        self._ensure_logged_in()
        _rate_limit()
        resp = self._api.ltpData(exchange, symbol, token)
        if resp.get("status"):
            return float(resp["data"]["ltp"])
        return None

    def logout(self) -> None:
        """Terminate the Angel One session."""
        if self._api:
            try:
                _rate_limit()
                self._api.terminateSession(self.client_id)
                logger.info("Angel One session terminated.")
            except Exception as exc:
                logger.warning(f"Logout error: {exc}")
            finally:
                self._api = None
                self._session_token = None
