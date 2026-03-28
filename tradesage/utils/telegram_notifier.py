"""
TradeSage - Telegram Notifier (utils/telegram_notifier.py)

Sends trade signals, portfolio summaries, and error alerts via Telegram.
Features:
  - Retry logic (3 attempts with exponential backoff)
  - Critical-only error alerts (avoids noise)
  - Message length guard (Telegram limit: 4096 chars)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_MSG_LENGTH = 4096
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


def _truncate(msg: str) -> str:
    if len(msg) > _MAX_MSG_LENGTH:
        return msg[: _MAX_MSG_LENGTH - 4] + "…"
    return msg


class TelegramNotifier:
    """
    Thin wrapper around the Telegram Bot API.

    Parameters
    ----------
    token   : Telegram bot token (from @BotFather)
    chat_id : Target chat / channel ID
    """

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = str(chat_id)
        self._bot = None

    def _get_bot(self):
        if self._bot is None:
            try:
                from telegram import Bot  # type: ignore
                self._bot = Bot(token=self.token)
            except ImportError:
                raise ImportError(
                    "python-telegram-bot not installed. "
                    "Run: pip install python-telegram-bot"
                )
        return self._bot

    # ── Core send ─────────────────────────────────────────────────────────

    async def send(
        self,
        message: str,
        parse_mode: str = "Markdown",
        silent: bool = False,
    ) -> bool:
        """
        Send a message with retry logic.

        Returns True on success.
        """
        message = _truncate(message)
        bot = self._get_bot()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_notification=silent,
                )
                return True
            except Exception as exc:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"Telegram send attempt {attempt}/{_MAX_RETRIES} failed: {exc}. "
                    f"Retry in {delay:.1f}s…"
                )
                time.sleep(delay)

        logger.error("Telegram: all send attempts failed.")
        return False

    # ── Pre-built message templates ────────────────────────────────────────

    async def send_signal(
        self,
        symbol: str,
        price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        strategies: str = "",
    ) -> bool:
        rr = (target - price) / max(price - stop_loss, 1e-10)
        emoji = "🟢"
        msg = (
            f"{emoji} *BUY SIGNAL — {symbol}*\n\n"
            f"💰 Price      : ₹{price:,.2f}\n"
            f"🛑 Stop Loss  : ₹{stop_loss:,.2f}\n"
            f"🎯 Target     : ₹{target:,.2f}\n"
            f"📐 R/R Ratio  : {rr:.1f}x\n"
            f"🔮 Confidence : {confidence * 100:.1f}%\n"
        )
        if strategies:
            msg += f"📋 Strategy   : {strategies}\n"
        return await self.send(msg)

    async def send_portfolio_summary(self, summary: dict) -> bool:
        total_emoji = "🟢" if summary.get("total_pnl", 0) >= 0 else "🔴"
        day_emoji = "🟢" if summary.get("todays_pnl", 0) >= 0 else "🔴"

        msg = (
            f"💼 *Portfolio Summary*\n\n"
            f"💵 Cash       : ₹{summary.get('balance', 0):>10,.2f}\n"
            f"📈 Equity     : ₹{summary.get('equity', 0):>10,.2f}\n"
            f"{day_emoji} Today P&L  : ₹{summary.get('todays_pnl', 0):>+10,.2f}\n"
            f"{total_emoji} Total P&L  : ₹{summary.get('total_pnl', 0):>+10,.2f} "
            f"({summary.get('roi', 0):.2f}%)\n"
            f"📊 Positions  : {summary.get('open_positions', 0)} open  |  "
            f"{summary.get('closed_trades', 0)} closed\n"
        )
        return await self.send(msg)

    async def send_error(
        self, error: str, context: str = "", critical: bool = True
    ) -> bool:
        """Send an error alert. Only fires if *critical=True* to reduce noise."""
        if not critical:
            logger.warning(f"[Non-critical error, not sent] {error}")
            return False
        emoji = "🚨"
        msg = f"{emoji} *TradeSage Error*\n\n`{error}`"
        if context:
            msg += f"\n\nContext: {context}"
        return await self.send(msg)

    async def send_drawdown_alert(
        self, todays_pnl: float, limit: float
    ) -> bool:
        msg = (
            f"🛑 *DRAWDOWN PROTECTION ACTIVATED*\n\n"
            f"Today P&L  : ₹{todays_pnl:>+,.2f}\n"
            f"Loss Limit : ₹{limit:>,.2f}\n\n"
            f"Trading paused for today to protect capital."
        )
        return await self.send(msg)
