import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from datetime import time
import pytz

# Project Imports
# Ensure src is in path if running from root
import sys
sys.path.append(os.getcwd())

from src.engine.scanner import NSEScanner
from src.engine.data_manager import DataManager

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

class TradeSageBot:
    def __init__(self, mode='scan', trader=None):
        self.mode = mode
        self.trader = trader
        self.dm = DataManager()
        self.scanner = NSEScanner(self.dm)
        print(f"🤖 Bot Initialized. Mode: {self.mode.upper()}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        mode_text = f" ({self.mode.upper()} MODE)" if self.mode != 'scan' else ""
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🤖 **TradeSage AI is Online!**{mode_text}\n\nCommands:\n/scan - Run Live Market Scan 📉\n/status - Check Market Status 🕒\n/portfolio - View Portfolio (Paper Mode) 💼\n/help - Show this menu"
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        is_open = self.dm.get_market_status()
        status_text = "🟢 **Market is OPEN**" if is_open else "🔴 **Market is CLOSED**"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=status_text, parse_mode='Markdown')

    async def portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show paper trading portfolio status"""
        if self.mode != 'paper' or not self.trader:
            await context.bot.send_message(
                chat_id=update.effective_chat.id, 
                text="⚠️ Portfolio command only available in Paper Trading mode."
            )
            return
        
        summary = self.trader.get_summary()
        holdings = self.trader.portfolio.get('holdings', {})
        
        msg = "💼 **Paper Trading Portfolio**\n\n"
        msg += f"💰 **Cash Balance:** ₹{summary['balance']:,.2f}\n"
        msg += f"📊 **Open Positions:** {summary['open_positions']}\n"
        msg += f"📈 **Closed Trades:** {summary['closed_trades']}\n\n"
        
        if holdings:
            msg += "**Current Holdings:**\n"
            for ticker, pos in holdings.items():
                msg += f"• {ticker}: {pos['qty']} @ ₹{pos['avg_price']}\n"
        else:
            msg += "_No open positions_\n"
        
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode='Markdown')

    async def run_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text="🔍 **Scanning Full Market...** (Analyzing 50+ stocks)")
        
        try:
            # Run the scanner
            results = self.scanner.scan_market()
            
            if not results:
                await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="⚠️ No strong 'BUY' signals found right now.")
                return

            # Format Results with Rich Detail
            msg = "🚀 **TradeSage Market Analysis (Real-Time)**\n\n"
            
            for res in results[:3]: # Top 3 only to avoid spamming
                ticker = res['ticker']
                price = res['price']
                strategies = res['active_strategies']
                
                # Fetch Fundamentals (Real-time check)
                funds = self.dm.fetch_fundamentals(ticker)
                pe = round(funds.get('pe_ratio', 0), 1) if funds.get('pe_ratio') else "N/A"
                sector = funds.get('sector', "N/A")
                
                # Dual-Source Validation (TV vs YF)
                val = self.dm.verify_price(ticker, price)
                accuracy_icon = "✅" if val['is_accurate'] else "⚠️"
                accuracy_note = ""
                if not val['is_accurate'] and val['source_match']:
                    accuracy_note = f"(YF: ₹{val['yf_price']} Diff: {val['diff_pct']}%)"
                
                # Calculate Basic Trade Plan
                stop_loss = round(price * 0.98, 2) # 2% SL
                target = round(price * 1.05, 2)    # 5% Target
                risk_reward = round((target - price) / (price - stop_loss), 2)
                
                msg += f"📊 **{ticker}** @ ₹{price} {accuracy_icon} {accuracy_note}\n"
                msg += f"🏢 **Sector:** {sector} | **PE:** {pe}\n"
                msg += f"✅ **Action:** BUY\n"
                msg += f"🧠 **Rationale:** {len(strategies)} Strategies Confirmed\n"
                msg += f"📅 **Hold:** 1-2 Weeks (Swing)\n"
                msg += f"📉 **Stop Loss:** ₹{stop_loss} (-2%)\n"
                msg += f"🎯 **Target:** ₹{target} (+5%)\n"
                msg += f"⚖️ **Risk/Reward:** 1:{risk_reward}\n"

                # PAPER TRADING LOGIC
                if self.mode == 'paper' and self.trader:
                    signal = {
                        "ticker": ticker,
                        "action": "BUY",
                        "price": price,
                        "sl": stop_loss,
                        "target": target,
                        "sentiment_score": res.get('sentiment_score', 0)
                    }
                    trade_res = self.trader.execute_trade(signal)
                    if trade_res:
                        msg += f"\n📝 **{trade_res}**"

                msg += "----------------------------\n"
            
            if self.mode == 'paper':
                msg += "\n📝 _Paper Trading Mode Active. Trades recorded automatically._"
            else:
                msg += "\n🔍 _Verification: Prices cross-checked with Yahoo Finance._"

            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=msg, parse_mode='Markdown')
            
        except Exception as e:
            logging.error(f"Error during scan: {e}")
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=f"❌ Error during scan: {str(e)}")

    async def scheduled_scan(self, context: ContextTypes.DEFAULT_TYPE):
        """Job Queue callback for daily scans"""
        await context.bot.send_message(chat_id=CHAT_ID, text="⏰ **Auto-Scan Triggered!**")
        
        # We can reuse the logic, but we need to mock an update/context or extract the logic.
        # Simpler to just run logic and send message directly.
        try:
            results = self.scanner.scan_market()
            if results:
                msg = "☀️ **Morning Scan Report**\n\n"
                for res in results[:5]:
                    msg += f"📊 **{res['ticker']}** @ ₹{res['price']} (Score: {res['score']})\n"
                await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='Markdown')
            else:
                await context.bot.send_message(chat_id=CHAT_ID, text="⚠️ No signals found this morning.")
        except Exception as e:
            await context.bot.send_message(chat_id=CHAT_ID, text=f"❌ Scheduled scan failed: {e}")

if __name__ == '__main__':
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found.")
        exit(1)

    # Check mode from environment
    mode = os.getenv('TRADESAGE_MODE', 'scan')
    trader = None
    
    if mode == 'paper':
        trader_file = os.getenv('TRADESAGE_TRADER_FILE', 'data/paper_portfolio.json')
        from src.paper.paper_trader import PaperTrader
        trader = PaperTrader(portfolio_file=trader_file)

    bot_logic = TradeSageBot(mode=mode, trader=trader)
    
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Handlers
    start_handler = CommandHandler('start', bot_logic.start)
    scan_handler = CommandHandler('scan', bot_logic.run_scan)
    status_handler = CommandHandler('status', bot_logic.status)
    portfolio_handler = CommandHandler('portfolio', bot_logic.portfolio)
    
    application.add_handler(start_handler)
    application.add_handler(scan_handler)
    application.add_handler(status_handler)
    application.add_handler(portfolio_handler)

    # Job Queue for Daily Scans
    job_queue = application.job_queue
    ist = pytz.timezone('Asia/Kolkata')
    
    # Schedule at 09:30 AM IST
    job_queue.run_daily(bot_logic.scheduled_scan, time(hour=9, minute=30, tzinfo=ist), chat_id=CHAT_ID)
    
    # Schedule at 03:00 PM IST (BTST)
    job_queue.run_daily(bot_logic.scheduled_scan, time(hour=15, minute=0, tzinfo=ist), chat_id=CHAT_ID)

    print(f"🤖 Bot is running in {mode.upper()} mode...")
    application.run_polling()
