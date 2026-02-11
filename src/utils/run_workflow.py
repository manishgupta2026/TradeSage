
import os
import sys
from telegram import Bot
import asyncio
from src.paper.paper_trader import PaperTrader
from src.engine.data_manager import DataManager
from src.engine.scanner import NSEScanner

async def main():
    try:
        # Initialize Components
        print("Initializing Trading Bot Components...")
        trader = PaperTrader(portfolio_file='data/paper_portfolio.json', initial_capital=50000)
        dm = DataManager()
        scanner = NSEScanner(dm)
        
        # Get Telegram credentials
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Check if Telegram is configured
        telegram_enabled = bool(telegram_token and telegram_token.strip() and chat_id and chat_id.strip())
        
        if telegram_enabled:
            bot = Bot(token=telegram_token)
        else:
            print('⚠️  Telegram bot token not configured. Running without notifications.')
        
        # Check for exits on existing positions first
        # Update Portfolio (Paper Trading)
        exit_msgs = []
        if trader:
            print('Updating portfolio...')
            current_prices = {}
            prev_closes = {}
            
            # Fetch live data for held stocks
            for ticker, holding_data in trader.portfolio['holdings'].items():
                try:
                    df = dm.fetch_data(ticker, use_cache=False)
                    if not df.empty:
                        current_prices[ticker] = df.iloc[-1]['Close']
                        if len(df) > 1:
                            prev_closes[ticker] = df.iloc[-2]['Close']
                        else:
                            prev_closes[ticker] = holding_data['avg_price']
                    else:
                        # Fallback if fetch fails
                        print(f"WARNING: Could not fetch data for {ticker}, using avg_price.")
                        current_prices[ticker] = holding_data['avg_price']
                        prev_closes[ticker] = holding_data['avg_price']
                except Exception as e:
                    print(f"Error updating {ticker}: {e}")
                    current_prices[ticker] = holding_data['avg_price']
                    prev_closes[ticker] = holding_data['avg_price']
                    
            exit_msgs = trader.update_portfolio(current_prices)
        
        # Scan market for new opportunities
        results = scanner.scan_market()
        
        msg = '🤖 **GitHub Actions Scan**\n\n'
        
        # Report exits first
        if exit_msgs:
            msg += '**📈 Position Exits:**\n'
            for exit_msg in exit_msgs:
                msg += f'{exit_msg}\n'
            msg += '\n'
        
        # Report new signals
        if results:
            msg += '**🔍 New Signals:**\n'
            for res in results[:3]:
                ticker = res['ticker']
                price = res['price']
                strategies = ', '.join(res['active_strategies'])
                signal = {
                    'ticker': ticker,
                    'price': price,
                    'strategies': strategies,
                    'stop_loss': price * 0.95,
                    'target': price * 1.10
                }
                trade_msg = trader.execute_trade(signal)
                if trade_msg:
                    msg += f'📊 {ticker} @ ₹{price}\n{trade_msg}\n'
                    # Add Sentiment Info if available
                    if res.get('sentiment_score'):
                            s_score = res['sentiment_score']
                            s_emoji = '🤖' if s_score > 0 else '⚠️'
                            msg += f'{s_emoji} **Sentiment:** {s_score} ({res["sentiment_reason"]})\n'
                    msg += '\n'
        
        # Portfolio Summary
        if trader:
             summary = trader.get_summary(current_prices, prev_closes)
             total_pnl_emoji = '🟢' if summary['total_pnl'] >= 0 else '🔴'
             day_pnl_emoji = '🟢' if summary['todays_pnl'] >= 0 else '🔴'
             
             msg += '────────────────\n'
             msg += '💼 **Portfolio Summary**\n'
             msg += f'💰 **Equity:** ₹{summary["equity"]:,.2f}\n'
             msg += f'💵 **Cash:** ₹{summary["balance"]:,.2f}\n'
             msg += f'{day_pnl_emoji} **Today:** ₹{summary["todays_pnl"]:,.2f}\n'
             msg += f'{total_pnl_emoji} **Total P&L:** ₹{summary["total_pnl"]:,.2f} ({summary["roi"]:.2f}%)\n'
             msg += f'📊 **Trades:** {summary["open_positions"]} Open | {summary["closed_trades"]} Closed\n'
             
             # List Holdings
             if summary['holdings']:
                 msg += '\n📂 **Holdings:**\n'
                 for h in summary['holdings']:
                     pnl_emoji = '🟢' if h['pnl'] >= 0 else '🔴'
                     msg += f'🔹 **{h["ticker"]}** ({h["qty"]}):\n'
                     msg += f'   Avg: ₹{h["avg"]:,.2f} ➝ CMP: ₹{h["cmp"]:,.2f}\n'
                     msg += f'   {pnl_emoji} P&L: ₹{h["pnl"]:,.2f} ({h["pnl_pct"]:.2f}%)\n'

        print(msg)
        
        if telegram_enabled and (results or exit_msgs or (trader and trader.portfolio['holdings'])):
             await bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
        else:
             print("No significant updates to send.")

    except Exception as e:
        print(f"Error in execution: {e}")
        # Notify user of failure
        if 'bot' in locals() and 'chat_id' in locals() and telegram_enabled:
            await bot.send_message(chat_id=chat_id, text=f"⚠️ Bot Execution Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
