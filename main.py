import argparse
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import after adding src to path
from src.paper.paper_trader import PaperTrader

def main():
    parser = argparse.ArgumentParser(description="TradeSage AI - Algo Trading Bot")
    parser.add_argument('--mode', type=str, choices=['scan', 'paper', 'live'], default='paper', help='Operation Mode')
    parser.add_argument('--init-capital', type=int, default=50000, help='Initial Capital for Paper Trading')
    
    args = parser.parse_args()
    
    load_dotenv()
    
    print(f"🚀 TradeSage AI Starting in [{args.mode.upper()}] Mode...")
    
    if args.mode == 'paper':
        print(f"💰 Paper Trading Initialized with ₹{args.init_capital}")
        trader = PaperTrader(initial_capital=args.init_capital)
        
        # Run bot script directly with environment variable
        os.environ['TRADESAGE_MODE'] = 'paper'
        os.environ['TRADESAGE_TRADER_FILE'] = trader.portfolio_file
        
        # Run the telegram bot with venv python
        import subprocess
        python_path = os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe')
        subprocess.run([python_path, 'src/bot/telegram_bot.py'])
        
    elif args.mode == 'live':
        print("⚠️ LIVE TRADING MODE (Not implemented yet)")
        os.environ['TRADESAGE_MODE'] = 'live'
        import subprocess
        python_path = os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe')
        subprocess.run([python_path, 'src/bot/telegram_bot.py'])
        
    elif args.mode == 'scan':
        print("🔍 Scan-Only Mode")
        os.environ['TRADESAGE_MODE'] = 'scan'
        import subprocess
        python_path = os.path.join(os.getcwd(), 'venv', 'Scripts', 'python.exe')
        subprocess.run([python_path, 'src/bot/telegram_bot.py'])

if __name__ == "__main__":
    main()
