import os
import json
import logging
from datetime import datetime
import pandas as pd

class PaperTrader:
    def __init__(self, portfolio_file="data/paper_portfolio.json", initial_capital=100000):
        self.portfolio_file = portfolio_file
        self.initial_capital = initial_capital
        self.data_dir = os.path.dirname(portfolio_file)
        
        # Setup logging
        self.logger = logging.getLogger("PaperTrader")
        
        # Ensure data dir exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Load or Init Portfolio
        self.portfolio = self.load_portfolio()

    def load_portfolio(self):
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "balance": self.initial_capital,
                "holdings": {}, # {ticker: {qty, avg_price, entry_date}}
                "history": [],  # List of closed trades
                "start_date": str(datetime.now().date())
            }

    def save_portfolio(self):
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=4)

    def execute_trade(self, signal: dict):
        """
        Simulates a trade execution.
        signal = {'ticker': 'RELIANCE', 'action': 'BUY', 'price': 2500, 'sl': 2400, 'target': 2600}
        """
        ticker = signal['ticker']
        price = signal['price']
        action = signal['action']
        date = str(datetime.now())

        if action == "BUY":
            # Position Sizing: Risk 2% of equity per trade
            risk_per_trade = self.portfolio['balance'] * 0.02
            stop_loss = signal.get('sl', price * 0.95)
            risk_per_share = price - stop_loss
            
            if risk_per_share <= 0:
                qty = 1 # Fallback
            else:
                qty = int(risk_per_trade / risk_per_share)
            
            # Cap max allocation to 10% of portfolio
            max_cost = self.portfolio['balance'] * 0.10
            if qty * price > max_cost:
                qty = int(max_cost / price)

            if qty < 1: qty = 1
            
            cost = qty * price
            
            if cost > self.portfolio['balance']:
                self.logger.warning(f"Insufficient funds for {ticker}")
                return

            # Update Portfolio
            self.portfolio['balance'] -= cost
            self.portfolio['holdings'][ticker] = {
                "qty": qty,
                "avg_price": price,
                "entry_date": date,
                "sl": stop_loss,
                "target": signal.get('target', price * 1.05)
            }
            self.save_portfolio()
            return f"✅ PAPER BUY: {qty} shares of {ticker} @ ₹{price}"

        elif action == "SELL":
            # For now, we only buy, but Sell logic would go here for Exits
            pass
            
    def update_portfolio(self, current_prices: dict):
        """
        Checks current prices against SL/Target to close positions.
        """
        messages = []
        for ticker, position in list(self.portfolio['holdings'].items()):
            if ticker in current_prices:
                price = current_prices[ticker]
                
                # Check Target
                if price >= position['target']:
                    pnl = (price - position['avg_price']) * position['qty']
                    self.portfolio['balance'] += (price * position['qty'])
                    del self.portfolio['holdings'][ticker]
                    self.portfolio['history'].append({
                        "ticker": ticker,
                        "pnl": pnl,
                        "exit_price": price,
                        "reason": "TARGET"
                    })
                    messages.append(f"🎯 TARGET HIT: {ticker} (+₹{pnl:.2f})")
                
                # Check Stop Loss
                elif price <= position['sl']:
                    pnl = (price - position['avg_price']) * position['qty']
                    self.portfolio['balance'] += (price * position['qty'])
                    del self.portfolio['holdings'][ticker]
                    self.portfolio['history'].append({
                        "ticker": ticker,
                        "pnl": pnl,
                        "exit_price": price,
                        "reason": "STOPLOSS"
                    })
                    messages.append(f"🛑 STOP HIT: {ticker} (₹{pnl:.2f})")
                    
        self.save_portfolio()
        return messages

    def get_summary(self):
        # Calculate Unrealized PnL
        warnings = [] # Placeholder
        return {
            "balance": self.portfolio['balance'],
            "open_positions": len(self.portfolio['holdings']),
            "closed_trades": len(self.portfolio['history'])
        }
