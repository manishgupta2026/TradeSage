import os
import json
import logging
from datetime import datetime
import pandas as pd

class PaperTrader:
    def __init__(self, portfolio_file="data/paper_portfolio.json", initial_capital=50000):
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
            # Check if already holding this ticker
            if ticker in self.portfolio['holdings']:
                return f"⚠️ Already holding {ticker}"
            
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
                return f"⚠️ Insufficient funds for {ticker}"

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
            # Manual sell (not used yet, but available)
            if ticker not in self.portfolio['holdings']:
                return f"⚠️ No position in {ticker}"
            
            position = self.portfolio['holdings'][ticker]
            pnl = (price - position['avg_price']) * position['qty']
            self.portfolio['balance'] += (price * position['qty'])
            
            del self.portfolio['holdings'][ticker]
            self.portfolio['history'].append({
                "ticker": ticker,
                "pnl": pnl,
                "exit_price": price,
                "exit_date": date,
                "reason": "MANUAL"
            })
            self.save_portfolio()
            return f"✅ PAPER SELL: {ticker} @ ₹{price} (P&L: ₹{pnl:.2f})"
            
    def update_portfolio(self, current_prices: dict):
        """
        Checks current prices against SL/Target to close positions.
        current_prices = {'RELIANCE': 2550, 'TCS': 3400, ...}
        """
        messages = []
        date = str(datetime.now())
        
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
                        "exit_date": date,
                        "reason": "TARGET"
                    })
                    messages.append(f"🎯 TARGET HIT: {ticker} @ ₹{price} (+₹{pnl:.2f})")
                
                # Check Stop Loss
                elif price <= position['sl']:
                    pnl = (price - position['avg_price']) * position['qty']
                    self.portfolio['balance'] += (price * position['qty'])
                    del self.portfolio['holdings'][ticker]
                    self.portfolio['history'].append({
                        "ticker": ticker,
                        "pnl": pnl,
                        "exit_price": price,
                        "exit_date": date,
                        "reason": "STOPLOSS"
                    })
                    messages.append(f"🛑 STOP HIT: {ticker} @ ₹{price} (₹{pnl:.2f})")
                    
        if messages:
            self.save_portfolio()
        return messages

    def get_summary(self, current_prices: dict = {}, prev_closes: dict = {}):
        """
        Calculates detailed portfolio metrics including Realized, Unrealized, and Day's P&L.
        Returns a dictionary with all metrics.
        """
        # 1. Calculate Holdings Value
        holdings_value = 0
        unrealized_pnl = 0
        todays_unrealized_change = 0
        detailed_holdings = []
        
        for ticker, pos in self.portfolio['holdings'].items():
            # Current Price (use avg_price as fallback if missing)
            price = current_prices.get(ticker, pos['avg_price'])
            qty = pos['qty']
            avg_price = pos['avg_price']
            
            # Value & Unrealized
            market_val = price * qty
            holdings_value += market_val
            unrealized_pnl += (price - avg_price) * qty
            
            # Today's Change (Unrealized part)
            # If we have prev_close, change is (Price - PrevClose). 
            # If new position today? fallback to entry price (avg_price) approx.
            prev_close = prev_closes.get(ticker, avg_price)
            todays_unrealized_change += (price - prev_close) * qty
            
            # Detailed Holding Stats
            pnl_val = (price - avg_price) * qty
            pnl_pct = ((price - avg_price) / avg_price) * 100
            
            detailed_holdings.append({
                "ticker": ticker,
                "qty": qty,
                "avg": avg_price,
                "cmp": price,
                "pnl": pnl_val,
                "pnl_pct": pnl_pct
            })

        # 2. Total Equity
        total_equity = self.portfolio['balance'] + holdings_value
        
        # 3. Total P&L
        total_pnl = total_equity - self.initial_capital
        roi = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # 4. Realized P&L (Total & Today)
        realized_pnl = sum([t['pnl'] for t in self.portfolio['history']])
        
        # Calculate Realized P&L for TODAY
        today_str = str(datetime.now().date())
        realized_today = 0
        for t in self.portfolio['history']:
            # t['exit_date'] is usually full timestamp "2024-02-10 14:30:00"
            if t['exit_date'].startswith(today_str):
                realized_today += t['pnl']

        # 5. Total Day's P&L
        todays_pnl = todays_unrealized_change + realized_today

        return {
            "equity": total_equity,
            "balance": self.portfolio['balance'],
            "holdings_val": holdings_value,
            "total_pnl": total_pnl,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "todays_pnl": todays_pnl,
            "roi": roi,
            "open_positions": len(self.portfolio['holdings']),
            "closed_trades": len(self.portfolio['history']),
            "holdings": detailed_holdings
        }
