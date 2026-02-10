import math

class RiskManager:
    def __init__(self, portfolio_value=100000, risk_per_trade_pct=1.0):
        self.portfolio_value = portfolio_value
        self.risk_per_trade_pct = risk_per_trade_pct

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> dict:
        """
        Calculates how many shares to buy based on risk.
        Risk = Portfolio Value * Risk%
        Quantity = Risk / (Entry - StopLoss)
        """
        if entry_price <= stop_loss:
            return {"error": "Entry price must be above stop loss for a long trade."}

        risk_amount = self.portfolio_value * (self.risk_per_trade_pct / 100)
        risk_per_share = entry_price - stop_loss
        
        quantity = math.floor(risk_amount / risk_per_share)
        total_value = quantity * entry_price
        
        return {
            "risk_amount": round(risk_amount, 2),
            "quantity": quantity,
            "total_value": round(total_value, 2),
            "portfolio_pct": round((total_value / self.portfolio_value) * 100, 2)
        }

    def suggest_stop_loss(self, entry_price: float, atr: float = None, pct: float = 2.0) -> float:
        """Suggests a stop loss based on ATR or fixed percentage."""
        if atr:
            return entry_price - (2 * atr)
        return entry_price * (1 - pct / 100)
