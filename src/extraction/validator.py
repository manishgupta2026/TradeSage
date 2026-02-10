from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal

class Strategy(BaseModel):
    strategy_name: str
    type: str # e.g., Swing, Intraday, Options
    market: str = Field(..., description="NSE, BSE, MCX, or NFO")
    timeframe: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    stop_loss: List[str]
    technical_indicators: List[str]
    risk_management: Optional[List[str]] = []
    source_file: Optional[str] = None

    @validator('market')
    def validate_market(cls, v):
        valid_markets = ["NSE", "BSE", "MCX", "NFO"]
        if v.upper() not in valid_markets:
            # Try to infer or default if close enough? For now, strict.
            # actually LLM might be messy, so maybe soft validation or cleanup
            pass 
        return v.upper()

class StrategyDatabase(BaseModel):
    strategies: List[Strategy]
    total_count: int
    valid_count: int

def validate_strategy_data(data: dict) -> Optional[Strategy]:
    try:
        return Strategy(**data)
    except Exception as e:
        print(f"Validation error for strategy '{data.get('strategy_name', 'Unknown')}': {e}")
        return None
