import pandas as pd
import json
import re
from typing import List, Dict, Any
from src.engine.indicators import IndicatorLibrary

class StrategyExecutor:
    def __init__(self, indicator_lib: IndicatorLibrary):
        self.lib = indicator_lib

    def load_strategy_from_json(self, file_path: str) -> List[Dict]:
        """Loads strategies from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading strategy file {file_path}: {e}")
            return []

    def _parse_condition(self, condition: str) -> Dict[str, Any]:
        """
        Parses a string condition like 'RSI < 30' or 'EMA_20 > EMA_50' into parts.
        """
        # Supported patterns: [Indicator] [Operator] [Value or Indicator]
        # Matches numbers (30, 0.5) or strings (EMA_50) on RHS
        pattern = r"([A-Za-z0-9_]+)\s*([<>=!]+)\s*([A-Za-z0-9_\.]+)"
        match = re.search(pattern, condition)
        if match:
            raw_value = match.group(3)
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value  # Keep as string (likely a column name)

            return {
                "indicator": match.group(1),
                "operator": match.group(2),
                "value": value
            }
        return None

    def _resolve_column(self, df: pd.DataFrame, col_name: str) -> str:
        """Helper to resolve fuzzy column names to actual DataFrame columns."""
        # 1. Direct Exact Match (Case Insensitive)
        for col in df.columns:
            if col_name.upper() == col.upper():
                return col

        # 2. Fuzzy Match
        for col in df.columns:
             if col_name.upper() in col.upper():
                 return col

        # 3. Specific Mappings
        if "STOCH" in col_name.upper() or "%K" in col_name.upper():
            return next((c for c in df.columns if "STOCHk" in c), None)
        elif "%D" in col_name.upper():
            return next((c for c in df.columns if "STOCHd" in c), None)
        elif "ATR" in col_name.upper():
            return next((c for c in df.columns if "ATRr" in c), None)
        elif "ADX" in col_name.upper():
            return next((c for c in df.columns if "ADX_" in c), None)
        elif "WILLIAMS" in col_name.upper() or "%R" in col_name.upper():
            return next((c for c in df.columns if "WILLR" in c), None)
        elif "VOLUME" in col_name.upper() and ("AVG" in col_name.upper() or "SMA" in col_name.upper()):
            return "VOL_SMA_20"
        elif "CLOSE" in col_name.upper():
             return next((c for c in df.columns if c.upper() == "CLOSE"), None)

        return None

    def evaluate_strategy(self, df: pd.DataFrame, strategy: Dict) -> pd.Series:
        """
        Evaluates a strategy against a dataframe.
        Returns a boolean series where True means buy/entry signal.
        """
        if df.empty:
            return pd.Series([False] * len(df))

        entry_conditions = strategy.get("entry_conditions", [])
        signals = pd.Series([True] * len(df), index=df.index)

        found_any_valid_condition = False
        for cond_str in entry_conditions:
            parsed = self._parse_condition(cond_str)
            if parsed:
                # Resolve LHS Column
                matched_lhs = self._resolve_column(df, parsed["indicator"])
                
                # Resolve RHS Value (if it's a string, try to find a column)
                rhs_value = parsed["value"]
                if isinstance(rhs_value, str):
                    matched_rhs = self._resolve_column(df, rhs_value)
                    if matched_rhs:
                        rhs_value = matched_rhs  # Pass the column name
                    # If not matched, it stays as a string

                if matched_lhs:
                    cond_signal = self.lib.check_condition(
                        df, matched_lhs, parsed["operator"], rhs_value
                    )
                    signals = signals & cond_signal
                    found_any_valid_condition = True
                else:
                    if parsed["indicator"] != "RRR":
                        # print(f"Warning: Indicator '{parsed['indicator']}' not found in DataFrame.")
                        pass # Squelch warnings

        return signals if found_any_valid_condition else pd.Series([False] * len(df), index=df.index)

    def generate_signals(self, df: pd.DataFrame, strategies: List[Dict]) -> pd.DataFrame:
        """Generates signals for all strategies in the list."""
        signal_df = pd.DataFrame(index=df.index)
        
        for strategy in strategies:
            name = strategy.get("strategy_name", "Unknown Strategy")
            signal_df[name] = self.evaluate_strategy(df, strategy)
            
        return signal_df
