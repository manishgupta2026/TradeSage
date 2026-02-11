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
        Parses a string condition like 'RSI < 30' into parts.
        This is a basic parser and might need refinement for complex rules.
        """
        # Supported patterns: [Indicator] [Operator] [Value]
        pattern = r"([A-Za-z0-9_]+)\s*([<>=!]+)\s*(\d+\.?\d*)"
        match = re.search(pattern, condition)
        if match:
            return {
                "indicator": match.group(1),
                "operator": match.group(2),
                "value": float(match.group(3))
            }
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
                # Map common names to column names if necessary
                # e.g., 'RSI' -> 'RSI_14'
                col_name = parsed["indicator"]
                
                # Enhanced Mapping for new indicators
                matched_col = None
                
                # 1. Direct Fuzzy Match
                for col in df.columns:
                    if col_name.upper() in col.upper():
                        matched_col = col
                        break
                
                # 2. Specific Mappings (if fuzzy fails or is ambiguous)
                if not matched_col:
                    if "STOCH" in col_name.upper() or "%K" in col_name.upper():
                        # Find the %K column (usually starts with STOCHk)
                        matched_col = next((c for c in df.columns if "STOCHk" in c), None)
                    elif "%D" in col_name.upper():
                        # Find the %D column (usually starts with STOCHd)
                        matched_col = next((c for c in df.columns if "STOCHd" in c), None)
                    elif "ATR" in col_name.upper():
                        matched_col = next((c for c in df.columns if "ATRr" in c), None)
                    elif "ADX" in col_name.upper():
                        matched_col = next((c for c in df.columns if "ADX_" in c), None)
                    elif "WILLIAMS" in col_name.upper() or "%R" in col_name.upper():
                        matched_col = next((c for c in df.columns if "WILLR" in c), None)
                    elif "VOLUME" in col_name.upper() and ("AVG" in col_name.upper() or "SMA" in col_name.upper()):
                        matched_col = "VOL_SMA_20"
                
                if matched_col:
                    cond_signal = self.lib.check_condition(
                        df, matched_col, parsed["operator"], parsed["value"]
                    )
                    signals = signals & cond_signal
                    found_any_valid_condition = True
                else:
                    # RRR (Risk Reward Ratio) is a planning metric, not a historical indicator.
                    # We can safely ignore this warning for now.
                    if col_name != "RRR":
                        print(f"Warning: Indicator '{col_name}' not found in DataFrame.")

        return signals if found_any_valid_condition else pd.Series([False] * len(df), index=df.index)

    def generate_signals(self, df: pd.DataFrame, strategies: List[Dict]) -> pd.DataFrame:
        """Generates signals for all strategies in the list."""
        signal_df = pd.DataFrame(index=df.index)
        
        for strategy in strategies:
            name = strategy.get("strategy_name", "Unknown Strategy")
            signal_df[name] = self.evaluate_strategy(df, strategy)
            
        return signal_df
