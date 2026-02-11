
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.indicators import IndicatorLibrary
from src.engine.strategy_executor import StrategyExecutor

def test_indicators():
    print("🧪 Testing Indicator Library Enhancement...")
    
    # Create Dummy Data (100 days)
    data = {
        'Open': np.random.normal(100, 5, 100),
        'High': np.random.normal(105, 5, 100),
        'Low': np.random.normal(95, 5, 100),
        'Close': np.random.normal(100, 5, 100),
        'Volume': np.random.randint(1000, 5000, 100)
    }
    df = pd.DataFrame(data)
    
    # Run Indicators
    print("   Adding indicators...", end="")
    df = IndicatorLibrary.add_standard_indicators(df)
    print(" Done.")
    
    # Check for New Columns
    expected_cols = ["ADX", "STOCHk", "STOCHd", "ATRr", "WILLR", "VOL_SMA_20"]
    missing = []
    
    print("\n   Checking new columns:")
    found_cols = df.columns.tolist()
    
    for expected in expected_cols:
        # Fuzzy check because pandas_ta adds params to names (e.g., ADX_14)
        match = any(expected in col for col in found_cols)
        status = "✅" if match else "❌"
        print(f"   - {expected}: {status} {[c for c in found_cols if expected in c]}")
        if not match:
            missing.append(expected)
            
    if missing:
        print(f"\n❌ FAILED. Missing columns: {missing}")
        return False
        
    print("\n✅ Indicator Calculation Success!")
    
    # Test Executor Mapping
    print("\n🧪 Testing Strategy Executor Mapping...")
    executor = StrategyExecutor(IndicatorLibrary())
    
    test_cases = [
        {"cond": "RSI > 50", "expected": True}, # Standard
        {"cond": "ADX > 20", "expected": True}, # New
        {"cond": "STOCH %K > 50", "expected": True}, # New Mapping
        {"cond": "Williams %R < -20", "expected": True} # New Mapping
    ]
    
    # Mock One Strategy
    strategy = {
        "strategy_name": "Test Strat",
        "entry_conditions": [c["cond"] for c in test_cases]
    }
    
    # We just want to see if it parses without crashing and finds the columns
    print("   Evaluating test strategy...")
    try:
        results = executor.evaluate_strategy(df, strategy)
        print(f"   Result Length: {len(results)}")
        print("✅ Executor Evaluation Success!")
        return True
    except Exception as e:
        print(f"❌ Executor Failed: {e}")
        return False

if __name__ == "__main__":
    if test_indicators():
        print("\n🚀 All Tests Passed. Strategy Engine Expanded.")
        sys.exit(0)
    else:
        print("\n💥 Tests Failed.")
        sys.exit(1)
