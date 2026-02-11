
import glob
import json
import os

def list_strategies():
    strategy_files = glob.glob("data/strategies/*.json")
    all_strategies = []
    
    print(f"📂 Found {len(strategy_files)} Strategy Files.")
    
    for f in strategy_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                # Handle both list of strategies and single strategy object
                if isinstance(data, list):
                    for s in data:
                        if s.get('strategy_name'):
                            all_strategies.append(s['strategy_name'])
                elif isinstance(data, dict):
                    if data.get('strategy_name'):
                        all_strategies.append(data['strategy_name'])
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")

    print(f"\n✅ Total Strategies Loaded: {len(all_strategies)}")
    print("-" * 50)
    
    # Sort and Print
    all_strategies.sort()
    for i, name in enumerate(all_strategies, 1):
        print(f"{i}. {name}")

if __name__ == "__main__":
    list_strategies()
