import json
import csv
import urllib.request
import io
import os

def read_local_csv_symbols(filepath, symbol_col_index=2, series_col_index=None):
    """Reads local CSV and returns a list of symbols."""
    print(f"Reading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader) # Skip header
            
            symbols = []
            for row in csv_reader:
                if not row: continue
                
                # Filter by Series if column index provided (usually 'EQ')
                if series_col_index is not None:
                    if len(row) > series_col_index and row[series_col_index] != 'EQ':
                        continue
                
                if len(row) > symbol_col_index:
                    symbols.append(row[symbol_col_index])
                    
            print(f"  -> Found {len(symbols)} symbols.")
            return symbols
    except Exception as e:
        print(f"  -> Error reading: {e}")
        return []

def main():
    # 1. Start with Nifty 500 (Base Liquidity)
    try:
        with open("data/nifty500.json", "r") as f:
            universe = json.load(f)
            print(f"Loaded {len(universe)} from nifty500.json")
    except:
        print("Warning: nifty500.json not found, starting empty.")
        universe = []

    # 2. Add Nifty Microcap 250 (Next tier)
    microcap_symbols = read_local_csv_symbols("data/microcap.csv", symbol_col_index=2)
    
    # 3. Add Nifty Smallcap 250
    smallcap_symbols = read_local_csv_symbols("data/smallcap.csv", symbol_col_index=2)
    
    # 4. Add 'All Active Equity' (Filler)
    # EQUITY_L.csv: SYMBOL,NAME OF COMPANY, SERIES,... -> Symbol is 0, Series is 2
    all_equity_symbols = read_local_csv_symbols("data/equity_l.csv", symbol_col_index=0, series_col_index=2)

    # Combine ensuring uniqueness and priority
    # Priority: Nifty 500 > Microcap 250 > Smallcap 250 > Rest of Market
    
    final_list = []
    seen = set()

    # Helper to add
    def add_tickers(source_list):
        count = 0
        for sym in source_list:
            if sym not in seen:
                final_list.append(sym)
                seen.add(sym)
                count += 1
        return count

    print("\nCompiling Universe:")
    c1 = add_tickers(universe) # Nifty 500
    print(f"  + {c1} from Nifty 500")
    
    c2 = add_tickers(microcap_symbols)
    print(f"  + {c2} from Microcap 250")
    
    c3 = add_tickers(smallcap_symbols)
    print(f"  + {c3} from Smallcap 250")
    
    # Fill remaining spots up to 1200
    target = 1200
    current = len(final_list)
    needed = target - current
    
    if needed > 0:
        print(f"  Need {needed} more to reach {target}. Filling from All Equity list...")
        c4 = 0
        for sym in all_equity_symbols:
            if len(final_list) >= target:
                break
            if sym not in seen:
                final_list.append(sym)
                seen.add(sym)
                c4 += 1
        print(f"  + {c4} from All Equity List")
    
    print(f"\nFinal Universe Size: {len(final_list)}")
    
    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/nse_1200.json", "w") as f:
        json.dump(final_list, f, indent=2)
    print("Saved to data/nse_1200.json")

if __name__ == "__main__":
    main()
