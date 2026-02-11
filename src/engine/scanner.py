from src.engine.data_manager import DataManager
from src.engine.indicators import IndicatorLibrary
from src.engine.strategy_executor import StrategyExecutor
import pandas as pd
import glob
import os

class NSEScanner:
    def __init__(self, data_manager=None):
        self.dm = data_manager if data_manager is not None else DataManager()
        self.lib = IndicatorLibrary()
        self.executor = StrategyExecutor(self.lib)
        # Broad list of Nifty 50 + Midcap stocks
        try:
            with open("data/nse_1200.json", "r") as f:
                import json
                self.tickers = json.load(f)
            print(f"Loaded {len(self.tickers)} NSE stocks (1200 Universe) for scanning.")
        except Exception as e:
            print(f"Warning: Could not load nse_1200.json ({e}), falling back to Nifty 50.")
            self.tickers = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "ASIANPAINT",
                "KOTAKBANK", "LT", "AXISBANK", "HCLTECH", "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "NESTLEIND",
                "WIPRO", "ONGC", "NTPC", "POWERGRID", "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS", "GRASIM", "COALINDIA",
                "BAJAJFINSV", "TECHM", "HINDALCO", "DIVISLAB", "CIPLA", "EICHERMOT", "BPCL", "TATAMOTORS", "DRREDDY", "HEROMOTOCO",
                "UPL", "APOLLOHOSP", "SBILIFE", "BRITANNIA", "INDUSINDBK", "BAJAJ-AUTO", "TATACONSUM", "M&M", "HDFCLIFE", "LTIM"
            ]

    def load_all_strategies(self):
        strategies = []
        files = glob.glob("data/strategies/*.json")
        for f in files:
            strategies.extend(self.executor.load_strategy_from_json(f))
        return strategies

    def scan_market(self):
        print(f"Starting NSE Scanner for {len(self.tickers)} stocks (Full Market)...")
        all_strategies = self.load_all_strategies()
        total_strategies = len(all_strategies)
        print(f"Loaded {total_strategies} strategies.")

        results = []

        for ticker in self.tickers:
            print(f"Scanning {ticker}...", end="\r")
            try:
                # Updated for new DataManager (yfinance)
                # Force REAL-TIME check (no cache)
                df = self.dm.fetch_data(ticker, exchange="NSE", use_cache=False)
                if df.empty: continue

                current_price = float(df.iloc[-1]['Close'])
                
                # Filter: Removed price cap as requested
                
                df = self.lib.add_standard_indicators(df)
                signals = self.executor.generate_signals(df, all_strategies)
                
                # Check latest signals (last row)
                latest_signals = signals.iloc[-1]
                
                # Count True signals
                active_strats = latest_signals[latest_signals == True].index.tolist()
                score = len(active_strats)
                
                if score > 0:
                    results.append({
                        "ticker": ticker,
                        "price": round(current_price, 2),
                        "score": score,
                        "score_pct": round((score / total_strategies) * 100, 1),
                        "active_strategies": active_strats
                    })
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")

        # Filter Results
        final_results = []
        
        # Initialize Sentiment Analyzer
        try:
            from src.analysis.sentiment import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer()
            use_sentiment = True
            print("🧠 AI Sentiment Analysis: ENABLED")
        except Exception as e:
            print(f"⚠️ AI Sentiment Analysis: DISABLED ({e})")
            use_sentiment = False

        for res in results:
            if res['score_pct'] >= 50: # Minimum technical score
                
                # AI Sentiment Check
                if use_sentiment:
                    print(f"Analyzing sentiment for {res['ticker']}...")
                    s_data = sentiment_analyzer.analyze_sentiment(res['ticker'])
                    res['sentiment_score'] = s_data['score']
                    res['sentiment_reason'] = s_data['reason']
                    
                    if s_data['score'] < -0.3:
                        print(f"❌ Skipped {res['ticker']}: Negative News Sentiment ({s_data['score']})")
                        continue
                    else:
                        print(f"✅ {res['ticker']} Sentiment: {s_data['score']} ({s_data['reason']})")
                else:
                     res['sentiment_score'] = 0
                     res['sentiment_reason'] = "N/A"

                final_results.append(res)
        
        # Sort by Technical Score then Sentiment
        final_results.sort(key=lambda x: (x['score_pct'], x.get('sentiment_score', 0)), reverse=True)
        
        print("\nScan Complete.")
        return final_results

if __name__ == "__main__":
    scanner = NSEScanner()
    scan_results = scanner.scan_market()
    
    if scan_results:
        print("\n--- TOP PICK SCORES (< ₹1000) ---")
        for res in scan_results[:5]: # Show top 5
            print(f"STOCK: {res['ticker']} | Price: ₹{res['price']} | Score: {res['score']}/{len(scanner.load_all_strategies())} ({res['score_pct']}%)")
            print(f"  Top Strategies: {', '.join(res['active_strategies'][:3])}...")
    else:
        print("\nNo immediate buy signals found for the selected stocks.")
