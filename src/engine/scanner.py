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

        # New Step: Filter out invalid strategies
        valid_strategies = []
        for strategy in strategies:
            entry_conditions = strategy.get("entry_conditions", [])
            has_valid_condition = False
            for cond in entry_conditions:
                if self.executor._parse_condition(cond):
                    has_valid_condition = True
                    break
            if has_valid_condition:
                valid_strategies.append(strategy)

        print(f"Filtered {len(strategies)} strategies down to {len(valid_strategies)} executable strategies.")
        return valid_strategies

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
                
                # Count True signals with weighted scoring
                active_strats = latest_signals[latest_signals == True].index.tolist()
                
                # Weighted Scoring: Prioritize momentum/volatility strategies
                weighted_score = 0
                for strat in active_strats:
                    strat_lower = strat.lower()
                    if any(keyword in strat_lower for keyword in ['stoch', 'rsi', 'macd', 'williams']):
                        weighted_score += 1.5  # Momentum indicators
                    elif any(keyword in strat_lower for keyword in ['atr', 'bollinger']):
                        weighted_score += 1.2  # Volatility indicators
                    elif any(keyword in strat_lower for keyword in ['volume', 'obv']):
                        weighted_score += 1.0  # Volume indicators
                    else:
                        weighted_score += 1.0  # Default weight
                
                score = len(active_strats)
                
                if score > 0:
                    # Multi-Timeframe Confirmation: Check Weekly trend
                    try:
                        weekly_df = self.dm.fetch_data(ticker, interval="1wk", use_cache=False)
                        if not weekly_df.empty and len(weekly_df) >= 50:
                            weekly_df = self.lib.add_standard_indicators(weekly_df)
                            weekly_ema50 = weekly_df.iloc[-1].get('EMA_50', 0)
                            weekly_ema200 = weekly_df.iloc[-1].get('EMA_200', 0)
                            
                            if weekly_ema50 < weekly_ema200:
                                print(f"⏭️  {ticker} skipped: Weekly downtrend (EMA50 < EMA200)")
                                continue  # Skip if weekly trend is down
                    except Exception as e:
                        print(f"Warning: Could not fetch weekly data for {ticker}: {e}")
                        # Continue anyway if weekly data fails
                    
                    results.append({
                        "ticker": ticker,
                        "price": round(current_price, 2),
                        "score": score,
                        "weighted_score": round(weighted_score, 2),
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
            sentiment_threshold = float(os.getenv("SENTIMENT_THRESHOLD", "-0.3"))
            print(f"🧠 AI Sentiment Analysis: ENABLED (Threshold: {sentiment_threshold})")
        except Exception as e:
            print(f"⚠️ AI Sentiment Analysis: DISABLED ({e})")
            use_sentiment = False

        for res in results:
            if res['score_pct'] >= 50: # Minimum technical score
                
                # AI Sentiment Check
                if use_sentiment:
                    print(f"Analyzing sentiment for {res['ticker']}...")
                    s_data = sentiment_analyzer.analyze_sentiment(res['ticker'])
                    res['sentiment_score'] = s_data.get('score', 0)
                    res['sentiment_reason'] = s_data.get('reason', 'N/A')
                    
                    if res['sentiment_score'] < sentiment_threshold:
                        print(f"❌ Skipped {res['ticker']}: Negative News Sentiment ({res['sentiment_score']})")
                        continue
                    else:
                        print(f"✅ {res['ticker']} Sentiment: {res['sentiment_score']} ({res['sentiment_reason']})")
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
