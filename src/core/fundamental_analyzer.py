import yfinance as yf
import logging
from tradingview_ta import TA_Handler, Interval, Exchange

# NLP dependencies
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from transformers import pipeline
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

logger = logging.getLogger(__name__)

class FundamentalAnalyzer:
    """
    Extends purely technical ML models by filtering trading candidates through 
    Warren-Buffett style fundamentals and institutional News Sentiment.
    """
    
    def __init__(self):
        self.nlp = None
        if HAS_NLP:
            try:
                # Load FinBERT - specialized specifically in financial sentiment
                logger.info("loading FinBERT NLP model from HuggingFace (Takes ~2GB RAM)...")
                self.nlp = pipeline("text-classification", model="ProsusAI/finbert")
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {e}")
                HAS_NLP = False

    def fetch_yfinance_fundamentals(self, symbol: str):
        """Fetch strict fundamental ratios"""
        try:
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            info = stock.info
            
            return {
                'forwardPE': info.get('forwardPE'),
                'trailingPE': info.get('trailingPE'),
                'priceToBook': info.get('priceToBook'),
                'returnOnEquity': info.get('returnOnEquity'),
                'debtToEquity': info.get('debtToEquity'),
                'profitMargins': info.get('profitMargins'),
            }
        except Exception as e:
            logger.debug(f"Failed to fetch fundamentals for {symbol}: {e}")
            return {}

    def fetch_tradingview_rating(self, symbol: str):
        """Hit TradingView API for 26-indicator consensus rating"""
        try:
            handler = TA_Handler(
                symbol=symbol,
                exchange="NSE",
                screener="india",
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            # Returns 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
            return analysis.summary.get('RECOMMENDATION')
        except Exception as e:
            logger.debug(f"TradingView fetch failed for {symbol}: {e}")
            return "UNKNOWN"

    def analyze_news_sentiment(self, symbol: str):
        """Scrape latest 5 news headlines from Yahoo and pass through FinBERT"""
        if not self.nlp:
            return {'score': 0, 'label': 'NEUTRAL', 'confidence': 0}
            
        try:
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            news = stock.news
            
            if not news:
                return {'score': 0, 'label': 'NEUTRAL', 'confidence': 0}
                
            headlines = [article['title'] for article in news[:5]]
            results = self.nlp(headlines)
            
            # Map FinBERT labels to scores
            score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            total_score = 0
            
            for res in results:
                total_score += score_map.get(res['label'], 0) * res['score']
                
            avg_score = total_score / len(headlines)
            
            if avg_score > 0.2:
                final_label = 'POSITIVE'
            elif avg_score < -0.2:
                final_label = 'NEGATIVE'
            else:
                final_label = 'NEUTRAL'
                
            return {
                'score': avg_score,
                'label': final_label,
                'confidence': sum([r['score'] for r in results]) / len(results)
            }
        except Exception as e:
            logger.debug(f"Sentiment analysis failed for {symbol}: {e}")
            return {'score': 0, 'label': 'NEUTRAL', 'confidence': 0}

    def evaluate_candidate(self, symbol: str) -> bool:
        """
        The Master Filter. Returns True if the stock survives Fundamental scraping.
        """
        logger.info(f"   [Fundamental Scan] evaluating {symbol}...")
        
        fundamentals = self.fetch_yfinance_fundamentals(symbol)
        tv_rating = self.fetch_tradingview_rating(symbol)
        sentiment = self.analyze_news_sentiment(symbol)
        
        logger.info(f"      TV Rating: {tv_rating}")
        logger.info(f"      Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
        
        # 1. Reject if TradingView consensus is bearish
        if tv_rating in ['SELL', 'STRONG_SELL']:
            logger.info("      ❌ Rejected: TradingView consensus is SELL.")
            return False
            
        # 2. Reject if News Sentiment is severely negative
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] < -0.4:
            logger.info("      ❌ Rejected: Extremely negative news cycle.")
            return False
            
        # 3. Reject severely overvalued or unprofitable fundamental garbage
        trailing_pe = fundamentals.get('trailingPE')
        profit_margin = fundamentals.get('profitMargins')
        
        if trailing_pe is not None and trailing_pe < 0:
            logger.info("      ❌ Rejected: Company has negative P/E (Loss making).")
            return False
            
        if trailing_pe is not None and trailing_pe > 150:
             logger.info("      ❌ Rejected: Company is severely overvalued (P/E > 150).")
             return False
             
        if profit_margin is not None and profit_margin < -0.05:
            logger.info("      ❌ Rejected: Negative profit margins.")
            return False
            
        logger.info("      ✅ Passed Fundamental Screen!")
        return {
            "tv_rating": tv_rating if tv_rating else "N/A",
            "sentiment": sentiment['label'],
            "pe_ratio": round(trailing_pe, 1) if trailing_pe is not None else "N/A"
        }
