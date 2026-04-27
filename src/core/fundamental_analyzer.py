import yfinance as yf
import logging
from tradingview_ta import TA_Handler, Interval, Exchange
from src.core.screener_scraper import ScreenerScraper

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
                
        self.screener = ScreenerScraper()

    def fetch_fundamentals(self, symbol: str):
        """Fetch strict fundamental ratios using Screener.in with yfinance fallback"""
        # Try Screener first
        try:
            scr_data = self.screener.fetch_fundamentals(symbol)
            if scr_data and len(scr_data) > 0:
                logger.debug(f"Fetched fundamentals from Screener for {symbol}")
                return scr_data
        except Exception as e:
            logger.debug(f"Screener fetch failed for {symbol}: {e}")
            
        # Fallback to yfinance
        try:
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            info = stock.info
            
            logger.debug(f"Fetched fundamentals from yfinance fallback for {symbol}")
            return {
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'priceToBook': info.get('priceToBook'),
                'roe': info.get('returnOnEquity') * 100 if info.get('returnOnEquity') else None,
                'debt_to_equity': info.get('debtToEquity'),
                'profit_margins': info.get('profitMargins'),
            }
        except Exception as e:
            logger.debug(f"Failed to fetch fundamentals for {symbol} on fallback: {e}")
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
        
        fundamentals = self.fetch_fundamentals(symbol)
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
            
        # 3. Reject based on stringent fundamental rules
        pe_ratio = fundamentals.get('pe_ratio')
        debt_to_equity = fundamentals.get('debt_to_equity')
        promoter_holding = fundamentals.get('promoter_holding')
        profit_margins = fundamentals.get('profit_margins')
        
        if pe_ratio is not None and pe_ratio < 0:
            logger.info("      ❌ Rejected: Company has negative P/E (Loss making).")
            return False
            
        if pe_ratio is not None and pe_ratio > 150:
             logger.info(f"      ❌ Rejected: Company is severely overvalued (P/E {pe_ratio} > 150).")
             return False
             
        if debt_to_equity is not None and debt_to_equity > 2.0:
            logger.info(f"      ❌ Rejected: High debt to equity ratio ({debt_to_equity:.2f} > 2.0).")
            return False
            
        if promoter_holding is not None and promoter_holding < 25.0:
            logger.info(f"      ❌ Rejected: Low promoter holding ({promoter_holding}% < 25%). Poor skin in the game.")
            return False
            
        if profit_margins is not None and profit_margins < -0.05:
            logger.info("      ❌ Rejected: Negative profit margins.")
            return False
            
        logger.info("      ✅ Passed Fundamental Screen!")
        return {
            "tv_rating": tv_rating if tv_rating else "N/A",
            "sentiment": sentiment['label'],
            "pe_ratio": round(pe_ratio, 1) if pe_ratio is not None else "N/A"
        }
