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
        """
        Hit TradingView API for 26-indicator consensus rating.
        Returns a dict with recommendation, buy/sell/neutral counts, and a weighted score.
        """
        try:
            handler = TA_Handler(
                symbol=symbol,
                exchange="NSE",
                screener="india",
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            summary = analysis.summary
            
            recommendation = summary.get('RECOMMENDATION', 'UNKNOWN')
            buy_count = summary.get('BUY', 0)
            sell_count = summary.get('SELL', 0)
            neutral_count = summary.get('NEUTRAL', 0)
            total = buy_count + sell_count + neutral_count
            
            # Weighted score: -1.0 (strong sell) to +1.0 (strong buy)
            if total > 0:
                weighted_score = round((buy_count - sell_count) / total, 3)
            else:
                weighted_score = 0.0
            
            return {
                'recommendation': recommendation,
                'buy': buy_count,
                'sell': sell_count,
                'neutral': neutral_count,
                'score': weighted_score,  # -1.0 to +1.0
            }
        except Exception as e:
            logger.debug(f"TradingView fetch failed for {symbol}: {e}")
            return {
                'recommendation': 'UNKNOWN',
                'buy': 0, 'sell': 0, 'neutral': 0,
                'score': 0.0,
            }

    def analyze_news_sentiment(self, symbol: str):
        """Scrape latest 5 news headlines from Yahoo and pass through FinBERT"""
        if not self.nlp:
            return {'score': 0, 'label': 'NO_NLP', 'confidence': 0, 'headlines_found': 0}
            
        try:
            yf_symbol = f"{symbol}.NS"
            stock = yf.Ticker(yf_symbol)
            news = stock.news
            
            if not news:
                # No news = NO_DATA, NOT neutral (important distinction)
                return {'score': 0, 'label': 'NO_DATA', 'confidence': 0, 'headlines_found': 0}
                
            headlines = [article.get('content', {}).get('title', article.get('title', '')) for article in news[:5]]
            headlines = [h for h in headlines if h]  # Filter out empty ones
            
            if not headlines:
                return {'score': 0, 'label': 'NO_DATA', 'confidence': 0, 'headlines_found': 0}
            
            logger.debug(f"Sentiment headlines for {symbol}: {headlines}")
            results = self.nlp(headlines)
            
            # Map FinBERT labels to scores
            score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            total_score = 0
            
            for res in results:
                total_score += score_map.get(res['label'], 0) * res['score']
                
            avg_score = total_score / len(headlines)
            avg_confidence = sum([r['score'] for r in results]) / len(results)
            
            # Tighter thresholds: ±0.1 instead of ±0.2 (anti-neutral bias)
            if avg_score > 0.1:
                final_label = 'POSITIVE'
            elif avg_score < -0.1:
                final_label = 'NEGATIVE'
            else:
                final_label = 'NEUTRAL'
                
            return {
                'score': round(avg_score, 4),
                'label': final_label,
                'confidence': round(avg_confidence, 4),
                'headlines_found': len(headlines),
            }
        except Exception as e:
            logger.debug(f"Sentiment analysis failed for {symbol}: {e}")
            return {'score': 0, 'label': 'NO_DATA', 'confidence': 0, 'headlines_found': 0}

    def compute_composite_score(self, tv_data: dict, sentiment_data: dict, fundamentals: dict) -> dict:
        """
        Produce a single 0-100 conviction score combining:
        - TradingView Rating (40%)
        - News Sentiment (30%)
        - Fundamental Health (30%)
        """
        # TV Score: convert -1..+1 to 0..100
        tv_score = (tv_data.get('score', 0) + 1) * 50  # 0 to 100
        
        # Sentiment Score: convert -1..+1 to 0..100
        sent_raw = sentiment_data.get('score', 0)
        if sentiment_data.get('label') in ('NO_DATA', 'NO_NLP'):
            sent_score = 50  # Neutral default when no data
        else:
            sent_score = (sent_raw + 1) * 50
        
        # Fundamental Score (basic health check)
        fund_score = 50  # Default neutral
        pe = fundamentals.get('pe_ratio')
        roe = fundamentals.get('roe')
        debt = fundamentals.get('debt_to_equity')
        
        if pe is not None:
            if 0 < pe <= 25:
                fund_score += 15
            elif 25 < pe <= 50:
                fund_score += 5
            elif pe > 100:
                fund_score -= 15
        
        if roe is not None:
            if roe > 15:
                fund_score += 15
            elif roe > 10:
                fund_score += 5
        
        if debt is not None:
            if debt < 0.5:
                fund_score += 10
            elif debt > 2.0:
                fund_score -= 15
        
        fund_score = max(0, min(100, fund_score))
        
        # Weighted composite
        composite = round(tv_score * 0.4 + sent_score * 0.3 + fund_score * 0.3, 1)
        composite = max(0, min(100, composite))
        
        return {
            'composite_score': composite,
            'tv_component': round(tv_score, 1),
            'sentiment_component': round(sent_score, 1),
            'fundamental_component': round(fund_score, 1),
        }

    def evaluate_candidate(self, symbol: str):
        """
        The Master Filter. Returns a dict with enriched data if the stock survives,
        or False if it should be rejected.
        """
        logger.info(f"   [Fundamental Scan] evaluating {symbol}...")
        
        fundamentals = self.fetch_fundamentals(symbol)
        tv_data = self.fetch_tradingview_rating(symbol)
        sentiment = self.analyze_news_sentiment(symbol)
        
        tv_rating = tv_data.get('recommendation', 'UNKNOWN')
        
        logger.info(f"      TV Rating: {tv_rating} (B:{tv_data.get('buy',0)} S:{tv_data.get('sell',0)} N:{tv_data.get('neutral',0)} score:{tv_data.get('score',0)})")
        logger.info(f"      Sentiment: {sentiment['label']} (score:{sentiment['score']:.3f}, headlines:{sentiment.get('headlines_found',0)})")
        
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
        
        # Compute composite conviction score
        composite = self.compute_composite_score(tv_data, sentiment, fundamentals)
            
        logger.info(f"      ✅ Passed Fundamental Screen! Conviction: {composite['composite_score']}/100")
        return {
            "tv_rating": tv_rating if tv_rating else "N/A",
            "tv_score": tv_data.get('score', 0),
            "tv_buy": tv_data.get('buy', 0),
            "tv_sell": tv_data.get('sell', 0),
            "sentiment": sentiment['label'],
            "sentiment_score": sentiment.get('score', 0),
            "sentiment_confidence": sentiment.get('confidence', 0),
            "pe_ratio": round(pe_ratio, 1) if pe_ratio is not None else "N/A",
            "conviction_score": composite['composite_score'],
        }
