import feedparser
import json
import re
from src.llm.engine import LLMEngine

class SentimentAnalyzer:
    def __init__(self):
        self.llm = LLMEngine()
        
    def fetch_news(self, ticker):
        """Fetches latest news from Google News RSS for a given ticker."""
        # Query format: "TATASTEEL stock news India"
        query = f"{ticker} stock news India"
        formatted_query = query.replace(" ", "%20")
        url = f"https://news.google.com/rss/search?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            feed = feedparser.parse(url)
            headlines = []
            # Get top 5 unique headlines
            seen = set()
            for entry in feed.entries[:7]:
                title = entry.title
                # Clean generic sources appending like " - Moneycontrol"
                clean_title = title.split(" - ")[0]
                if clean_title not in seen:
                    headlines.append(clean_title)
                    seen.add(clean_title)
                    if len(headlines) >= 5: break
            
            return headlines
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []

    def analyze_sentiment(self, ticker):
        """Fetches news and returns a sentiment score (-1 to 1) and summary."""
        headlines = self.fetch_news(ticker)
        if not headlines:
            return {"score": 0, "reason": "No recent news found.", "headlines": []}
            
        prompt = f"""
        Analyze the sentiment of the following news headlines for the stock '{ticker}':
        {json.dumps(headlines, indent=2)}
        
        Determine if the news is BULLISH, BEARISH, or NEUTRAL.
        Return a JSON object with:
        - "score": A float between -1.0 (Very Bearish) and 1.0 (Very Bullish).
        - "reason": A brief 1-sentence summary of why.
        
        JSON Response Only:
        """
        
        try:
            response = self.llm.generate(prompt)
            # Clean response to ensure valid JSON
            clean_resp = re.search(r'\{.*\}', response, re.DOTALL)
            if clean_resp:
                data = json.loads(clean_resp.group())
                # Validate that required keys exist
                if 'score' not in data or 'reason' not in data:
                    print(f"LLM response missing required keys. Got: {data.keys()}")
                    return {"score": 0, "reason": "LLM response format error", "headlines": headlines[:2]}
                data['headlines'] = headlines[:2] # Keep top 2 for display
                return data
            else:
                return {"score": 0, "reason": "LLM format error", "headlines": headlines[:2]}
                
        except Exception as e:
            print(f"Sentiment Analysis Failed: {e}")
            return {"score": 0, "reason": "Analysis failed", "headlines": headlines[:2]}

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print(analyzer.analyze_sentiment("RELIANCE"))
