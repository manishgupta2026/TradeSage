import subprocess
import os
import sys
import tempfile
import logging
import platform
import json
import time
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Disk cache for fundamentals (avoids re-scraping during training) ──
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache" / "fundamentals"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "fundamentals_cache.json"
_CACHE_MAX_AGE_DAYS = 7  # Re-scrape after 7 days

def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.debug(f"Failed to save fundamentals cache: {e}")


class ScreenerScraper:
    """
    Scrapes fundamental data from screener.in using the Obscura headless browser
    to safely bypass bot protections.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.obscura_bin = self._find_obscura()
        self._mem_cache = {}  # In-memory cache for current session
        self._disk_cache = _load_cache()
        
    def _find_obscura(self):
        """Find the Obscura binary based on OS."""
        system = platform.system()
        if system == "Linux":
            if os.path.exists("/usr/local/bin/obscura"):
                return "/usr/local/bin/obscura"
            # Fallback
            return "obscura"
        elif system == "Windows":
            win_bin = self.project_root / "bin" / "obscura.exe"
            if win_bin.exists():
                return str(win_bin)
            return "obscura.exe"
        else:
            return "obscura"
    
    def fetch_fundamentals(self, symbol: str, use_cache: bool = True) -> dict:
        """
        Fetch and parse fundamental ratios and shareholding from Screener.
        Returns a dictionary of key metrics.
        Uses disk cache to avoid redundant scraping during training.
        """
        # Check in-memory cache first (fastest)
        if use_cache and symbol in self._mem_cache:
            return self._mem_cache[symbol]
            
        # Check disk cache (fast, survives restarts)
        if use_cache and symbol in self._disk_cache:
            entry = self._disk_cache[symbol]
            age_days = (time.time() - entry.get("_ts", 0)) / 86400
            if age_days < _CACHE_MAX_AGE_DAYS:
                data = {k: v for k, v in entry.items() if not k.startswith("_")}
                self._mem_cache[symbol] = data
                return data
        
        # Live scrape
        formatted_sym = symbol.replace("&", "%26")
        url = f"https://www.screener.in/company/{formatted_sym}/consolidated/"
        
        html = self._run_obscura(url)
        if not html or "404 Not Found" in html or "Oops" in html:
            url_standalone = f"https://www.screener.in/company/{formatted_sym}/"
            logger.debug(f"Consolidated not found for {symbol}, trying standalone: {url_standalone}")
            html = self._run_obscura(url_standalone)
            
        if not html:
            logger.warning(f"Failed to fetch HTML for {symbol} via Obscura.")
            return {}
            
        result = self._parse_html(html)
        
        # Save to both caches
        self._mem_cache[symbol] = result
        self._disk_cache[symbol] = {**result, "_ts": time.time()}
        _save_cache(self._disk_cache)
        
        return result
    
    def fetch_fundamentals_batch(self, symbols: list, delay: float = 1.0) -> dict:
        """
        Fetch fundamentals for a list of symbols with rate limiting.
        Returns {symbol: {metrics}} dict.
        Used by the training pipeline.
        """
        results = {}
        cached = 0
        scraped = 0
        
        for symbol in symbols:
            data = self.fetch_fundamentals(symbol, use_cache=True)
            if data:
                results[symbol] = data
                # If it came from cache, no delay needed
                if symbol in self._disk_cache:
                    age_days = (time.time() - self._disk_cache[symbol].get("_ts", 0)) / 86400
                    if age_days < _CACHE_MAX_AGE_DAYS:
                        cached += 1
                        continue
                scraped += 1
                time.sleep(delay)  # Rate limit live scrapes
            
        logger.info(f"Fundamentals batch: {len(results)}/{len(symbols)} fetched "
                     f"({cached} cached, {scraped} scraped)")
        return results
        
    def _run_obscura(self, url: str) -> str:
        """Execute Obscura CLI and return HTML."""
        try:
            cmd = [
                self.obscura_bin,
                "fetch",
                url,
                "--stealth",
                "--dump", "html",
                "--quiet"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.debug(f"Obscura execution failed: {result.stderr}")
                return ""
                
            return result.stdout
        except Exception as e:
            logger.error(f"Error running Obscura: {e}")
            return ""

    def _parse_html(self, html: str) -> dict:
        """Parse Screener HTML using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")
        data = {}
        
        # 1. Ratios
        ratios_div = soup.find("div", {"id": "top-ratios"}) or soup.find("div", {"class": "company-ratios"})
        if ratios_div:
            for li in ratios_div.find_all("li"):
                name_span = li.find("span", {"class": "name"})
                val_span = li.find("span", {"class": "number"})
                if name_span and val_span:
                    name = name_span.text.strip().replace("\n", " ").strip()
                    val = val_span.text.strip().replace(",", "")
                    try:
                        data[name] = float(val) if val else None
                    except ValueError:
                        data[name] = val

        # 2. Shareholding Pattern
        sh_section = soup.find("section", {"id": "shareholding"})
        if sh_section:
            table = sh_section.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all("td")
                    if not cols:
                        continue
                    name_raw = cols[0].text.lower()
                    latest_val = cols[-1].text.strip().replace("%", "")
                    try:
                        val = float(latest_val)
                        if "promoter" in name_raw:
                            data["sh_Promoters"] = val
                        elif "fii" in name_raw:
                            data["sh_FIIs"] = val
                        elif "dii" in name_raw:
                            data["sh_DIIs"] = val
                    except ValueError:
                        pass
                        
        # 3. Debt to Equity (Calculated from Balance Sheet)
        bs_section = soup.find("section", {"id": "balance-sheet"})
        if bs_section:
            table = bs_section.find("table")
            if table:
                rows = table.find_all("tr")
                borrowings = 0
                equity = 0
                for row in rows:
                    cols = row.find_all("td")
                    if not cols:
                        continue
                    name = cols[0].text.strip()
                    latest_val = cols[-1].text.strip().replace(",", "")
                    if "Borrowings" in name:
                        borrowings = float(latest_val) if latest_val.replace('.', '').isdigit() else 0
                    if "Share Capital" in name or "Reserves" in name:
                        equity += float(latest_val) if latest_val.replace('.', '').isdigit() else 0
                if equity > 0:
                    data["debt_to_equity"] = borrowings / equity

        # Standardize keys for the master filter AND ML features
        result = {
            "pe_ratio": data.get("Stock P/E"),
            "roe": data.get("ROE"),
            "roce": data.get("ROCE"),
            "profit_margins": None,
            "debt_to_equity": data.get("debt_to_equity"),
            "promoter_holding": data.get("sh_Promoters"),
            "fii_holding": data.get("sh_FIIs"),
            "dii_holding": data.get("sh_DIIs"),
            "dividend_yield": data.get("Dividend Yield"),
        }
        
        # Clean up keys
        return {k: v for k, v in result.items() if v is not None}

if __name__ == "__main__":
    # Test script locally
    logging.basicConfig(level=logging.DEBUG)
    scraper = ScreenerScraper()
    res = scraper.fetch_fundamentals("RELIANCE")
    print("RELIANCE Fundamentals:")
    for k, v in res.items():
        print(f"  {k}: {v}")

