import os
import requests
import time
from urllib.parse import urlparse, unquote

URLS = [
    "https://nsearchives.nseindia.com/web/sites/default/files/2023-03/Brochure_Trading_Strategy_for_Market.pdf",
    "https://www.neuroquantology.com/media/article_pdfs/NQ_Sep_713_U_A_study_on_the_effectiveness_of_trading_strategies_in_stock_ma_iz08Gr8.pdf",
    "https://catalogimages.wiley.com/images/db/pdf/9780470293683.excerpt.pdf",
    "https://download.e-bookshelf.de/download/0000/5841/69/L-G-0000584169-0002384251.pdf",
    "https://www.ijstr.org/final-print/mar2020/Prediction-Of-Stock-Trend-For-Swing-Trades-Using-Long-Short-term-Memory-Neural-Network-Model.pdf",
    "https://bearbulltraders.com/docs/HowtoSwingTrade-AudioBook-compressed.pdf",
    "https://dl.fxf1.com/files/books/english/Alan%20Farley%20-%20The%20Master%20Swing%20Trader.pdf",
    "https://rcptec.com/wp-content/uploads/stock-market-course-content.pdf",
    "https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4307517_code5629373.pdf?abstractid=4307517&mirid=1"
]

DOWNLOAD_DIR = "z:/Trade AI/data/raw_pdfs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print(f"Starting download of {len(URLS)} files...")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

for i, url in enumerate(URLS):
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(unquote(parsed_url.path))
        
        # Fallback if filename is empty or generic
        if not filename or filename.endswith('/'):
            filename = f"document_{i+1}.pdf"
        
        # Clean query params if they exist in filename (rare but possible with some parsing)
        if "?" in filename:
            filename = filename.split("?")[0]

        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        save_path = os.path.join(DOWNLOAD_DIR, filename)
        
        # Handle duplicates
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(save_path):
            save_path = os.path.join(DOWNLOAD_DIR, f"{base_name}_{counter}{ext}")
            counter += 1
            
        print(f"[{i+1}/{len(URLS)}] Downloading: {filename} ...")
        
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   -> Saved to: {save_path}")
        else:
            print(f"   -> Failed! Status Code: {response.status_code}")

    except Exception as e:
        print(f"   -> Error: {str(e)}")
    
    time.sleep(1)

print("\nAll downloads processed.")
