import requests
import time

URLS = [
    ("https://dl.fxf1.com/files/books/english/Alan%20Farley%20-%20The%20Master%20Swing%20Trader.pdf", "z:/Trade AI/data/raw_pdfs/Alan_Farley_Master_Swing_Trader.pdf"),
    ("https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4307517_code5629373.pdf?abstractid=4307517&mirid=1", "z:/Trade AI/data/raw_pdfs/SSRN_Study.pdf")
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://google.com"
}

print("Retrying failed downloads...")

for url, save_path in URLS:
    try:
        print(f"Downloading: {url} ...")
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   -> SUCCESS: Saved to {save_path}")
        else:
            print(f"   -> FAILED: Status Code {response.status_code}")
    except Exception as e:
        print(f"   -> ERROR: {e}")

print("Retry process complete.")
