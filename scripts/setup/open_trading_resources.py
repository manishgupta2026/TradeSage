import webbrowser
import time

print("Opening best resource pages for Swing Trading PDFs...")

urls = [
    # Zerodha Varsity - Technical Analysis
    "https://zerodha.com/varsity/module/technical-analysis/",
    
    # NISM Downloads
    "https://www.nism.ac.in/workbook/",
    
    # NSE India - Technical Analysis Resources
    "https://www.nseindia.com/learn/invest-basics",
    
    # General Search for specific PDF types
    "https://www.google.com/search?q=filetype%3Apdf+swing+trading+strategies+indian+stock+market"
]

for url in urls:
    print(f"Opening: {url}")
    webbrowser.open(url)
    time.sleep(1)

print("\n------------------------------------------------")
print("Please manually download the PDFs from these tabs.")
print("Save them to: z:\\Trade AI\\data\\raw_pdfs")
print("------------------------------------------------")
