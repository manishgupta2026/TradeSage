import paramiko

script = """
import logging
logging.basicConfig(level=logging.WARNING)
from src.core.screener_scraper import ScreenerScraper

s = ScreenerScraper()
html = s._fetch_html("https://www.screener.in/company/ASIANTILES/consolidated/")
print(f"Len: {len(html)}")
print(f"HTML: {html[:100]}")
"""

with paramiko.SSHClient() as client:
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('64.227.139.165', username='root', password='maniS@12345H')
    
    # Save script to container
    cmd = f"docker exec tradesage-scanner sh -c 'echo \"{script.replace('\"', '\\\"').replace('$', '\\$')}\" > /app/test_fetch.py'"
    client.exec_command(cmd)
    
    # Run script
    stdin, stdout, stderr = client.exec_command('docker exec tradesage-scanner python /app/test_fetch.py')
    print('OUT:', stdout.read().decode('utf-8', 'ignore'))
    print('ERR:', stderr.read().decode('utf-8', 'ignore'))
