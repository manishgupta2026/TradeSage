import paramiko
import sys

with paramiko.SSHClient() as client:
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('64.227.139.165', username='root', password='maniS@12345H')
    
    stdin, stdout, stderr = client.exec_command('docker exec tradesage-scanner bash -c "obscura fetch https://www.screener.in/company/ASIANTILES/consolidated/ --stealth --dump html > /tmp/out.html; echo EXIT_CODE: $?"')
    sys.stdout.buffer.write(stdout.read())
