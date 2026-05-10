import paramiko
import sys

with paramiko.SSHClient() as client:
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('64.227.139.165', username='root', password='maniS@12345H')
    
    stdin, stdout, stderr = client.exec_command('docker logs --tail 2000 tradesage-scanner | grep -E "Obscura execution failed|Screener returned 403|Screener returned HTTP|Obscura timed out" | tail -n 50')
    sys.stdout.buffer.write(stdout.read())
    sys.stderr.buffer.write(stderr.read())
