import paramiko
import sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('64.227.139.165', username='root', password='maniS@12345H')

_, stdout, stderr = ssh.exec_command('docker ps --format "table {{.Names}}\t{{.Status}}" && echo "===" && docker logs tradesage-scanner --tail 10')
exit_status = stdout.channel.recv_exit_status()

print(''.join(stdout))
print(''.join(stderr), file=sys.stderr)

sys.exit(exit_status)
