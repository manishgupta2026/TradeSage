import paramiko
import os
from pathlib import Path

VPS_IP = "64.227.139.165"
VPS_USER = "root"
VPS_PASS = "maniS@12345H"
VPS_DIR = "/root/TradeSage"

FILES_TO_UPLOAD = [
    "Dockerfile",
    "api/main.py",
    "src/core/screener_scraper.py",
    "frontend/index.html",
    "frontend/portfolio.html",
    "frontend/stock.html",
    "scripts/install_obscura.py"
]

def deploy():
    print(f"Connecting to {VPS_IP}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_IP, username=VPS_USER, password=VPS_PASS, timeout=10)
    
    sftp = ssh.open_sftp()
    
    for rel_path in FILES_TO_UPLOAD:
        local_path = Path(rel_path)
        remote_path = f"{VPS_DIR}/{rel_path}"
        
        # Ensure remote directory exists
        remote_dir = os.path.dirname(remote_path)
        ssh.exec_command(f"mkdir -p '{remote_dir}'")
        
        print(f"Uploading {rel_path} -> {remote_path}")
        sftp.put(str(local_path), remote_path)
        
    sftp.close()
    
    print("Files uploaded. Rebuilding Docker containers...")
    stdin, stdout, stderr = ssh.exec_command(f"cd '{VPS_DIR}' && docker-compose down && docker-compose up --build -d")
    
    for line in iter(stdout.readline, ""):
        print(line.encode("ascii", "ignore").decode("ascii"), end="")
        
    for line in iter(stderr.readline, ""):
        print("ERR:", line.encode("ascii", "ignore").decode("ascii"), end="")

    ssh.close()
    print("Deployment complete.")

if __name__ == "__main__":
    deploy()
