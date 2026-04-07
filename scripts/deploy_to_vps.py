#!/usr/bin/env python3
"""
Trade AI - VPS Deployment Script
Connects to the VPS and automates the environment setup for the paper trading bot.
"""

import paramiko
import os
import zipfile
import time
from pathlib import Path

# VPS Credentials provided by user
VPS_IP = "64.227.139.165"
VPS_USER = "root"
VPS_PASS = "maniS@12345H"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VPS_PROJECT_DIR = "/root/TradeSage"

def create_zip_archive():
    print("📦 Packing local Trade AI codebase...")
    zip_path = PROJECT_ROOT / "deploy_archive.zip"
    
    # Files and folders to exclude to save bandwidth (we don't upload caches)
    exclude_dirs = {'.git', '__pycache__', 'data_cache_angel', 'data_cache_yfinance', 'venv', '.env', 'node_modules'}
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Strip out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.zip') or file.endswith('.pkl') and 'tradesage_10y.pkl' not in file:
                    continue # Only package the specific model we trained
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, PROJECT_ROOT)
                zipf.write(file_path, arcname)
    
    print(f"✓ Archive created: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
    return zip_path

def deploy_to_vps():
    print(f"\n🚀 Connecting to VPS at {VPS_IP}...")
    
    # Initialize SSH Client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(VPS_IP, username=VPS_USER, password=VPS_PASS, timeout=10)
        print("✓ Connected successfully!")

        # 1. Server Setup
        print("\n🔧 Installing system dependencies (Python, pip, unzip, screen) on Ubuntu...")
        commands = [
            "apt-get update -y",
            "apt-get install -y python3 python3-pip python3-venv unzip screen",
            f"mkdir -p {VPS_PROJECT_DIR}"
        ]
        
        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            stdout.channel.recv_exit_status() # Wait for command to finish

        # 2. Upload Codebase via SFTP
        zip_path = create_zip_archive()
        print(f"\n📤 Uploading codebase to {VPS_PROJECT_DIR}... (This may take a minute depending on model size)")
        
        sftp = ssh.open_sftp()
        sftp.put(str(zip_path), f"{VPS_PROJECT_DIR}/deploy_archive.zip")
        sftp.close()

        # 3. Unzip and Install Python Packages
        print("\n📦 Unzipping files and setting up Python environment on VPS...")
        setup_cmds = [
            f"cd {VPS_PROJECT_DIR} && unzip -o deploy_archive.zip",
            f"cd {VPS_PROJECT_DIR} && rm deploy_archive.zip",
            f"cd {VPS_PROJECT_DIR} && python3 -m venv venv",
            f"cd {VPS_PROJECT_DIR} && ./venv/bin/pip install --upgrade pip",
            f"cd {VPS_PROJECT_DIR} && ./venv/bin/pip install -r requirements.txt",
        ]

        for cmd in setup_cmds:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                print(f"⚠ Warning on command '{cmd}': {stderr.read().decode()}")

        print("\n✅ VPS Setup Complete!")
        print(f"You can now SSH into {VPS_IP} and start the bot using screen.")
        print(f"SSH Command: ssh root@{VPS_IP}")
        print(f"Run Bot: cd {VPS_PROJECT_DIR} && source venv/bin/activate && python scripts/live_trading_angel.py")

    except Exception as e:
        print(f"❌ Connection or processing failed: {e}")
    finally:
        ssh.close()
        # Clean up local zip
        zip_path = PROJECT_ROOT / "deploy_archive.zip"
        if zip_path.exists():
            os.remove(zip_path)

if __name__ == "__main__":
    deploy_to_vps()
