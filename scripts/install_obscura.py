#!/usr/bin/env python3
"""
Downloads and installs the Obscura headless browser binary for the current platform.
Linux -> /usr/local/bin/obscura
Windows -> bin/obscura.exe
"""

import os
import sys
import platform
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path

VERSION = "v0.1.1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINUX_URL = f"https://github.com/h4ckf0r0day/obscura/releases/download/{VERSION}/obscura-x86_64-linux.tar.gz"
WINDOWS_URL = f"https://github.com/h4ckf0r0day/obscura/releases/download/{VERSION}/obscura-x86_64-windows.zip"

def download_file(url, dest):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded to {dest}")

def install_linux():
    print("Installing for Linux...")
    tmp_archive = "/tmp/obscura.tar.gz"
    download_file(LINUX_URL, tmp_archive)
    
    print("Extracting archive...")
    with tarfile.open(tmp_archive, "r:gz") as tar:
        tar.extractall(path="/tmp")
        
    extracted_binary = "/tmp/obscura"
    dest_binary = "/usr/local/bin/obscura"
    
    if os.path.exists(extracted_binary):
        shutil.move(extracted_binary, dest_binary)
        os.chmod(dest_binary, 0o755)
        print(f"Successfully installed Obscura to {dest_binary}")
    else:
        print("Failed to find extracted binary.")
        sys.exit(1)
        
    os.remove(tmp_archive)

def install_windows():
    print("Installing for Windows...")
    bin_dir = PROJECT_ROOT / "bin"
    bin_dir.mkdir(exist_ok=True)
    
    tmp_archive = bin_dir / "obscura.zip"
    download_file(WINDOWS_URL, tmp_archive)
    
    print("Extracting archive...")
    with zipfile.ZipFile(tmp_archive, 'r') as zip_ref:
        zip_ref.extractall(bin_dir)
        
    extracted_binary = bin_dir / "obscura.exe"
    
    if extracted_binary.exists():
        print(f"Successfully installed Obscura to {extracted_binary}")
    else:
        print(f"Failed to find extracted binary at {extracted_binary}.")
        
    tmp_archive.unlink()

def main():
    system = platform.system()
    if system == "Linux":
        # Requires sudo
        if os.geteuid() != 0:
            print("Please run as root (sudo) to install in /usr/local/bin")
            sys.exit(1)
        install_linux()
    elif system == "Windows":
        install_windows()
    else:
        print(f"Unsupported OS: {system}. Obscura requires Linux or Windows (x86_64).")
        sys.exit(1)

if __name__ == "__main__":
    main()
