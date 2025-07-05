#!/usr/bin/env python3
"""
Script to install/upgrade dependencies for the portfolio optimizer
"""

import subprocess
import sys

def run_command(command):
    """Run a command and return success status"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Install dependencies"""
    print("🔧 Installing Portfolio Optimizer Dependencies")
    print("=" * 60)
    
    commands = [
        "pip uninstall yfinance -y",
        "pip install --upgrade pip",
        "pip install -r requirements.txt --upgrade --no-cache-dir"
    ]
    
    success_count = 0
    for cmd in commands:
        if run_command(cmd):
            success_count += 1
        print()
    
    print("=" * 60)
    if success_count == len(commands):
        print("🎉 All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_data_fetch.py")
        print("2. If tests pass, run: streamlit run app.py")
    else:
        print(f"⚠️ {len(commands) - success_count} command(s) failed")
        print("Please check the errors above and try again")

if __name__ == "__main__":
    main()