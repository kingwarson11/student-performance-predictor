#!/usr/bin/env python3
"""
setup.py — One-click setup: installs deps, trains model, starts server.
Usage: python setup.py
"""
import subprocess
import sys
import os

def run(cmd, desc):
    print(f"\n{'='*50}")
    print(f"▶  {desc}")
    print('='*50)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {desc}")
        sys.exit(1)
    print(f"✅ Done: {desc}")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║   EduPredict — Student Performance AI   ║
║         Setup & Launch Script           ║
╚══════════════════════════════════════════╝
""")
    run(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies")
    run(f"{sys.executable} backend/train.py", "Training ML model")

    print("""
╔══════════════════════════════════════════╗
║  🚀  Server starting on port 5000        ║
║  🌐  Open: http://localhost:5000         ║
╚══════════════════════════════════════════╝
""")
    os.system(f"{sys.executable} backend/app.py")
