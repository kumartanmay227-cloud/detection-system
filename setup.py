#!/usr/bin/env python
"""
Setup script for Road Hazard Detection project.
"""

import subprocess
import sys
import os

def create_venv():
    """Create virtual environment."""
    if sys.platform == "win32":
        venv_cmd = [sys.executable, "-m", "venv", "venv"]
    else:
        venv_cmd = ["python3", "-m", "venv", "venv"]
    
    subprocess.run(venv_cmd, check=True)
    print("✅ Virtual environment created: venv/")

def install_requirements():
    """Install project requirements."""
    if sys.platform == "win32":
        pip_path = "./venv/Scripts/pip"
    else:
        pip_path = "./venv/bin/pip"
    
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    print("✅ Requirements installed.")

def main():
    print("Setting up Road Hazard Detection project...")
    if not os.path.exists("venv"):
        create_venv()
    install_requirements()
    print("\nSetup complete! Run: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    print("Then: python preprocess.py")

if __name__ == "__main__":
    main()

