#!/usr/bin/env python3
"""
Install required packages for TTS functionality.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for TTS functionality."""
    requirements = [
        "gtts",       # Google Text-to-Speech
        "pydub",      # Audio processing
        "audioread"   # Audio file reading
    ]
    
    print("Installing TTS requirements...")
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            print("You may need to manually install this package.")
    
    print("Installation complete!")
    print("Now you can run the TTS tests with: python scripts/test_speech_endpoints_fixed.py")

if __name__ == "__main__":
    install_requirements()