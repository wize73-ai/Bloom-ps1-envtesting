#!/usr/bin/env python3
"""
Direct fix for the 'NoneType' object has no attribute 'lower' error in the simplifier wrapper.
"""

import re
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Path to the wrapper.py file
WRAPPER_PATH = Path(__file__).parent / "app" / "services" / "models" / "wrapper.py"

def fix_domain_null_error():
    """Fix the 'NoneType' object has no attribute 'lower' error in SimplifierWrapper."""
    print(f"Fixing domain null error in {WRAPPER_PATH}")
    
    # Read the wrapper.py file
    with open(WRAPPER_PATH, 'r') as f:
        content = f.read()
    
    # Fix the error in the _preprocess method
    content = content.replace(
        "is_legal_domain = domain.lower() in",
        "is_legal_domain = domain and domain.lower() in"
    )
    
    # Write the updated content back to the file
    with open(WRAPPER_PATH, 'w') as f:
        f.write(content)
    
    print(f"Successfully fixed domain null error in {WRAPPER_PATH}")
    return True

if __name__ == "__main__":
    success = fix_domain_null_error()
    if success:
        print("Fix applied successfully.")
    else:
        print("Failed to apply fix.")
        sys.exit(1)