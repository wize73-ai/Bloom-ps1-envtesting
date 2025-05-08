#!/usr/bin/env python3
"""
Run just the language detection functional test.
"""
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    """Main entry point."""
    # Set up environment
    os.environ["CASALINGUA_ENV"] = "test"
    os.environ["CASALINGUA_TEST_SERVER"] = "http://localhost:8000"
    
    # Test file path
    test_file = ROOT_DIR / "tests" / "functional" / "api" / "test_language_detection.py"
    
    # Build the pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v",
    ]
    
    # Run the test
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())