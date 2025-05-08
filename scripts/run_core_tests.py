#!/usr/bin/env python3
"""
Run core unit tests with coverage reporting.

This script runs tests for the core modules (main, config) and generates
coverage reports.
"""
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Terminal colors for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def run_tests():
    """Run the core unit tests with coverage"""
    print_header("Running Core Unit Tests with Coverage")
    
    # Set environment variables for testing
    os.environ["CASALINGUA_ENV"] = "test"
    
    # Create coverage directory if it doesn't exist
    coverage_dir = ROOT_DIR / "coverage-reports"
    coverage_dir.mkdir(exist_ok=True)
    
    # List of core modules to test
    core_modules = [
        "app.main",
        "app.utils.config",
        "app.utils.error_handler",
        "app.utils.helpers",
        "app.utils.logging",
    ]
    
    # Test files to run
    test_files = [
        "tests/unit/test_main.py",
        "tests/unit/utils/test_config.py",
    ]
    
    # Build the command
    cmd = [
        sys.executable, "-m", "pytest",
        *test_files,
        f"--cov={','.join(core_modules)}",
        "--cov-report=term",
        "--cov-report=html:coverage-reports/html",
        "--cov-report=xml:coverage-reports/coverage.xml",
        "-v"
    ]
    
    # Run the tests
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{Colors.GREEN}✓ Tests completed successfully{Colors.ENDC}")
        print(f"\n{Colors.CYAN}Coverage reports generated in:{Colors.ENDC}")
        print(f"  - HTML: {coverage_dir / 'html/index.html'}")
        print(f"  - XML: {coverage_dir / 'coverage.xml'}")
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}✗ Tests failed with exit code {e.returncode}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()