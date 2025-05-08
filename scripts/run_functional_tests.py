#!/usr/bin/env python3
"""
Run functional tests for the CasaLingua application.

This script sets up the environment and runs the functional tests.
"""
import os
import subprocess
import sys
import argparse
import time
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

def check_server_running(url="http://localhost:8000"):
    """Check if the CasaLingua server is running."""
    import requests
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_server():
    """Start the server for testing."""
    print_header("Starting CasaLingua Server")
    
    server_process = subprocess.Popen(
        [sys.executable, str(ROOT_DIR / "scripts" / "run_casalingua.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy()
    )
    
    # Wait for server to start
    print(f"{Colors.CYAN}Waiting for server to start...{Colors.ENDC}")
    tries = 0
    max_tries = 30
    while tries < max_tries:
        if check_server_running():
            print(f"{Colors.GREEN}Server is running!{Colors.ENDC}")
            return server_process
        tries += 1
        time.sleep(2)
    
    # If server failed to start, kill the process and exit
    print(f"{Colors.RED}Server failed to start after {max_tries * 2} seconds.{Colors.ENDC}")
    server_process.terminate()
    sys.exit(1)

def stop_server(server_process):
    """Stop the server after testing."""
    print_header("Stopping CasaLingua Server")
    
    if server_process:
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
            print(f"{Colors.GREEN}Server stopped.{Colors.ENDC}")
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
            print(f"{Colors.YELLOW}Server killed forcefully.{Colors.ENDC}")

def run_tests(args):
    """Run the functional tests."""
    print_header("Running Functional Tests")
    
    # Set up environment
    os.environ["CASALINGUA_ENV"] = "test"
    os.environ["CASALINGUA_TEST_SERVER"] = args.server_url
    
    # Determine which tests to run
    if args.category:
        test_path = ROOT_DIR / "tests" / "functional" / args.category
    else:
        test_path = ROOT_DIR / "tests" / "functional"
    
    if args.pattern:
        test_spec = str(test_path / f"**/*{args.pattern}*.py")
    else:
        test_spec = str(test_path)
    
    # Build the pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        test_spec,
        "-v",
    ]
    
    if args.junit:
        # Add JUnit report generation
        junit_dir = ROOT_DIR / "junit-reports"
        junit_dir.mkdir(exist_ok=True)
        cmd.extend(["--junitxml", str(junit_dir / "functional-tests.xml")])
    
    # Run the tests
    print(f"{Colors.CYAN}Running: {' '.join(cmd)}{Colors.ENDC}")
    result = subprocess.run(cmd)
    
    return result.returncode

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run functional tests for CasaLingua")
    parser.add_argument("--server", action="store_true", help="Start a server for testing")
    parser.add_argument("--server-url", default="http://localhost:8000", 
                        help="URL of the server to test against (default: http://localhost:8000)")
    parser.add_argument("--category", choices=["api", "workflows", "pipeline", "server"],
                        help="Test category to run (api, workflows, pipeline, or server)")
    parser.add_argument("--pattern", help="Pattern to match test file names")
    parser.add_argument("--junit", action="store_true", help="Generate JUnit XML reports")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    server_process = None
    
    try:
        # Start server if requested
        if args.server:
            server_process = run_server()
        elif not check_server_running(args.server_url):
            print(f"{Colors.YELLOW}Warning: No server detected at {args.server_url}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Tests requiring a running server will be skipped.{Colors.ENDC}")
        
        # Run tests
        exit_code = run_tests(args)
        
        # Print summary
        if exit_code == 0:
            print(f"\n{Colors.GREEN}All tests passed!{Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}Some tests failed. Exit code: {exit_code}{Colors.ENDC}")
        
        return exit_code
        
    finally:
        # Stop server if we started it
        if server_process:
            stop_server(server_process)

if __name__ == "__main__":
    sys.exit(main())