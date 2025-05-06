#!/usr/bin/env python3
"""
Master test runner for CasaLingua tests
"""
import argparse
import asyncio
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

class Colors:
    """Terminal color codes for better output readability"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_header(message):
    """Print a header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}")

def log_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def log_failure(message):
    """Print a failure message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def log_info(message):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def load_test_module(file_path):
    """Load a Python module from a file path"""
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

async def run_python_test(file_path):
    """Run a Python test file"""
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            log_success(f"{file_path.name} - {duration:.2f}s")
            return True
        else:
            log_failure(f"{file_path.name} - {duration:.2f}s")
            print(f"  Stdout: {result.stdout.strip()[:100]}...")
            print(f"  Stderr: {result.stderr.strip()[:100]}...")
            return False
    except Exception as e:
        duration = time.time() - start_time
        log_failure(f"{file_path.name} - {duration:.2f}s - {str(e)}")
        return False

async def run_shell_script(file_path):
    """Run a shell script test"""
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["/bin/bash", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            log_success(f"{file_path.name} - {duration:.2f}s")
            return True
        else:
            log_failure(f"{file_path.name} - {duration:.2f}s")
            print(f"  Stdout: {result.stdout.strip()[:100]}...")
            print(f"  Stderr: {result.stderr.strip()[:100]}...")
            return False
    except Exception as e:
        duration = time.time() - start_time
        log_failure(f"{file_path.name} - {duration:.2f}s - {str(e)}")
        return False

async def run_all_tests(test_type=None, test_category=None, test_path=None):
    """Run all tests matching the given criteria"""
    tests_dir = Path(__file__).parent
    
    # Set up test types and categories
    test_types = {
        "unit": tests_dir / "unit",
        "integration": tests_dir / "integration",
        "scripts": tests_dir / "scripts"
    }
    
    # Track results
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # Single file test
    if test_path:
        test_path = Path(test_path)
        if not test_path.exists():
            log_failure(f"Test file not found: {test_path}")
            return 1
        
        log_header(f"Running single test: {test_path.name}")
        
        if test_path.suffix == '.py':
            success = await run_python_test(test_path)
        elif test_path.suffix == '.sh':
            success = await run_shell_script(test_path)
        else:
            log_failure(f"Unsupported file type: {test_path}")
            return 1
        
        results["total"] += 1
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Run by test type and category
    else:
        # Filter by test type
        if test_type:
            if test_type not in test_types:
                log_failure(f"Unknown test type: {test_type}")
                return 1
            
            test_types = {test_type: test_types[test_type]}
        
        # Run tests for each type
        for type_name, type_dir in test_types.items():
            if not type_dir.exists():
                continue
            
            # Find all test files of this type
            if type_name == "scripts":
                test_files = sorted(type_dir.glob("*.sh"))
                if test_category:
                    test_files = [f for f in test_files if test_category in f.name]
            else:
                # Get categories within this test type
                categories = [d for d in type_dir.iterdir() if d.is_dir()]
                
                # Filter by category if specified
                if test_category:
                    categories = [d for d in categories if d.name == test_category]
                
                # Add root directory tests
                root_tests = [f for f in type_dir.glob("*.py") if f.name != "__init__.py"]
                
                # Get tests from categories
                category_tests = []
                for category in categories:
                    category_tests.extend([f for f in category.glob("*.py") if f.name != "__init__.py"])
                
                test_files = sorted(root_tests + category_tests)
            
            if not test_files:
                log_info(f"No tests found for type '{type_name}'" + 
                        (f" and category '{test_category}'" if test_category else ""))
                continue
            
            # Run tests of this type
            log_header(f"Running {type_name} tests" + 
                      (f" ({test_category})" if test_category else ""))
            
            for file_path in test_files:
                results["total"] += 1
                
                if file_path.suffix == '.py':
                    success = await run_python_test(file_path)
                elif file_path.suffix == '.sh':
                    success = await run_shell_script(file_path)
                else:
                    log_info(f"Skipping unsupported file: {file_path.name}")
                    results["skipped"] += 1
                    continue
                
                if success:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
    
    # Print summary
    log_header("Test Results")
    log_info(f"Total tests: {results['total']}")
    log_success(f"Passed: {results['passed']}")
    
    if results["failed"] > 0:
        log_failure(f"Failed: {results['failed']}")
    else:
        log_info(f"Failed: {results['failed']}")
    
    if results["skipped"] > 0:
        log_info(f"Skipped: {results['skipped']}")
    
    # Calculate pass rate
    if results['total'] > 0:
        pass_rate = (results['passed'] / results['total']) * 100
        if pass_rate >= 90:
            log_success(f"Pass rate: {pass_rate:.1f}%")
        elif pass_rate >= 75:
            log_info(f"Pass rate: {pass_rate:.1f}%")
        else:
            log_failure(f"Pass rate: {pass_rate:.1f}%")
    
    # Return non-zero exit code if any tests failed
    return 0 if results["failed"] == 0 else 1

async def main():
    parser = argparse.ArgumentParser(description="CasaLingua Test Runner")
    parser.add_argument("--type", choices=["unit", "integration", "scripts"],
                       help="Type of tests to run (unit, integration, scripts)")
    parser.add_argument("--category", help="Test category to run (e.g., models, pipeline, api)")
    parser.add_argument("--file", help="Run a specific test file")
    parser.add_argument("--env", default="development", help="Environment (development, production)")
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["CASALINGUA_ENV"] = args.env
    
    log_header(f"CasaLingua Test Runner (Environment: {args.env})")
    
    # Run tests
    return await run_all_tests(args.type, args.category, args.file)

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)