#!/usr/bin/env python3
"""
Test Coverage Runner for CasaLingua

This script runs the test suite with coverage analysis and generates comprehensive reports.
It allows targeting specific modules or running the full test suite.

Usage:
  python run_coverage.py [options]

Options:
  --module MODULE    Run tests only for the specified module (e.g., 'api', 'core')
  --file FILE        Run tests only for the specified file (e.g., 'app/utils/config.py')
  --html             Generate HTML coverage report
  --xml              Generate XML coverage report
  --json             Generate JSON coverage report
  --console          Show coverage report in console
  --min-coverage N   Set minimum required coverage percentage (default: 85)
  --fail-under N     Exit with status code 2 if coverage is below N percent
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def get_coverage_targets(module=None, file_path=None):
    """
    Determine which parts of the codebase to analyze for coverage.
    
    Args:
        module: Optional module name (e.g., 'api', 'core')
        file_path: Optional specific file path to check
    
    Returns:
        List of module paths to include in coverage
    """
    if file_path:
        # If a specific file is specified, target only that file
        return [file_path]
    
    if module:
        # If a module is specified, target that module
        return [f"app/{module}"]
    
    # Default to covering the entire app
    return ["app"]


def run_coverage(targets, html=False, xml=False, json=False, console=True, min_coverage=85, fail_under=None):
    """
    Run tests with coverage analysis and generate reports.
    
    Args:
        targets: List of module paths to include in coverage
        html: Whether to generate HTML report
        xml: Whether to generate XML report
        json: Whether to generate JSON report
        console: Whether to show coverage report in console
        min_coverage: Minimum required coverage percentage
        fail_under: Exit with status code 2 if coverage is below this percent
    
    Returns:
        Tuple of (success, coverage_percent)
    """
    # Set up coverage output directories
    coverage_dir = ROOT_DIR / "coverage-reports"
    os.makedirs(coverage_dir, exist_ok=True)
    os.makedirs(coverage_dir / "html", exist_ok=True)
    
    # Build the pytest command
    cmd = ["pytest"]
    
    # Add coverage targets
    for target in targets:
        cmd.extend(["--cov", target])
    
    # Add report options
    if html:
        cmd.extend(["--cov-report", f"html:{coverage_dir}/html"])
    if xml:
        cmd.extend(["--cov-report", f"xml:{coverage_dir}/coverage.xml"])
    if json:
        cmd.extend(["--cov-report", f"json:{coverage_dir}/coverage.json"])
    if console:
        cmd.append("--cov-report=term")
    
    # Add fail-under if specified
    if fail_under is not None:
        cmd.extend(["--cov-fail-under", str(fail_under)])
    
    # Run the command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("\n" + "=" * 80)
    print("TEST RESULTS:")
    print("=" * 80)
    print(result.stdout)
    
    if result.stderr:
        print("\n" + "=" * 80)
        print("ERRORS:")
        print("=" * 80)
        print(result.stderr)
    
    # Parse coverage percentage from output
    coverage_percent = None
    for line in result.stdout.splitlines():
        if "TOTAL" in line and "%" in line:
            parts = line.split("%")[0].split()
            try:
                coverage_percent = float(parts[-1])
                break
            except (ValueError, IndexError):
                pass
    
    # Show coverage summary
    if coverage_percent is not None:
        print("\n" + "=" * 80)
        print(f"COVERAGE SUMMARY: {coverage_percent:.2f}%")
        if coverage_percent < min_coverage:
            print(f"WARNING: Coverage below target of {min_coverage}%")
        print("=" * 80)
    
    return result.returncode == 0, coverage_percent


def main():
    parser = argparse.ArgumentParser(description="Run tests with coverage analysis")
    parser.add_argument("--module", help="Run tests only for the specified module (e.g., 'api', 'core')")
    parser.add_argument("--file", help="Run tests only for the specified file")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report")
    parser.add_argument("--json", action="store_true", help="Generate JSON coverage report")
    parser.add_argument("--console", action="store_true", help="Show coverage report in console")
    parser.add_argument("--min-coverage", type=float, default=85.0, help="Minimum required coverage percentage")
    parser.add_argument("--fail-under", type=float, help="Exit with status code 2 if coverage is below N percent")
    
    args = parser.parse_args()
    
    # Default to console report if no report types specified
    if not any([args.html, args.xml, args.json, args.console]):
        args.console = True
    
    # Get coverage targets
    targets = get_coverage_targets(args.module, args.file)
    
    # Run coverage
    success, coverage_percent = run_coverage(
        targets=targets,
        html=args.html,
        xml=args.xml,
        json=args.json,
        console=args.console,
        min_coverage=args.min_coverage,
        fail_under=args.fail_under
    )
    
    # Determine exit code
    if not success:
        return 1
    if args.fail_under is not None and coverage_percent < args.fail_under:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())