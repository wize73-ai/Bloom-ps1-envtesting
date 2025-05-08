#!/usr/bin/env python3
"""
Coverage Analysis Tool for CasaLingua

This script analyzes coverage reports to identify modules with the lowest coverage,
helping prioritize testing efforts.

Usage:
  python analyze_coverage.py [--source SOURCE]

Options:
  --source SOURCE   Path to coverage data file (default: coverage-reports/coverage.json)
  --min N           Only show modules with coverage below N% (default: 90)
  --top N           Show top N modules with lowest coverage (default: 10)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_coverage_data(source_path: str) -> Dict:
    """
    Load coverage data from JSON file.
    
    Args:
        source_path: Path to coverage JSON file
        
    Returns:
        Coverage data dictionary
    """
    if not os.path.exists(source_path):
        print(f"Error: Coverage data file not found: {source_path}")
        print("Run coverage analysis first with: python run_coverage.py --json")
        sys.exit(1)
    
    with open(source_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in coverage data file: {source_path}")
            sys.exit(1)


def calculate_module_coverage(coverage_data: Dict) -> List[Tuple[str, float, int, int]]:
    """
    Calculate coverage percentage for each module.
    
    Args:
        coverage_data: Coverage data dictionary
        
    Returns:
        List of tuples (module_path, coverage_percent, covered_lines, total_lines)
    """
    results = []
    
    # Process each file in the coverage data
    for file_path, file_data in coverage_data['files'].items():
        # Skip files outside the app directory
        if not file_path.startswith('app/'):
            continue
        
        # Calculate coverage for this file
        total_lines = len(file_data['lines'])
        missing_lines = len(file_data.get('missing_lines', []))
        covered_lines = total_lines - missing_lines
        
        if total_lines > 0:
            coverage_percent = (covered_lines / total_lines) * 100
            results.append((file_path, coverage_percent, covered_lines, total_lines))
    
    # Sort by coverage percentage (ascending)
    results.sort(key=lambda x: x[1])
    
    return results


def analyze_by_module_group(coverage_results: List[Tuple[str, float, int, int]]) -> Dict[str, Dict]:
    """
    Group coverage results by module and calculate aggregate coverage.
    
    Args:
        coverage_results: List of file coverage results
        
    Returns:
        Dictionary of module groups with coverage statistics
    """
    module_groups = {}
    
    for file_path, coverage_percent, covered_lines, total_lines in coverage_results:
        # Determine module group (first two path components)
        parts = file_path.split('/')
        if len(parts) >= 2:
            module_group = '/'.join(parts[:2])
        else:
            module_group = parts[0]
        
        # Initialize or update module group data
        if module_group not in module_groups:
            module_groups[module_group] = {
                'covered_lines': 0,
                'total_lines': 0,
                'files': 0
            }
        
        # Add file stats to module group
        module_groups[module_group]['covered_lines'] += covered_lines
        module_groups[module_group]['total_lines'] += total_lines
        module_groups[module_group]['files'] += 1
    
    # Calculate coverage percentage for each module group
    for module_group in module_groups:
        covered = module_groups[module_group]['covered_lines']
        total = module_groups[module_group]['total_lines']
        if total > 0:
            module_groups[module_group]['coverage_percent'] = (covered / total) * 100
        else:
            module_groups[module_group]['coverage_percent'] = 0
    
    return module_groups


def print_coverage_report(module_results: Dict[str, Dict], file_results: List[Tuple[str, float, int, int]], 
                         min_threshold: float = 90, top_n: int = 10):
    """
    Print coverage analysis report.
    
    Args:
        module_results: Module group coverage results
        file_results: File coverage results
        min_threshold: Only show items with coverage below this percentage
        top_n: Show only the top N items with lowest coverage
    """
    # Convert module results to a sorted list
    module_list = [(name, data['coverage_percent'], data['covered_lines'], 
                   data['total_lines'], data['files']) 
                  for name, data in module_results.items()]
    module_list.sort(key=lambda x: x[1])  # Sort by coverage percentage
    
    # Overall statistics
    total_covered = sum(data['covered_lines'] for data in module_results.values())
    total_lines = sum(data['total_lines'] for data in module_results.values())
    overall_coverage = (total_covered / total_lines) * 100 if total_lines > 0 else 0
    
    print("=" * 80)
    print(f"COVERAGE ANALYSIS REPORT - Overall: {overall_coverage:.2f}%")
    print("=" * 80)
    
    # Module group report
    print("\nMODULE COVERAGE:")
    print("-" * 80)
    print(f"{'Module':<30} {'Coverage':<10} {'Lines':<15} {'Files':<8}")
    print("-" * 80)
    
    for name, coverage, covered, total, file_count in module_list:
        if coverage >= min_threshold:
            continue
        print(f"{name:<30} {coverage:>7.2f}%  {covered:>6}/{total:<6}  {file_count:>6}")
    
    # Low coverage files report
    print("\n\nLOWEST COVERAGE FILES:")
    print("-" * 80)
    print(f"{'File Path':<50} {'Coverage':<10} {'Lines':<15}")
    print("-" * 80)
    
    shown = 0
    for file_path, coverage, covered, total in file_results:
        if coverage >= min_threshold or shown >= top_n:
            continue
        print(f"{file_path:<50} {coverage:>7.2f}%  {covered:>6}/{total:<6}")
        shown += 1
    
    # Improvement targets
    target_coverage = 85
    lines_needed = int((target_coverage / 100 * total_lines) - total_covered)
    
    print("\n\nIMPROVEMENT TARGETS:")
    print("-" * 80)
    print(f"Current overall coverage: {overall_coverage:.2f}%")
    print(f"Target coverage: {target_coverage:.2f}%")
    if overall_coverage < target_coverage:
        print(f"Lines needed to reach target: {lines_needed}")
        
        # Calculate how many tests might be needed
        avg_lines_per_test = 10  # Rough estimate
        estimated_tests = lines_needed // avg_lines_per_test + 1
        print(f"Estimated number of tests needed: ~{estimated_tests}")
    else:
        print("Target already achieved!")


def main():
    parser = argparse.ArgumentParser(description="Analyze test coverage data")
    parser.add_argument("--source", default="coverage-reports/coverage.json", 
                       help="Path to coverage data file")
    parser.add_argument("--min", type=float, default=90, 
                       help="Only show modules with coverage below N%%")
    parser.add_argument("--top", type=int, default=10, 
                       help="Show top N modules with lowest coverage")
    
    args = parser.parse_args()
    
    # Load coverage data
    coverage_data = load_coverage_data(args.source)
    
    # Calculate file coverage
    file_results = calculate_module_coverage(coverage_data)
    
    # Group by module and calculate aggregate coverage
    module_results = analyze_by_module_group(file_results)
    
    # Print report
    print_coverage_report(module_results, file_results, args.min, args.top)


if __name__ == "__main__":
    main()