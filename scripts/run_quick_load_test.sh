#!/bin/bash
# Run a quick load test for enhanced language models
# This is a smaller version of the full load test for quick results

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd "$BASE_DIR" || exit 1

# Create results directory
RESULTS_DIR="$BASE_DIR/load_test_results"
mkdir -p "$RESULTS_DIR"

# Function to print colored output
print_blue() {
  printf "\033[0;34m%s\033[0m\n" "$1"
}

print_yellow() {
  printf "\033[0;33m%s\033[0m\n" "$1"
}

print_green() {
  printf "\033[0;32m%s\033[0m\n" "$1"
}

# Log start of test
print_blue "==================================================="
print_blue "Running quick load test for enhanced language models"
print_blue "Results will be saved in $RESULTS_DIR"
print_blue "==================================================="

# Run quick test with low concurrency and request count
print_yellow "Running quick test: Low load (concurrency=2, requests=5+5)"
python scripts/load_test_enhanced_models.py \
  --concurrency 2 \
  --translation-requests 5 \
  --simplification-requests 5 \
  --output "$RESULTS_DIR/quick_test_results.json"
print_green "Quick test completed"

# Generate quick analysis
print_yellow "Analyzing results..."
python scripts/analyze_load_test_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$RESULTS_DIR"
print_green "Analysis completed"

print_blue "==================================================="
print_blue "Quick load test completed"
print_blue "Results available at: $RESULTS_DIR/quick_test_results.json"
print_blue "Analysis available at: $RESULTS_DIR/performance_analysis.md"
print_blue "==================================================="