#!/bin/bash
# Run load tests for enhanced language models with different concurrency levels
# and request counts to generate performance metrics

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

print_green() {
  printf "\033[0;32m%s\033[0m\n" "$1"
}

print_yellow() {
  printf "\033[0;33m%s\033[0m\n" "$1"
}

# Log start of tests
print_blue "==================================================="
print_blue "Starting load tests for enhanced language models"
print_blue "Results will be saved in $RESULTS_DIR"
print_blue "==================================================="

# Test 1: Low concurrency, low request count
print_yellow "Running test 1: Low load (concurrency=2, requests=10+10)"
python scripts/load_test_enhanced_models.py \
  --concurrency 2 \
  --translation-requests 10 \
  --simplification-requests 10 \
  --output "$RESULTS_DIR/low_load_results.json"
print_green "Test 1 completed"

# Test 2: Medium concurrency, medium request count
print_yellow "Running test 2: Medium load (concurrency=5, requests=20+20)"
python scripts/load_test_enhanced_models.py \
  --concurrency 5 \
  --translation-requests 20 \
  --simplification-requests 20 \
  --output "$RESULTS_DIR/medium_load_results.json"
print_green "Test 2 completed"

# Test 3: High concurrency, high request count
print_yellow "Running test 3: High load (concurrency=10, requests=50+50)"
python scripts/load_test_enhanced_models.py \
  --concurrency 10 \
  --translation-requests 50 \
  --simplification-requests 50 \
  --output "$RESULTS_DIR/high_load_results.json"
print_green "Test 3 completed"

# Test 4: Translation focused
print_yellow "Running test 4: Translation focused (concurrency=8, requests=100+10)"
python scripts/load_test_enhanced_models.py \
  --concurrency 8 \
  --translation-requests 100 \
  --simplification-requests 10 \
  --output "$RESULTS_DIR/translation_focused_results.json"
print_green "Test 4 completed"

# Test 5: Simplification focused
print_yellow "Running test 5: Simplification focused (concurrency=8, requests=10+100)"
python scripts/load_test_enhanced_models.py \
  --concurrency 8 \
  --translation-requests 10 \
  --simplification-requests 100 \
  --output "$RESULTS_DIR/simplification_focused_results.json"
print_green "Test 5 completed"

# Generate summary report
print_yellow "Generating summary report..."
cat > "$RESULTS_DIR/summary.txt" << EOL
# Load Test Summary

Test performed on: $(date)

## Test Scenarios
1. Low load: concurrency=2, translation=10, simplification=10
2. Medium load: concurrency=5, translation=20, simplification=20
3. High load: concurrency=10, translation=50, simplification=50
4. Translation focused: concurrency=8, translation=100, simplification=10
5. Simplification focused: concurrency=8, translation=10, simplification=100

## Results Location
Detailed JSON results for each test can be found in:
- $RESULTS_DIR/low_load_results.json
- $RESULTS_DIR/medium_load_results.json
- $RESULTS_DIR/high_load_results.json
- $RESULTS_DIR/translation_focused_results.json
- $RESULTS_DIR/simplification_focused_results.json

## Next Steps
1. Analyze the results to identify performance bottlenecks
2. Compare throughput and latency across different concurrency levels
3. Check memory usage patterns for potential memory leaks
4. Consider optimizations for the most resource-intensive operations
EOL

print_blue "==================================================="
print_blue "Load tests completed"
print_blue "Summary available at: $RESULTS_DIR/summary.txt"
print_blue "==================================================="