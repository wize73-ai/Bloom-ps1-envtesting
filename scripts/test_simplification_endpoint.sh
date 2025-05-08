#!/bin/bash
# Test script for the simplification endpoint

# Set base URL - default to localhost:8000 if not provided
BASE_URL=${1:-"http://localhost:8000"}
echo "Using base URL: $BASE_URL"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test function for simplification endpoint
test_simplification() {
    local level=$1
    local input_text="$2"
    local expected_content="$3"
    local endpoint="/api/v1/simplify"
    
    echo -e "${BLUE}Testing simplification at level $level${NC}"
    echo "Input text: $input_text"
    
    # Make the API call
    response=$(curl -s -X POST "$BASE_URL$endpoint" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$input_text\", \"parameters\": {\"level\": $level}}")
    
    # Extract simplified text from response
    simplified_text=$(echo $response | jq -r '.result')
    
    echo "Simplified text: $simplified_text"
    
    # Simple validation - check if response contains expected content
    if [[ $simplified_text == *"$expected_content"* ]]; then
        echo -e "${GREEN}✓ Test passed: Response contains expected content${NC}"
        return 0
    else
        echo -e "${RED}✗ Test failed: Response does not contain expected content${NC}"
        echo "Expected to find: $expected_content"
        return 1
    fi
}

# Test with different simplification levels
echo -e "\n${BLUE}===== Testing Simplification Endpoint =====${NC}\n"

# Level 1 (minimal simplification)
test_simplification 1 "The implementation of the algorithm necessitated a comprehensive understanding of advanced mathematical principles and computational methodologies." "algorithm" || failed=1

# Level 3 (medium simplification)
test_simplification 3 "The cardiovascular system functions through a complex interaction of hemodynamic processes and electrophysiological mechanisms to maintain homeostasis." "heart" || failed=1

# Level 5 (maximum simplification)
test_simplification 5 "The meteorological conditions for tomorrow indicate a high probability of precipitation accompanied by electrical atmospheric discharges." "rain" || failed=1

# Test with domain-specific text (legal)
test_simplification 3 "The tenant hereby agrees to refrain from any activities that may cause disturbance to other occupants of the premises, and acknowledges that violation of this covenant may constitute grounds for termination of the lease agreement." "tenant" "agreement" || failed=1

if [ "$failed" == "1" ]; then
    echo -e "\n${RED}Some tests failed.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
fi