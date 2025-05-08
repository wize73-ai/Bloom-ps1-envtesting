#!/bin/bash
# CasaLingua Server Testing Demo
# This script directly tests the running server and ensures all JSON output is properly displayed

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Server URL
SERVER_URL="http://localhost:8000"

print_request_info() {
    local method=$1
    local endpoint=$2
    local data=$3

    echo -e "${BOLD}${YELLOW}➡️ Request Overview${NC}"
    echo -e "${YELLOW}──────────────────────────────${NC}"
    echo -e "${BOLD}Method:${NC} $method"
    echo -e "${BOLD}URL:${NC} $SERVER_URL$endpoint"
    if [[ "$method" == "POST" ]]; then
        echo -e "${BOLD}Payload:${NC}"
        echo "$data" | jq . 2>/dev/null | sed 's/^/    /' || echo -e "    $data"
    fi
    echo -e "${YELLOW}──────────────────────────────${NC}"
}

# Function to pretty print JSON
pretty_print_json() {
    if command -v jq &> /dev/null; then
        # Use jq if available (better JSON formatting)
        echo "$1" | jq
    else
        # Otherwise use Python
        echo "$1" | python3 -m json.tool
    fi
}

# Function to make API calls and display all JSON content
test_endpoint() {
    local endpoint=$1
    local method=${2:-GET}
    local data=${3:-""}
    local description=${4:-"Testing endpoint"}
    
    echo -e "\n${YELLOW}────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}${GREEN}Test #$test_num – $description${NC}"
    echo -e "${YELLOW}────────────────────────────────────────────────────${NC}"

    echo -e "${BOLD}${BLUE}$description${NC}"
    echo -e "${BLUE}Endpoint: $method $endpoint${NC}"
    
    print_request_info "$method" "$endpoint" "$data"
    
    if [[ "$method" == "GET" ]]; then
        result=$(curl -s "$SERVER_URL$endpoint")
        status=$?
    else
        result=$(curl -s -X "$method" -H "Content-Type: application/json" -d "$data" "$SERVER_URL$endpoint")
        status=$?
    fi
    
    if [ $status -ne 0 ]; then
        echo -e "${RED}Request failed: curl error $status${NC}"
        return 1
    fi
    
    if [[ "$result" == "" ]]; then
        echo -e "${RED}Empty response from server${NC}"
        return 1
    fi
    
    # Check if result is valid JSON
    if echo "$result" | python3 -m json.tool &> /dev/null; then
        echo -e "${YELLOW}📥 Raw Server Response:${NC}"
        echo -e "${YELLOW}──────────────────────────────${NC}"
        echo "$result"
        echo -e "${YELLOW}──────────────────────────────${NC}"
        echo
        echo -e "${GREEN}✅ Parsed JSON Response:${NC}"
        pretty_print_json "$result"
    else
        echo -e "${YELLOW}Response (not JSON):${NC}"
        echo "$result"
    fi
    
    echo
    return 0
}

# Sample texts for demo
ENGLISH_TEXTS=(
    "The quick brown fox jumps over the lazy dog."
    "Learning a new language opens doors to new cultures and perspectives."
    "The housing agreement must be signed by all tenants prior to occupancy."
    "The patient should take this medication twice daily with food."
)

COMPLEX_TEXTS=(
    "The aforementioned contractual obligations shall be considered null and void if the party of the first part fails to remit payment within the specified timeframe."
    "Notwithstanding the provisions outlined in section 3.2, the tenant hereby acknowledges that the landlord retains the right to access the premises for inspection purposes given reasonable notice."
    "The acquisition of language proficiency necessitates consistent immersion in linguistic contexts that facilitate the assimilation of vocabulary and grammatical constructs."
)

TARGET_LANGUAGES=("es" "fr" "de" "it")

clear
echo -e "${BOLD}${GREEN}===== CasaLingua Server Testing Demo =====${NC}"
echo "This script tests various endpoints of the CasaLingua server."
echo "All JSON responses will be displayed in full detail."
echo

# Check if jq is installed, recommend it for better JSON formatting
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Tip: Install 'jq' for better JSON formatting: brew install jq (macOS) or apt install jq (Linux)${NC}"
    echo
fi

# Start timestamp
START_TIME=$(date +%s)
END_TIME=$((START_TIME + 60))  # Run for approximately 60 seconds

# Test health endpoint with full JSON display
test_endpoint "/health" "GET" "" "Testing Health Endpoint"

# Test detailed health endpoint with full JSON display
test_endpoint "/health/detailed" "GET" "" "Testing Detailed Health Endpoint"

# Test model health endpoint with full JSON display
test_endpoint "/health/models" "GET" "" "Testing Models Health Endpoint"

echo -e "${BOLD}${GREEN}Running multiple tests for about a minute...${NC}"

run_tests() {
  local test_num=1
  while [ $(date +%s) -lt $END_TIME ]; do
    rand_text_idx=$((RANDOM % ${#ENGLISH_TEXTS[@]}))
    rand_lang_idx=$((RANDOM % ${#TARGET_LANGUAGES[@]}))
    text="${ENGLISH_TEXTS[$rand_text_idx]}"
    target="${TARGET_LANGUAGES[$rand_lang_idx]}"

    echo -e "\n${YELLOW}────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}${GREEN}Test #$test_num – Testing Translation Endpoint (EN->$target)${NC}"
    echo -e "${YELLOW}────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}${GREEN}Test #$test_num: Translation EN->$target${NC}"
    translation_data="{\"text\": \"$text\", \"source_language\": \"en\", \"target_language\": \"$target\", \"preserve_formatting\": true}"
    test_endpoint "/pipeline/translate" "POST" "$translation_data" "Testing Translation Endpoint (EN->$target)"

    rand_complex_idx=$((RANDOM % ${#COMPLEX_TEXTS[@]}))
    level=$((RANDOM % 5 + 1))
    text="${COMPLEX_TEXTS[$rand_complex_idx]}"

    echo -e "\n${YELLOW}────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}${GREEN}Test #$test_num – Testing Simplification Endpoint (Level $level)${NC}"
    echo -e "${YELLOW}────────────────────────────────────────────────────${NC}"
    echo -e "${BOLD}${GREEN}Test #$test_num: Simplification Level $level${NC}"
    simplification_data="{\"text\": \"$text\", \"language\": \"en\", \"target_level\": \"$level\"}"
    test_endpoint "/pipeline/simplify" "POST" "$simplification_data" "Testing Simplification Endpoint (Level $level)"

    ((test_num++))

    if [ $(($(date +%s) + 10)) -gt $END_TIME ]; then
        break
    fi

    sleep 0.1
  done
}

if [[ "$PARALLEL_TESTS" =~ ^[1-9][0-9]*$ ]]; then
  echo -e "${YELLOW}Running $PARALLEL_TESTS parallel test loops...${NC}"
  for i in $(seq 1 $PARALLEL_TESTS); do
    run_tests & 
  done
  wait
else
  run_tests
fi

# Final test - verification endpoint 
verification_data="{\"source_text\": \"Hello, how are you?\", \"translation\": \"Hola, ¿cómo estás?\", \"source_language\": \"en\", \"target_language\": \"es\"}"
test_endpoint "/verify" "POST" "$verification_data" "Final Verification Test"

# Calculate and display runtime
RUNTIME=$(($(date +%s) - START_TIME))
echo -e "${BOLD}${GREEN}Testing completed in $RUNTIME seconds!${NC}"
echo "All tests showed full JSON responses for server endpoints."
echo -e "${BOLD}${GREEN}All outputs were formatted for clarity and teaching use.${NC}"
exit 0