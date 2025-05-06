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
    
    echo -e "${BOLD}${BLUE}$description${NC}"
    echo -e "${BLUE}Endpoint: $method $endpoint${NC}"
    
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
        echo -e "${GREEN}Response (JSON):${NC}"
        pretty_print_json "$result"
    else
        echo -e "${YELLOW}Response (not JSON):${NC}"
        echo "$result"
    fi
    
    echo
    return 0
}

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

# Check if server is available
echo -e "${BLUE}Checking if CasaLingua server is running...${NC}"
if ! curl -s "$SERVER_URL/health" &> /dev/null; then
    echo -e "${RED}Error: CasaLingua server is not running!${NC}"
    echo "Please start the server first with:"
    echo "python -m app.main"
    echo ""
    echo "Would you like to:"
    echo "1) Start the server for me (in a new terminal)"
    echo "2) Run pure simulation demo instead"
    echo "3) Exit"
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            echo "Starting CasaLingua server in a new terminal..."
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && python -m app.main"'
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                # Linux with X11
                if command -v gnome-terminal &> /dev/null; then
                    gnome-terminal -- bash -c "cd $(pwd) && python -m app.main; exec bash"
                elif command -v xterm &> /dev/null; then
                    xterm -e "cd $(pwd) && python -m app.main; exec bash" &
                else
                    echo "Could not start a new terminal automatically."
                    echo "Please start the server manually and try again."
                    exit 1
                fi
            else
                echo "Could not start a new terminal automatically."
                echo "Please start the server manually and try again."
                exit 1
            fi
            
            echo "Waiting for server to start..."
            for i in {1..30}; do
                if curl -s "$SERVER_URL/health" &> /dev/null; then
                    echo -e "${GREEN}Server is now running!${NC}"
                    break
                fi
                echo -n "."
                sleep 1
                if [ $i -eq 30 ]; then
                    echo -e "${RED}Timed out waiting for server to start.${NC}"
                    echo "Running simulation demo instead."
                    python ./pure_demo.py
                    exit 0
                fi
            done
            ;;
        2)
            echo "Starting pure simulation demo..."
            python ./pure_demo.py
            exit 0
            ;;
        3)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
else
    echo -e "${GREEN}CasaLingua server is running!${NC}"
fi

# Test health endpoint with full JSON display
test_endpoint "/health" "GET" "" "Testing Health Endpoint"

# Test detailed health endpoint with full JSON display
test_endpoint "/health/detailed" "GET" "" "Testing Detailed Health Endpoint"

# Test health metrics endpoint with full JSON display
test_endpoint "/health/metrics" "GET" "" "Testing Health Metrics Endpoint"

# Test translation endpoint with a simple English to Spanish translation
translation_data='{"text": "Hello, how are you?", "source_language": "en", "target_language": "es", "preserve_formatting": true}'
test_endpoint "/pipeline/translate" "POST" "$translation_data" "Testing Translation Endpoint (EN->ES)"

# Test simplification endpoint with a complex sentence
simplification_data='{"text": "The aforementioned contractual obligations shall be considered null and void if the party of the first part fails to remit payment within the specified timeframe.", "language": "en", "level": 5}'
test_endpoint "/pipeline/simplify" "POST" "$simplification_data" "Testing Simplification Endpoint"

# Verification endpoint test
verification_data='{"source_text": "Hello, how are you?", "translation": "Hola, ¿cómo estás?", "source_language": "en", "target_language": "es"}'
test_endpoint "/verify" "POST" "$verification_data" "Testing Verification Endpoint"

# Run the main server demo
echo -e "${BOLD}${GREEN}All tests completed. Running the full server demo...${NC}"
echo "The demo will continue for 2 minutes, showing all features of the CasaLingua server."
echo

# Run the server demo with a simplified output option
if [ "$1" == "--simple" ]; then
    python ./simple_server_demo.py
else
    python ./server_demo.py
fi

echo -e "${BOLD}${GREEN}Server testing complete!${NC}"
exit 0