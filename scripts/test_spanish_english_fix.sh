#!/bin/bash
# Test script for Spanish to English translation fixes

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Testing Spanish to English Translation Fix${NC}"
echo -e "${BLUE}============================================${NC}"

# Make the test script executable
chmod +x "$SCRIPT_DIR/test_spanish_english_translation.py"

# Run our Python test script
echo -e "${YELLOW}Running Python test script...${NC}"
python3 "$SCRIPT_DIR/test_spanish_english_translation.py"

# Also test via curl if server is running
echo -e "\n${YELLOW}Testing via API endpoint (if server is running)...${NC}"

# Test sentences
TEST_SENTENCES=(
  "Estoy muy feliz de conocerte hoy. El clima es hermoso y espero que estés bien."
  "Hola, ¿cómo estás?"
  "Me gusta aprender idiomas nuevos."
)

# Server URL (default to localhost:8000 but can be overridden)
SERVER_URL=${1:-"http://localhost:8000"}

for sentence in "${TEST_SENTENCES[@]}"; do
  echo -e "\n${YELLOW}Testing sentence:${NC} $sentence"
  
  # Prepare JSON data for the request
  JSON_DATA=$(cat <<EOF
{
  "text": "$sentence",
  "source_language": "es",
  "target_language": "en"
}
EOF
)

  # Try to send a request to the translation endpoint
  RESPONSE=$(curl -s -X POST \
    "$SERVER_URL/pipeline/translate" \
    -H "Content-Type: application/json" \
    -d "$JSON_DATA" 2>/dev/null)
  
  # Check if curl command succeeded
  if [ $? -eq 0 ] && [ -n "$RESPONSE" ]; then
    echo -e "${GREEN}API Response:${NC} $RESPONSE"
    
    # Check for Czech words in response
    if echo "$RESPONSE" | grep -q "Jsem\|velmi\|že\|vás\|poznávám\|dnes"; then
      echo -e "${RED}ERROR: Response contains Czech words!${NC}"
    else
      echo -e "${GREEN}SUCCESS: No Czech words detected in response${NC}"
    fi
  else
    echo -e "${RED}Failed to connect to server at $SERVER_URL${NC}"
    echo -e "${YELLOW}Is the server running?${NC}"
    break
  fi
done

echo -e "\n${BLUE}Testing complete!${NC}"