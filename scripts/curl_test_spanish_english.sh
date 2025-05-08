#!/bin/bash
# Simple curl script to test Spanish to English translation

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Server URL (default to localhost:8000 but can be overridden)
SERVER_URL=${1:-"http://localhost:8000"}

# Test sentence (the problematic one)
TEST_SENTENCE="Estoy muy feliz de conocerte hoy. El clima es hermoso y espero que estés bien."

echo -e "${BLUE}Testing Spanish to English Translation API${NC}"
echo -e "${YELLOW}Server:${NC} $SERVER_URL"
echo -e "${YELLOW}Test sentence:${NC} $TEST_SENTENCE"

# Prepare JSON data for the request
JSON_DATA=$(cat <<EOF
{
  "text": "$TEST_SENTENCE",
  "source_language": "es",
  "target_language": "en"
}
EOF
)

echo -e "\n${YELLOW}Sending request...${NC}"

# Send request to the translation endpoint
curl -v -X POST \
  "$SERVER_URL/pipeline/translate" \
  -H "Content-Type: application/json" \
  -d "$JSON_DATA"

echo -e "\n\n${GREEN}Request completed${NC}"