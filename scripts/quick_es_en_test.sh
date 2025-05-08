#!/bin/bash
# Quick test for Spanish to English translation

# Set server URL (default to localhost:8000)
SERVER_URL=${1:-"http://localhost:8000"}
ENDPOINT="/pipeline/translate"

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing Spanish to English Translation API${NC}"
echo -e "${YELLOW}Using server:${NC} $SERVER_URL$ENDPOINT"

# Our test sentence - the one with issues
TEST_SENTENCE="Estoy muy feliz de conocerte hoy. El clima es hermoso y espero que estés bien."

echo -e "\n${YELLOW}Test input:${NC} $TEST_SENTENCE"

# Prepare JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
  "text": "$TEST_SENTENCE",
  "source_language": "es",
  "target_language": "en"
}
EOF
)

# Send request using curl
echo -e "\n${YELLOW}Sending request...${NC}"

RESULT=$(curl -s -X POST \
  "$SERVER_URL$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "$JSON_PAYLOAD")

# Extract translated text from response (assumes JSON response)
TRANSLATED_TEXT=$(echo $RESULT | grep -o '"translated_text":"[^"]*"' | sed 's/"translated_text":"//g' | sed 's/"//g')

echo -e "\n${GREEN}Response:${NC}"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

echo -e "\n${GREEN}Extracted translation:${NC} $TRANSLATED_TEXT"

# Check for Czech words that would indicate the bug is still present
if echo "$TRANSLATED_TEXT" | grep -q "Jsem\|velmi\|že\|vás\|poznávám\|dnes"; then
  echo -e "${RED}ERROR: Response contains Czech words!${NC}"
else
  echo -e "${GREEN}SUCCESS: No Czech words detected!${NC}"
fi

# Check if translation is empty
if [ -z "$TRANSLATED_TEXT" ]; then
  echo -e "${RED}ERROR: Translation is empty!${NC}"
fi

echo -e "\n${BLUE}Test complete!${NC}"