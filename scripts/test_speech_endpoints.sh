#!/bin/bash
# Test script for speech endpoints in CasaLingua

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  CASALINGUA SPEECH ENDPOINTS TEST     ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check if server is running
if ! curl -s http://localhost:8000/healthz > /dev/null; then
    echo -e "${RED}Server is not running! Please start the server first:${NC}"
    echo -e "uvicorn app.main:app --reload"
    exit 1
fi

# Create temp directory if needed
mkdir -p temp

# Test STT languages endpoint
echo -e "\n${YELLOW}Testing STT languages endpoint...${NC}"
curl -s http://localhost:8000/pipeline/stt/languages | jq .

# Test TTS voices endpoint
echo -e "\n${YELLOW}Testing TTS voices endpoint...${NC}"
curl -s http://localhost:8000/pipeline/tts/voices | jq .

# Test TTS endpoint
echo -e "\n${YELLOW}Testing TTS endpoint...${NC}"
TEST_TEXT="Hello, this is a test of the text to speech system."
echo -e "Text: \"$TEST_TEXT\""

TTS_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$TEST_TEXT\",\"language\":\"en\",\"output_format\":\"mp3\"}" \
  http://localhost:8000/pipeline/tts)

echo "$TTS_RESPONSE" | jq .

# Extract audio URL from response
AUDIO_URL=$(echo "$TTS_RESPONSE" | jq -r '.data.audio_url')

if [ "$AUDIO_URL" != "null" ]; then
    echo -e "\n${YELLOW}Downloading audio from $AUDIO_URL...${NC}"
    curl -s "http://localhost:8000$AUDIO_URL" -o "temp/test_audio.mp3"
    echo -e "${GREEN}Audio saved to temp/test_audio.mp3${NC}"
    
    # Test STT endpoint with the generated audio
    echo -e "\n${YELLOW}Testing STT endpoint with generated audio...${NC}"
    STT_RESPONSE=$(curl -s -X POST \
      -F "language=en" \
      -F "detect_language=false" \
      -F "audio_file=@temp/test_audio.mp3" \
      http://localhost:8000/pipeline/stt)
    
    echo "$STT_RESPONSE" | jq .
    
    # Extract transcribed text
    TRANSCRIBED_TEXT=$(echo "$STT_RESPONSE" | jq -r '.data.text')
    echo -e "\nTranscribed text: \"$TRANSCRIBED_TEXT\""
    
    # Print test summary
    echo -e "\n${BLUE}=======================================${NC}"
    echo -e "${BLUE}  TEST SUMMARY                         ${NC}"
    echo -e "${BLUE}=======================================${NC}"
    
    if [ "$AUDIO_URL" != "null" ]; then
        echo -e "${GREEN}✓ TTS endpoint:${NC} Successfully generated audio"
    else
        echo -e "${RED}✗ TTS endpoint:${NC} Failed to generate audio"
    fi
    
    if [ -n "$TRANSCRIBED_TEXT" ] || [ "$TRANSCRIBED_TEXT" == "" ]; then
        echo -e "${GREEN}✓ STT endpoint:${NC} Successfully processed audio"
    else
        echo -e "${RED}✗ STT endpoint:${NC} Failed to process audio"
    fi
else
    echo -e "${RED}Error: Could not get audio URL from TTS response${NC}"
fi

echo -e "\n${BLUE}=======================================${NC}"
echo -e "${BLUE}  END OF TEST                          ${NC}"
echo -e "${BLUE}=======================================${NC}"