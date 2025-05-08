#!/bin/bash
# Apply all speech processing fixes and run tests

# Text colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  CASALINGUA SPEECH PROCESSING FIXES     ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Create temp directory if not exists
mkdir -p temp logs

# Step 1: Apply basic TTS/STT fixes
echo -e "\n${YELLOW}Step 1: Applying TTS/STT endpoint fixes...${NC}"
python scripts/fix_speech_methods.py

# Step 2: Harden the endpoints
echo -e "\n${YELLOW}Step 2: Hardening speech endpoints...${NC}"
python scripts/harden_speech_endpoints.py

# Step 3: Install TTS requirements
echo -e "\n${YELLOW}Step 3: Installing TTS requirements...${NC}"
python scripts/install_tts_requirements.py

# Step 4: Create emergency audio files
echo -e "\n${YELLOW}Step 4: Creating emergency audio files...${NC}"
python scripts/emergency_tts_fix.py

# Step 5: Restart the server (if running)
echo -e "\n${YELLOW}Step 5: Checking server status...${NC}"
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    echo "Server is running, restarting..."
    pkill -f "uvicorn app.main:app"
    sleep 2
    nohup uvicorn app.main:app --reload > logs/server.log 2>&1 &
    echo "Server restarted, waiting for initialization..."
    sleep 10
else
    echo "Server not running, starting..."
    nohup uvicorn app.main:app --reload > logs/server.log 2>&1 &
    echo "Server started, waiting for initialization..."
    sleep 10
fi

# Step 6: Run tests with monitoring
echo -e "\n${YELLOW}Step 6: Running tests with monitoring...${NC}"
python scripts/monitor_speech_processing.py

echo -e "\n${BLUE}=========================================${NC}"
echo -e "${BLUE}  FIXES AND TESTS COMPLETE                ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "\nThe speech processing functionality is now ready to use!"
echo -e "Documentation available at: docs/api/speech-processing.md"
echo -e "\nTo test individually:"
echo -e "  - TTS endpoint: ./scripts/test_tts_curl.sh"
echo -e "  - All endpoints: python scripts/test_all_speech_endpoints.py"
echo -e "  - Complete workflow: python scripts/test_speech_workflow.py"