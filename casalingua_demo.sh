#!/bin/bash
# CasaLingua Demonstration Script
# Carefully tests models and runs a stable 2-minute demo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   CasaLingua Demonstration Script   ${NC}"
echo -e "${BLUE}======================================${NC}"

# First, check if the server is running
echo -e "\nChecking if CasaLingua server is running..."
if ! curl -s http://localhost:8000/health &> /dev/null; then
    echo -e "${RED}Error: CasaLingua server is not running!${NC}"
    echo "Please start the server in a separate terminal with:"
    echo "  python -m app.main"
    echo -e "\nWould you like to:"
    echo "1) Continue with simulation mode (no server needed)"
    echo "2) Exit"
    read -p "Enter your choice (1-2): " choice
    
    if [ "$choice" == "1" ]; then
        echo -e "${YELLOW}Starting simulation demo...${NC}"
        python ./pure_demo.py
        exit 0
    else
        echo "Exiting."
        exit 1
    fi
fi

# Server is running, now verify models
echo -e "${GREEN}Server is running!${NC}"
echo -e "\nStep 1: Verifying models are properly loaded..."
echo "This step will ensure all required models are working correctly."
echo "This helps prevent server crashes during the demo."

# Run model verification with a 10-second wait between tests
python ./verify_models.py -w 10
VERIFY_STATUS=$?

if [ $VERIFY_STATUS -ne 0 ]; then
    echo -e "${YELLOW}Warning: Some models did not verify successfully.${NC}"
    echo -e "Continue anyway? Model errors may occur during the demo."
    read -p "Continue? (y/n): " continue_choice
    
    if [ "$continue_choice" != "y" ]; then
        echo "Exiting."
        exit 1
    fi
else
    echo -e "${GREEN}All models verified successfully!${NC}"
fi

# Now run the actual demo
echo -e "\nStep 2: Starting CasaLingua demonstration..."
echo "This demo will run for approximately 2 minutes."
echo "It will showcase translation, simplification, and other features."
echo -e "${YELLOW}Press Ctrl+C at any time to stop the demo.${NC}\n"

python ./minimal_demo.py

echo -e "\n${GREEN}Demo completed!${NC}"
exit 0