#!/bin/bash
# CasaLingua Robust Demo Runner
# This script first verifies that models are properly loaded
# before starting the demo to prevent triggering downloads

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PURE_DEMO_SCRIPT="${PROJECT_DIR}/pure_demo.py"
API_DEMO_SCRIPT="${PROJECT_DIR}/scripts/casalingua_api_demo.py"
VERIFIER_SCRIPT="${PROJECT_DIR}/ensure_models_loaded.py"

# Set environment variables to prevent automatic downloads
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "===== CasaLingua Demo Runner ====="
echo "Verifying system status before starting demo..."

# Install required packages for demo
pip install rich > /dev/null 2>&1

# Check if psutil is installed (needed for memory check)
if ! python -c "import psutil" &> /dev/null; then
    echo "Installing psutil for system verification..."
    pip install psutil > /dev/null 2>&1
fi

# Run the model verification script
echo "Checking model status..."
python "${VERIFIER_SCRIPT}" --check-downloads
VERIFY_STATUS=$?

# Determine which demo to run based on verification result
if [ $VERIFY_STATUS -eq 0 ]; then
    echo -e "${GREEN}All models verified! System appears stable for using larger models.${NC}"
    echo "Starting API-based demo in 3 seconds..."
    sleep 3
    
    # Check if API is running
    if curl -s http://localhost:8000/health &> /dev/null; then
        echo "API is running. Starting API-based demo..."
        python "${API_DEMO_SCRIPT}"
    else
        echo -e "${YELLOW}API is not running. Using simulation demo instead.${NC}"
        python "${PURE_DEMO_SCRIPT}"
    fi
else
    if [ $VERIFY_STATUS -eq 2 ]; then
        echo -e "${YELLOW}MBART model verification failed. System may try to download models during operation.${NC}"
    else
        echo -e "${YELLOW}System stability concerns detected. Using simulation to avoid potential issues.${NC}"
    fi
    
    echo "Would you like to:"
    echo "1) Run the pure simulation demo (safest option, no model loading)"
    echo "2) Run the API-based demo anyway (may trigger downloads)"
    echo "3) Cancel"
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            echo "Starting pure simulation demo..."
            python "${PURE_DEMO_SCRIPT}"
            ;;
        2)
            echo -e "${YELLOW}Warning: This may trigger model downloads if models aren't already cached.${NC}"
            echo "Starting API-based demo in 3 seconds..."
            sleep 3
            
            # Check if API is running first
            if curl -s http://localhost:8000/health &> /dev/null; then
                echo "API is running. Starting API-based demo..."
                # Disable offline mode to allow downloads if needed
                unset TRANSFORMERS_OFFLINE
                unset HF_DATASETS_OFFLINE
                python "${API_DEMO_SCRIPT}"
            else
                echo -e "${RED}Error: API is not running. Please start the API first.${NC}"
                exit 1
            fi
            ;;
        3)
            echo "Demo canceled."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Using pure simulation demo for safety.${NC}"
            python "${PURE_DEMO_SCRIPT}"
            ;;
    esac
fi

exit 0