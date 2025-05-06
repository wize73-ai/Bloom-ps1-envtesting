#!/bin/bash
# CasaLingua Demo Runner
# This script runs the CasaLingua demo in a new terminal window

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_SCRIPT="${PROJECT_DIR}/scripts/casalingua_api_demo.py"

# Check if python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found."
    exit 1
fi

# Check if the demo script exists
if [ ! -f "$DEMO_SCRIPT" ]; then
    echo "Error: Demo script not found at $DEMO_SCRIPT"
    exit 1
fi

# Check if rich library is installed
if ! python3 -c "import rich" &> /dev/null; then
    echo "Installing required dependencies..."
    pip install rich requests
fi

# Check if API server is running
if ! curl -s http://localhost:8000/health &> /dev/null; then
    echo "Warning: CasaLingua API doesn't appear to be running."
    echo "Starting API server in a new terminal window..."
    # Start the API server in a new terminal window based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        osascript -e 'tell app "Terminal" to do script "cd '${PROJECT_DIR}' && source venv/bin/activate && python -m app.main"'
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux with X11
        if command -v gnome-terminal &> /dev/null; then
            gnome-terminal -- bash -c "cd ${PROJECT_DIR} && source venv/bin/activate && python -m app.main; exec bash"
        elif command -v xterm &> /dev/null; then
            xterm -e "cd ${PROJECT_DIR} && source venv/bin/activate && python -m app.main; exec bash" &
        else
            echo "Please start the API server manually with: python -m app.main"
            echo "Then run this script again."
            exit 1
        fi
    else
        echo "Please start the API server manually with: python -m app.main"
        echo "Then run this script again."
        exit 1
    fi
    
    # Wait for API to start
    echo "Waiting for API to start..."
    for i in {1..10}; do
        if curl -s http://localhost:8000/health &> /dev/null; then
            echo "API server is now running."
            break
        fi
        sleep 2
        echo -n "."
        if [ $i -eq 10 ]; then
            echo "Timed out waiting for API to start."
            echo "Please check the API server window for errors."
            echo "Continuing anyway, but the demo might not work correctly."
        fi
    done
fi

# Run the demo
echo "Starting CasaLingua demo..."
cd "$PROJECT_DIR"
source venv/bin/activate
python3 "$DEMO_SCRIPT"

exit 0