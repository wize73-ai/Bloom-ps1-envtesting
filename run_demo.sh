#!/bin/bash
# CasaLingua Demo Runner - Simplified Version
# This script runs the CasaLingua API demo

echo "Starting CasaLingua demo..."

# Install required packages if needed
pip install rich requests

# Run the demo
python3 scripts/casalingua_api_demo.py

exit 0