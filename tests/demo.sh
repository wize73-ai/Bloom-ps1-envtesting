#!/bin/bash
# Ultra-simple CasaLingua demo runner

echo "Installing required packages..."
pip install rich

echo "Starting CasaLingua standalone demo..."
python ./standalone_demo.py

exit 0