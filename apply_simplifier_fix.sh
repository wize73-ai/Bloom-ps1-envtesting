#!/bin/bash

# Script to apply simplifier fix only

echo "====================================="
echo "  CasaLingua Simplifier Fix"
echo "====================================="
echo ""

# Run enhanced simplifier directly
echo "Applying enhanced simplifier implementation..."
python enhanced_simplifier.py

if [ $? -ne 0 ]; then
    echo "Failed to apply enhanced simplifier. Aborting."
    exit 1
fi

echo ""
echo "Simplifier fix applied successfully!"
echo ""
echo "Please restart the CasaLingua server for changes to take effect."
echo "You can restart it with:"
echo "  scripts/startdev.sh"
echo ""
echo "====================================="