#!/bin/bash

# Script to apply all CasaLingua fixes at once

echo "=================================="
echo "  CasaLingua System-wide Fix"
echo "=================================="
echo ""

# Make executable
chmod +x fix_processor_simplify.py
chmod +x fix_wrapper_text_input.py
chmod +x fix_simplifier_implementation.py

# Fix the processor first (add simplify_text method)
echo "Adding simplify_text method to processor..."
python fix_processor_simplify.py
if [ $? -ne 0 ]; then
    echo "Failed to add simplify_text method to processor. Aborting."
    exit 1
fi
echo ""

# Fix the wrapper input validation
echo "Adding type validation to translation wrapper..."
python fix_wrapper_text_input.py
if [ $? -ne 0 ]; then
    echo "Failed to add type validation to wrapper. Aborting."
    exit 1
fi
echo ""

# Fix the simplifier implementation
echo "Enhancing rule-based simplifier implementation..."
python fix_simplifier_implementation.py
if [ $? -ne 0 ]; then
    echo "Failed to enhance simplifier implementation. Aborting."
    exit 1
fi
echo ""

echo "All fixes applied successfully!"
echo ""
echo "Please restart the CasaLingua server for changes to take effect."
echo "You can restart it with:"
echo "  scripts/startdev.sh"
echo ""
echo "=================================="