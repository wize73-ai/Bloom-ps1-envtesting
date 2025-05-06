#!/bin/bash
# Script to apply fixes for the simplification wrapper and model loading issues

echo "=== Fixing Model Wrapper Issues ==="
echo

# Make the script executable
chmod +x app/services/models/wrapper_update.py
chmod +x test_model_wrapper_improvements.py

# Apply the wrapper fixes
echo "Applying wrapper fixes..."
python3 app/services/models/wrapper_update.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to apply wrapper fixes"
    exit 1
fi

echo
echo "=== Testing Fixes ==="
echo

# Run the test script
echo "Running tests..."
python3 test_model_wrapper_improvements.py

if [ $? -ne 0 ]; then
    echo "Error: Tests failed"
    exit 1
fi

echo
echo "=== Summary of Fixes ==="
echo "1. Fixed SimplifierWrapper implementation with robust rule-based fallback"
echo "2. Fixed 'NoneType' object has no attribute 'lower' error in domain handling"
echo "3. Enhanced model type detection to prevent reloading models"
echo "4. Improved memory pressure tracking for better monitoring"
echo "5. Added better error handling for more reliable operation"
echo
echo "All fixes have been successfully applied and tested."
echo