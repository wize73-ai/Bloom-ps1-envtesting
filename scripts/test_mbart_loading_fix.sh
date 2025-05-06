#!/bin/bash

# Test script for MBART model loading fix
echo "Testing MBART model loading fix..."

# Set environment vars for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the test script
python test_mbart_loading_fix.py

# Check the return code
if [ $? -eq 0 ]; then
    echo "✅ Test passed: MBART model loaded successfully!"
else
    echo "❌ Test failed: MBART model loading issue persists."
fi

# Test the original translation script as well
echo "Testing MBART translation with original script..."
python test_mbart_translation.py

# Check the return code
if [ $? -eq 0 ]; then
    echo "✅ Test passed: MBART translation functions correctly!"
else
    echo "❌ Test failed: MBART translation issue persists."
fi

echo "Tests completed."