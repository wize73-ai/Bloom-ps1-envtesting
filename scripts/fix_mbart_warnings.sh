#!/bin/bash
# Script to fix MBART tokenizer warnings and verify the fix

set -e  # Exit on error

# Output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Display header
echo "====================================="
echo "MBART Tokenizer Warning Fix Utility"
echo "====================================="
echo

# Apply the fix
echo "Applying the MBART tokenizer fix..."
python3 "$PROJECT_ROOT/fix_mbart_tokenizer.py"
echo

# Test the fix
echo "Verifying the fix..."
python3 "$SCRIPT_DIR/test_mbart_tokenizer_fix.py"
echo

# Success message
if [ $? -eq 0 ]; then
    echo "====================================="
    echo "✅ Fix successfully applied and tested"
    echo "====================================="
    echo
    echo "The MBART tokenizer warnings should no longer appear in your logs."
    echo "To verify in production, please restart your server and check the logs."
else
    echo "====================================="
    echo "❌ Fix verification failed"
    echo "====================================="
    echo
    echo "Please check the error messages above for details."
fi