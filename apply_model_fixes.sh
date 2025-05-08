#!/bin/bash
# Script to apply all model loading fixes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Applying model loading fixes..."

# Apply main fixes
python scripts/fix_translation_models.py

# Apply MT5 loader fix
python scripts/fixes/fix_mt5_loader.py

# Apply NLLB wrapper patch
python scripts/fixes/nllb_wrapper_patch.py

echo "✅ All fixes applied successfully"
echo ""
echo "Please restart the server to apply changes."
echo "To test NLLB translation, run:"
echo "python scripts/test_nllb_translation.py"