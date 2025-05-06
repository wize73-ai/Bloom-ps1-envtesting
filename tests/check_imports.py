#!/usr/bin/env python3
"""
Script to check and fix import paths in the migrated test files
"""
import os
import re
import sys
from pathlib import Path

def check_imports(file_path):
    """Check if imports in the file need to be updated"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Look for relative imports that might need updating
    relative_imports = re.findall(r'from\s+\.\.?[a-zA-Z0-9_.]+\s+import', content)
    
    if relative_imports:
        print(f"Found potential relative imports in {file_path}:")
        for imp in relative_imports:
            print(f"  - {imp}")
        return True
    
    return False

def main():
    tests_dir = Path(__file__).parent
    
    # Find all Python test files in the tests directory
    test_files = []
    for path in tests_dir.glob("**/*.py"):
        if path.name != "__init__.py" and path.name != "conftest.py" and path != Path(__file__):
            test_files.append(path)
    
    # Check imports in each file
    needs_fixing = []
    for path in test_files:
        if check_imports(path):
            needs_fixing.append(path)
    
    # Print summary
    print("\nImport Check Summary:")
    print(f"- {len(test_files)} test files checked")
    print(f"- {len(needs_fixing)} files may need import path fixes")
    
    if needs_fixing:
        print("\nFiles that may need import fixes:")
        for path in needs_fixing:
            print(f"  - {path.relative_to(tests_dir.parent)}")
    else:
        print("\nAll imports appear to be absolute - no changes needed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())