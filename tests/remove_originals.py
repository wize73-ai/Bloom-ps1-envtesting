#!/usr/bin/env python3
"""
Script to remove original test files from root directory
after confirming they exist in the tests directory.
"""
import os
import sys
from pathlib import Path

def main():
    """Remove original test files that have been migrated"""
    root_dir = Path(__file__).parent.parent
    tests_dir = Path(__file__).parent
    
    # Find all migrated Python test files
    migrated_py_files = set()
    for path in tests_dir.glob("**/*.py"):
        if path.name.startswith("test_") and path.name not in ["test_runner.py", "migrate_tests.py", "remove_originals.py", "check_imports.py"]:
            migrated_py_files.add(path.name)
    
    # Find all migrated shell scripts
    migrated_sh_files = set()
    for path in (tests_dir / "scripts").glob("*.sh"):
        migrated_sh_files.add(path.name)
    
    print(f"Found {len(migrated_py_files)} migrated Python test files")
    print(f"Found {len(migrated_sh_files)} migrated shell scripts")
    
    # Find Python tests in root directory that can be removed
    to_remove_py = []
    for path in root_dir.glob("test_*.py"):
        if path.name in migrated_py_files:
            to_remove_py.append(path)
    
    # Find Python tests in scripts directory that can be removed
    for path in (root_dir / "scripts").glob("test_*.py"):
        if path.name in migrated_py_files:
            to_remove_py.append(path)
    
    # Find shell scripts that can be removed
    to_remove_sh = []
    for path in root_dir.glob("*.sh"):
        if path.name in migrated_sh_files:
            to_remove_sh.append(path)
    
    # Find shell scripts in scripts directory that can be removed
    for path in (root_dir / "scripts").glob("*.sh"):
        if path.name in migrated_sh_files:
            to_remove_sh.append(path)
    
    print(f"Found {len(to_remove_py)} Python test files to remove")
    print(f"Found {len(to_remove_sh)} shell scripts to remove")
    
    # Get confirmation before removing
    confirmation = input("Do you want to proceed with removing these files? (yes/no): ")
    if confirmation.lower() != "yes":
        print("Aborted. No files were removed.")
        return 1
    
    # Remove files
    removed_count = 0
    for path in to_remove_py + to_remove_sh:
        try:
            os.remove(path)
            print(f"Removed: {path}")
            removed_count += 1
        except Exception as e:
            print(f"Error removing {path}: {str(e)}")
    
    print(f"\nRemoved {removed_count} files successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())