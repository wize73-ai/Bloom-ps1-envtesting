#!/usr/bin/env python3
"""
Script to migrate test files to the appropriate test directories
"""
import os
import sys
import shutil
from pathlib import Path

# Define test categories
CATEGORIES = {
    # Model-related tests
    "model": ["test_mbart_", "test_mt5_", "test_model_wrapper", "test_transformers_"],
    
    # Pipeline-related tests
    "pipeline": ["test_pipeline", "test_language_detection", "test_translation", 
                 "test_simplify", "test_veracity", "test_metrics"],
    
    # API-related tests
    "api": ["test_api_", "test_endpoints", "test_health_", "test_single_endpoint"],
    
    # Integration tests
    "integration": ["test_enhanced_", "test_direct_"]
}

def identify_test_type(filename):
    """Identify the type of test based on the filename"""
    # Integration tests (tests that involve multiple components)
    for pattern in CATEGORIES["integration"]:
        if pattern in filename:
            return "integration"
    
    # API tests
    for pattern in CATEGORIES["api"]:
        if pattern in filename:
            return "integration/api"
    
    # Model tests
    for pattern in CATEGORIES["model"]:
        if pattern in filename:
            return "unit/models"
    
    # Pipeline tests
    for pattern in CATEGORIES["pipeline"]:
        if pattern in filename:
            return "unit/pipeline"
    
    # Default to unit
    return "unit"

def main():
    root_dir = Path(__file__).parent.parent
    tests_dir = Path(__file__).parent
    
    # Find all Python test files not already in the tests directory
    test_files = []
    for path in root_dir.glob("test_*.py"):
        if "venv" not in str(path) and "tests" not in str(path):
            test_files.append(path)
    
    # Find all .sh files (script files)
    script_files = []
    for path in root_dir.glob("*.sh"):
        if "venv" not in str(path) and "tests" not in str(path):
            script_files.append(path)
    
    # Also check the scripts directory
    for path in (root_dir / "scripts").glob("test_*.py"):
        if "venv" not in str(path) and "tests" not in str(path):
            test_files.append(path)
    
    for path in (root_dir / "scripts").glob("*.sh"):
        if "venv" not in str(path) and "tests" not in str(path):
            script_files.append(path)
    
    # Move test files to the appropriate directories
    moved_test_files = []
    for path in test_files:
        test_type = identify_test_type(path.name)
        target_dir = tests_dir / test_type
        os.makedirs(target_dir, exist_ok=True)
        target_path = target_dir / path.name
        
        print(f"Moving {path.name} to {target_dir}")
        shutil.copy2(path, target_path)
        moved_test_files.append((path, target_path))
    
    # Move script files to the scripts directory
    moved_script_files = []
    for path in script_files:
        target_dir = tests_dir / "scripts"
        target_path = target_dir / path.name
        
        print(f"Moving {path.name} to {target_dir}")
        shutil.copy2(path, target_path)
        moved_script_files.append((path, target_path))
    
    # Print summary
    print("\nMigration Summary:")
    print(f"- {len(moved_test_files)} test files moved")
    print(f"- {len(moved_script_files)} script files moved")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())