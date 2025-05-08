#!/usr/bin/env python3
"""
Fix for the 'ModelConfig' object has no attribute 'get' error in TranslationModelWrapper.

This script addresses a compatibility issue between ModelConfig objects and dictionary access
patterns in the wrapper code.
"""

import os
import sys
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_wrapper_file():
    """Find the TranslationModelWrapper file."""
    base_dir = Path(".")
    wrapper_paths = list(base_dir.glob("**/wrapper.py"))
    wrapper_paths = [p for p in wrapper_paths if "app/services/models" in str(p)]
    
    if not wrapper_paths:
        logger.error("Could not find wrapper.py in app/services/models/")
        return None
        
    return wrapper_paths[0]

def create_backup(file_path):
    """Create a backup of the file."""
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak_config_fix")
    
    # Check if backup already exists
    if backup_path.exists():
        logger.info(f"Backup already exists at {backup_path}")
        return
    
    # Create backup
    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    
    logger.info(f"Created backup at {backup_path}")

def fix_model_config_usage(file_path):
    """Fix ModelConfig.get() usage in the TranslationModelWrapper class."""
    if not file_path or not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all instances where config.get() is used on possibly ModelConfig objects
    # and replace with safer access pattern
    
    # Pattern 1: self.config.get("key", default)
    pattern1 = r'self\.config\.get\("([^"]+)", ([^)]+)\)'
    replacement1 = r'(self.config.get("\1", \2) if isinstance(self.config, dict) else getattr(self.config, "\1", \2))'
    
    # Pattern 2: self.config.get("key")
    pattern2 = r'self\.config\.get\("([^"]+)"\)'
    replacement2 = r'(self.config.get("\1") if isinstance(self.config, dict) else getattr(self.config, "\1", None))'
    
    # Apply replacements
    updated_content = re.sub(pattern1, replacement1, content)
    updated_content = re.sub(pattern2, replacement2, updated_content)
    
    # Add a helper function at the top of the file (after imports)
    helper_function = """
def safe_config_get(config, key, default=None):
    '''Get a value from config, handling both dict and object access patterns.'''
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
"""
    
    # Insert after imports but before classes
    import_end = re.search(r'# Configure logging.*?\n', content)
    if import_end:
        insert_pos = import_end.end()
        updated_content = updated_content[:insert_pos] + helper_function + updated_content[insert_pos:]
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    logger.info(f"Updated {file_path} with safer config access patterns")
    return True

def main():
    """Main entry point."""
    logger.info("Fixing ModelConfig.get() usage in TranslationModelWrapper...")
    
    # Find the wrapper file
    wrapper_file = find_wrapper_file()
    if not wrapper_file:
        logger.error("Could not find wrapper file")
        return 1
    
    logger.info(f"Found wrapper file at: {wrapper_file}")
    
    # Create backup
    create_backup(wrapper_file)
    
    # Apply fix
    if fix_model_config_usage(wrapper_file):
        logger.info("✅ Successfully fixed ModelConfig.get() usage")
        return 0
    else:
        logger.error("❌ Failed to fix ModelConfig.get() usage")
        return 1

if __name__ == "__main__":
    sys.exit(main())