#!/usr/bin/env python3
"""
Script to fix and enhance the simplification functionality in CasaLingua.

This script copies the improved wrapper implementation to the correct location
and updates the necessary files to use the enhanced simplification.
"""

import os
import shutil
import sys

def main():
    """Main function to apply simplification fixes."""
    # Define the source and destination paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_fix_src = os.path.join(script_dir, "app/services/models/wrapper.py.fix")
    wrapper_dest = os.path.join(script_dir, "app/services/models/wrapper.py")
    
    # Check if the source file exists
    if not os.path.exists(wrapper_fix_src):
        print(f"Error: Source file {wrapper_fix_src} not found.")
        return 1
    
    # Create a backup of the original file
    if os.path.exists(wrapper_dest):
        backup_path = f"{wrapper_dest}.bak"
        print(f"Creating backup of {wrapper_dest} to {backup_path}")
        shutil.copy2(wrapper_dest, backup_path)
    
    # Copy the fixed implementation
    print(f"Copying enhanced wrapper implementation to {wrapper_dest}")
    shutil.copy2(wrapper_fix_src, wrapper_dest)
    
    # Make the wrapper.py file executable
    os.chmod(wrapper_dest, 0o755)
    
    print("Simplification enhancements have been applied successfully.")
    print("Please restart the application for changes to take effect.")
    return 0

if __name__ == "__main__":
    sys.exit(main())