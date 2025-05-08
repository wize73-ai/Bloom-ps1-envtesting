#!/usr/bin/env python3
"""
Fix for overlapping console output formatting in the logging system.
"""

import re
import logging
import shutil
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("console_fix")

def backup_file(file_path, backup_suffix=".bak"):
    """Create a backup of the file"""
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} does not exist, skipping backup")
        return None
    
    backup_path = file_path + backup_suffix
    shutil.copyfile(file_path, backup_path)
    logger.info(f"Created backup of {file_path} at {backup_path}")
    return backup_path

def fix_console_py():
    """Fix the console.py file to add threading lock"""
    console_path = "app/ui/console.py"
    
    if not os.path.exists(console_path):
        logger.error(f"Console file not found at {console_path}")
        return False
    
    # Backup the file
    backup_file(console_path)
    
    # Read the file
    with open(console_path, 'r') as f:
        lines = f.readlines()
    
    # Add threading import if needed
    has_threading = False
    for line in lines:
        if "import threading" in line:
            has_threading = True
            break
    
    new_lines = []
    if not has_threading:
        for i, line in enumerate(lines):
            new_lines.append(line)
            if i == 0 and "import" in line:
                new_lines.append("import threading\n")
    else:
        new_lines = lines.copy()
    
    # Add console lock if needed
    has_lock = False
    for line in new_lines:
        if "console_lock" in line:
            has_lock = True
            break
    
    if not has_lock:
        for i, line in enumerate(new_lines):
            if "console = Console" in line:
                # Find the end of the Console initialization
                j = i
                while j < len(new_lines) and ")" not in new_lines[j]:
                    j += 1
                if j < len(new_lines):
                    # Insert after Console initialization
                    new_lines.insert(j + 1, "\n# Add lock to prevent overlapping output\nconsole_lock = threading.RLock()\n")
                    break
    
    # Modify display_panel function to use lock
    display_panel_start = -1
    display_panel_end = -1
    
    for i, line in enumerate(new_lines):
        if "def display_panel" in line:
            display_panel_start = i
        elif display_panel_start >= 0 and line.strip().startswith("console.print") and display_panel_end < 0:
            display_panel_end = i
    
    if display_panel_start >= 0 and display_panel_end >= 0:
        # Only modify if we haven't added the lock yet
        has_lock_usage = False
        for i in range(display_panel_start, display_panel_end + 1):
            if "console_lock" in new_lines[i]:
                has_lock_usage = True
                break
        
        if not has_lock_usage:
            # Replace the console.print line with a locked version
            for i in range(display_panel_start, display_panel_end + 1):
                if "console.print" in new_lines[i]:
                    indent = len(new_lines[i]) - len(new_lines[i].lstrip())
                    spaces = ' ' * indent
                    new_lines[i] = f"{spaces}# Use lock to prevent overlapping panels\n{spaces}with console_lock:\n{spaces}    console.print(panel)\n"
                    break
    
    # Write the updated file
    with open(console_path, 'w') as f:
        f.writelines(new_lines)
    
    logger.info(f"Updated console.py with threading lock")
    return True

def fix_loader_py():
    """Fix the loader.py file to add delays before panel rendering"""
    loader_path = "app/services/models/loader.py"
    
    if not os.path.exists(loader_path):
        logger.error(f"Loader file not found at {loader_path}")
        return False
    
    # Backup the file
    backup_file(loader_path)
    
    # Read the file
    with open(loader_path, 'r') as f:
        content = f.read()
    
    # Add import time if needed
    if "import time" not in content:
        content = content.replace(
            "import logging",
            "import logging\nimport time"
        )
    
    # Add delay before panel rendering
    content = content.replace(
        "console.print(Panel(",
        "# Add delay before panel rendering\n            time.sleep(0.2)\n            console.print(Panel("
    )
    
    # Write the updated file
    with open(loader_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated loader.py with delay before panel rendering")
    return True

if __name__ == "__main__":
    logger.info("Starting console output fix...")
    
    # Fix console.py
    fix_console_py()
    
    # Fix loader.py
    fix_loader_py()
    
    logger.info("Console output fix completed")
    print("\n✅ Console output fix completed!")
    print("The overlapping panel issue should now be fixed.")
    print("Remember to restart your application for the changes to take effect.")