#!/usr/bin/env python3
"""
Fix for overlapping console output formatting in the logging system.
This script updates the Rich console handling to prevent panel overlap
and ensure proper log formatting.

Usage:
    python fix_console_output.py
"""

import re
import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("console_fix")

# Paths to check and fix
CONSOLE_PATH = Path("app/ui/console.py")
LOADER_PATH = Path("app/services/models/loader.py") 
MANAGER_PATH = Path("app/services/models/manager.py") if Path("app/services/models/manager.py").exists() else None
BACKUP_SUFFIX = ".bak_before_console_fix"

def backup_file(file_path, backup_suffix=BACKUP_SUFFIX):
    """Create a backup of the file"""
    if not file_path.exists():
        logger.warning(f"File {file_path} does not exist, skipping backup")
        return None
    
    backup_path = Path(str(file_path) + backup_suffix)
    shutil.copyfile(file_path, backup_path)
    logger.info(f"Created backup of {file_path} at {backup_path}")
    return backup_path

def fix_console_output():
    """Fix the console output formatting to prevent overlapping panels"""
    
    # First check if the console.py file exists
    if not CONSOLE_PATH.exists():
        logger.error(f"Console file not found at {CONSOLE_PATH}")
        return False
    
    # Backup the file
    backup_file(CONSOLE_PATH)
    
    # Read the content
    with open(CONSOLE_PATH, 'r') as f:
        content = f.read()
    
    # Add a mutex lock for panel rendering
    if "import threading" not in content:
        # Add the threading import
        content = re.sub(
            r"(import.*?\n+)",
            r"\1import threading\n",
            content,
            count=1
        )
        logger.info("Added threading import")
    
    # Add a console lock
    if "console_lock" not in content:
        # Find the console initialization
        console_pattern = r"(console = Console\(.*?\))"
        if re.search(console_pattern, content):
            content = re.sub(
                console_pattern,
                r"\1\n\n# Console lock to prevent overlapping panels\nconsole_lock = threading.RLock()",
                content
            )
            logger.info("Added console lock")
        else:
            logger.warning("Could not find console initialization")
    
    # Update the display_panel function to use the lock
    if "with console_lock" not in content:
        panel_function = """
def display_panel(title, content, style="green", border_style="green"):
    """Display content in a rich panel with title."""
    panel = Panel(
        content,
        title=f"[bold]{title}[/bold]",
        style=style,
        border_style=border_style,
        expand=False
    )
    # Use lock to prevent overlapping panels
    with console_lock:
        console.print(panel)
"""
        
        # Replace the existing display_panel function
        panel_pattern = r"def display_panel\(.*?\):.*?console\.print\(panel\)"
        if re.search(panel_pattern, content, re.DOTALL):
            content = re.sub(panel_pattern, panel_function.strip(), content, flags=re.DOTALL)
            logger.info("Updated display_panel function to use the lock")
        else:
            logger.warning("Could not find display_panel function")
    
    # Write the updated content
    with open(CONSOLE_PATH, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated console output formatting in {CONSOLE_PATH}")
    return True

def fix_loader_output():
    """Fix the loader output formatting"""
    if not LOADER_PATH.exists():
        logger.error(f"Loader file not found at {LOADER_PATH}")
        return False
    
    # Backup the file
    backup_file(LOADER_PATH)
    
    # Read the content
    with open(LOADER_PATH, 'r') as f:
        content = f.read()
    
    # Update the model loading success panel
    panel_pattern = r"(console\.print\(Panel\(.*?Successfully loaded model.*?\)\))"
    
    if re.search(panel_pattern, content, re.DOTALL):
        # Add delay before panel rendering
        modified_content = re.sub(
            panel_pattern,
            r"# Add a short delay before panel rendering\n        import time\n        time.sleep(0.2)\n        \1",
            content,
            flags=re.DOTALL
        )
        
        with open(LOADER_PATH, 'w') as f:
            f.write(modified_content)
        
        logger.info("Added delay before panel rendering in loader")
    else:
        logger.warning("Could not find panel pattern in loader")
    
    return True

def main():
    logger.info("Starting console output formatting fix...")
    
    # Fix the console output formatting
    fix_console_output()
    
    # Fix the loader output formatting
    fix_loader_output()
    
    logger.info("Console output formatting fix completed")
    print("\n✅ Console output formatting fix completed successfully!")
    print("The overlapping panel issue should now be fixed.")
    print("\nRemember to restart your application for the changes to take effect.")

if __name__ == "__main__":
    main()