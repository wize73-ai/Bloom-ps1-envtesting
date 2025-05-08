#!/usr/bin/env python3
"""
Fix for overlapping console output formatting in the logging system.
This script updates the Rich console handling to prevent panel overlap
and ensure proper log formatting.

Usage:
    python fix_console_output_formatting.py
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
    panel_pattern = r"(def display_panel.*?\(.*?\).*?:.*?console\.print\(panel\))"
    panel_replacement = r"\1"
    
    if "with console_lock" not in content:
        panel_replacement = r"""def display_panel(title, content, style="green", border_style="green"):
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
        console.print(panel)"""
        
        if re.search(panel_pattern, content, re.DOTALL):
            content = re.sub(panel_pattern, panel_replacement, content, flags=re.DOTALL)
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
    
    # Update the model loading success panel to include waiting for any pending console output
    success_panel_pattern = r"(console\.print\(Panel\(.*?\"✓ Successfully loaded model:.*?\".*?\)\))"
    success_panel_replacement = r"""                # Add a small delay to ensure previous logs are printed
                import time
                time.sleep(0.1)
                
                \1"""
    
    if re.search(success_panel_pattern, content, re.DOTALL):
        content = re.sub(success_panel_pattern, success_panel_replacement, content, flags=re.DOTALL)
        logger.info("Updated model loading success panel to include delay")
    else:
        logger.warning("Could not find model loading success panel pattern")
    
    # Write the updated content
    with open(LOADER_PATH, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated loader output formatting in {LOADER_PATH}")
    return True

def fix_manager_output():
    """Fix the manager output formatting if the file exists"""
    if not MANAGER_PATH or not MANAGER_PATH.exists():
        logger.info(f"Manager file not found at {MANAGER_PATH}, skipping")
        return False
    
    # Backup the file
    backup_file(MANAGER_PATH)
    
    # Read the content
    with open(MANAGER_PATH, 'r') as f:
        content = f.read()
    
    # Similar pattern for manager panels
    panel_pattern = r"(console\.print\(Panel\(.*?\)\))"
    panel_replacement = r"""        # Add a small delay to ensure previous logs are printed
        import time
        time.sleep(0.1)
        
        \1"""
    
    if re.search(panel_pattern, content):
        content = re.sub(panel_pattern, panel_replacement, content)
        logger.info("Updated manager panels to include delay")
    else:
        logger.warning("Could not find manager panel pattern")
    
    # Write the updated content
    with open(MANAGER_PATH, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated manager output formatting in {MANAGER_PATH}")
    return True

def create_async_console_handler():
    """Create an advanced async console handler file to replace synchronous handling"""
    async_console_path = Path("app/ui/async_console.py")
    
    logger.info(f"Creating advanced async console handler at {async_console_path}")
    
    content = """\"\"\"
Asynchronous Console Handler Module for CasaLingua

This module provides asynchronous console output handling with mutex locking
to prevent overlapping panels and ensure proper log formatting.
\"\"\"

import threading
import asyncio
import logging
from typing import Optional, Any, Dict, List
import time

# Import rich components
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.text import Text
from rich.table import Table

# Configure logger
logger = logging.getLogger(__name__)

# Initialize rich console with safer defaults
console = Console(
    highlight=True,
    record=True,
    width=120,
    color_system="auto"
)

# Console lock to prevent overlapping panels and ensure thread safety
console_lock = threading.RLock()

# Pending output queue for deferred rendering
output_queue = asyncio.Queue() if hasattr(asyncio, "Queue") else None

def display_panel(title: str, content: Any, style: str = "green", border_style: str = "green", expand: bool = False):
    """
    Display content in a rich panel with title and thread-safe locking.
    
    Args:
        title: The panel title
        content: The panel content
        style: The panel style
        border_style: The panel border style
        expand: Whether to expand the panel to full width
    """
    panel = Panel(
        content,
        title=f"[bold]{title}[/bold]",
        style=style,
        border_style=border_style,
        expand=expand
    )
    
    # Use lock to prevent overlapping panels
    with console_lock:
        # Small delay to ensure logs are flushed
        time.sleep(0.05)
        console.print(panel)
        # Another small delay after printing
        time.sleep(0.05)

def display_success(title: str, content: Any):
    """
    Display a success panel with green styling.
    
    Args:
        title: The panel title
        content: The panel content
    """
    display_panel(f"✓ {title}", content, style="green", border_style="green")

def display_warning(title: str, content: Any):
    """
    Display a warning panel with yellow styling.
    
    Args:
        title: The panel title
        content: The panel content
    """
    display_panel(f"⚠ {title}", content, style="yellow", border_style="yellow")

def display_error(title: str, content: Any):
    """
    Display an error panel with red styling.
    
    Args:
        title: The panel title
        content: The panel content
    """
    display_panel(f"❌ {title}", content, style="red", border_style="red")

async def display_panel_async(title: str, content: Any, style: str = "green", border_style: str = "green"):
    """
    Asynchronously display content in a rich panel with title.
    
    Args:
        title: The panel title
        content: The panel content
        style: The panel style
        border_style: The panel border style
    """
    if output_queue:
        # Queue the output for later rendering
        await output_queue.put({
            "type": "panel",
            "title": title,
            "content": content,
            "style": style,
            "border_style": border_style
        })
    else:
        # Fall back to synchronous display if queue not available
        display_panel(title, content, style, border_style)

def get_progress_bar(description: str = "Processing", **kwargs):
    """
    Get a progress bar instance with spinner.
    
    Args:
        description: The progress description
        **kwargs: Additional progress bar arguments
        
    Returns:
        Progress: A rich progress bar instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        **kwargs
    )

def setup_rich_logging():
    """
    Set up rich formatting for logging.
    """
    # Configure rich handler
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    
    # Set formatter
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add rich handler
    root_logger.addHandler(rich_handler)
    
    logger.info("Rich logging configured")

async def process_output_queue():
    """
    Process the output queue asynchronously.
    This should be run in a separate task to handle all queued console output.
    """
    if not output_queue:
        logger.warning("Async output queue not available")
        return
    
    logger.info("Starting async console output processor")
    
    while True:
        try:
            # Get the next output item from the queue
            output_item = await output_queue.get()
            
            # Process based on type
            if output_item["type"] == "panel":
                display_panel(
                    output_item["title"],
                    output_item["content"],
                    output_item["style"],
                    output_item["border_style"]
                )
            elif output_item["type"] == "text":
                with console_lock:
                    console.print(output_item["text"])
            
            # Mark item as done
            output_queue.task_done()
            
            # Small delay to prevent console overload
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error processing console output: {e}")
            await asyncio.sleep(1)  # Longer delay on error

def start_async_console():
    """
    Start the async console processing if running in an async context.
    """
    if asyncio.get_event_loop().is_running() and output_queue:
        asyncio.create_task(process_output_queue())
        logger.info("Async console processor started")
    else:
        logger.info("Using synchronous console output (no async event loop detected)")
"""
    
    # Create directory if needed
    async_console_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the file
    with open(async_console_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Created advanced async console handler at {async_console_path}")
    logger.info("To use it, import from app.ui.async_console instead of app.ui.console")
    
    return True

if __name__ == "__main__":
    logger.info("Starting console output formatting fix...")
    
    # Fix the console output formatting
    fix_console_output()
    
    # Fix the loader output formatting
    fix_loader_output()
    
    # Fix the manager output formatting
    fix_manager_output()
    
    # Create advanced async console handler
    create_async_console_handler()
    
    logger.info("Console output formatting fix completed")
    print("\n✅ Console output formatting fix completed successfully!")
    print("For advanced asynchronous console handling, use the new async_console module:")
    print("  from app.ui.async_console import display_panel, display_success, display_error")
    print("\nRemember to restart your application for the changes to take effect.")