#!/usr/bin/env python3
"""
Demonstration of the thread-safe console output with both approaches:
1. Using the AsyncConsole class for queue-based asynchronous output
2. Using the thread lock and timing approach
"""
import sys
import threading
import time
import random
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("demo")

# Import both console options
from app.ui.console import Console, console_lock, rich_console
from app.ui.async_console import async_console

def log_messages(thread_id):
    """Generate standard log messages."""
    for i in range(5):
        log_level = random.choice(["info", "warning", "error"])
        if log_level == "info":
            logger.info(f"Thread {thread_id}: This is information message {i}")
        elif log_level == "warning":
            logger.warning(f"Thread {thread_id}: This is warning message {i}")
        else:
            logger.error(f"Thread {thread_id}: This is error message {i}")
        time.sleep(random.uniform(0.1, 0.3))

def show_console_lock_panels(thread_id):
    """Demonstrate the console_lock approach."""
    colors = ["blue", "green", "yellow", "red", "magenta"]
    
    for i in range(3):
        # Small delay to allow log messages to complete
        time.sleep(0.2)
        
        # Create a panel
        from rich.panel import Panel
        panel = Panel(
            f"This is panel content {i} from thread {thread_id}",
            title=f"Thread {thread_id} Panel {i}",
            border_style=random.choice(colors)
        )
        
        # Use lock for thread safety
        with console_lock:
            rich_console.print(panel)
            
        # Small delay after panel render
        time.sleep(0.1)

def show_async_console_panels(thread_id):
    """Demonstrate the AsyncConsole approach."""
    for i in range(3):
        panel_type = random.choice(["info", "success", "warning", "error"])
        
        if panel_type == "info":
            async_console.info_panel(
                title=f"Thread {thread_id} Info {i}",
                content=f"This is information panel {i} from thread {thread_id}"
            )
        elif panel_type == "success":
            async_console.success_panel(
                title=f"Thread {thread_id} Success {i}",
                content=f"This is success panel {i} from thread {thread_id}"
            )
        elif panel_type == "warning":
            async_console.warning_panel(
                title=f"Thread {thread_id} Warning {i}",
                content=f"This is warning panel {i} from thread {thread_id}"
            )
        else:
            async_console.error_panel(
                title=f"Thread {thread_id} Error {i}",
                content=f"This is error panel {i} from thread {thread_id}"
            )
            
        time.sleep(random.uniform(0.3, 0.7))

def compare_approaches():
    """Demonstrate both approaches side by side."""
    # Create thread for regular logging
    log_thread = threading.Thread(target=log_messages, args=(0,))
    log_thread.daemon = True
    log_thread.start()
    
    # First approach: AsyncConsole
    logger.info("")
    logger.info("=" * 80)
    logger.info("DEMONSTRATING ASYNC_CONSOLE (QUEUE-BASED APPROACH)")
    logger.info("=" * 80)
    
    async_threads = []
    for i in range(3):
        t = threading.Thread(target=show_async_console_panels, args=(i+1,))
        t.daemon = True
        async_threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in async_threads:
        t.join()
    
    # Small delay between demonstrations
    time.sleep(1)
    
    # Second approach: Console lock with timing
    logger.info("")
    logger.info("=" * 80)
    logger.info("DEMONSTRATING CONSOLE_LOCK APPROACH")
    logger.info("=" * 80)
    
    lock_threads = []
    for i in range(3):
        t = threading.Thread(target=show_console_lock_panels, args=(i+1,))
        t.daemon = True
        lock_threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in lock_threads:
        t.join()

if __name__ == "__main__":
    try:
        # Display title
        async_console.panel(
            title="Thread-Safe Console Demo",
            content="This script demonstrates two approaches to thread-safe console output:\n" + 
                    "1. AsyncConsole with queue-based processing\n" +
                    "2. Direct console with thread locks and timing delays",
            style="bold blue"
        )
        
        # Run the comparison
        compare_approaches()
        
        # Display conclusion
        async_console.success_panel(
            title="Demo Complete",
            content="Both approaches provide thread-safe console output.\n\n" +
                    "AsyncConsole is more robust for complex applications with many threads.\n" +
                    "The console_lock approach is simpler for small applications."
        )
        
    finally:
        # Ensure clean shutdown
        from app.ui.async_console import shutdown_console
        shutdown_console()