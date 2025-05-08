#!/usr/bin/env python3
"""
Simple demonstration of the async_console module for thread-safe console output.
"""
import sys
import threading
import time
import random
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the async_console
from app.ui.async_console import async_console

def log_thread(thread_id):
    """Generate log messages in a thread."""
    for i in range(5):
        message_type = random.choice(["info", "success", "warning", "error"])
        
        if message_type == "info":
            async_console.info_panel(
                f"Information #{i} from Thread {thread_id}",
                f"This is information message {i} from thread {thread_id}"
            )
        elif message_type == "success":
            async_console.success_panel(
                f"Success #{i} from Thread {thread_id}",
                f"Operation {i} completed successfully in thread {thread_id}"
            )
        elif message_type == "warning":
            async_console.warning_panel(
                f"Warning #{i} from Thread {thread_id}",
                f"This is a warning about operation {i} in thread {thread_id}"
            )
        else:
            async_console.error_panel(
                f"Error #{i} from Thread {thread_id}",
                f"An error occurred in operation {i} in thread {thread_id}"
            )
            
        # Random delay between messages
        time.sleep(random.uniform(0.3, 0.8))

def main():
    """Run the demonstration."""
    async_console.panel(
        title="AsyncConsole Demo",
        content="This demonstrates thread-safe console output with multiple threads.\n"
                "Each thread will display 5 messages with random styling.",
        style="bold blue"
    )
    
    # Create multiple threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=log_thread, args=(i,))
        t.daemon = True
        threads.append(t)
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Show conclusion
    async_console.success_panel(
        "Demo Complete",
        "All threads have finished outputting messages.\n\n"
        "The async_console ensures that panels don't overlap, even when\n"
        "multiple threads are trying to write to the console simultaneously."
    )

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure proper shutdown
        from app.ui.async_console import shutdown_console
        shutdown_console()