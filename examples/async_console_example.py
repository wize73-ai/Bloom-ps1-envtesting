"""
Example demonstrating the usage of the async_console module with CasaLingua.
Run this script to see thread-safe console output in action.
"""
import sys
import threading
import time
import random
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app.ui.async_console import async_console
from rich.text import Text
from rich.table import Table
import logging

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("async_console_demo")

def simulate_concurrent_logging():
    """Simulate multiple threads logging simultaneously."""
    for i in range(10):
        log_type = random.choice(["INFO", "WARNING", "ERROR"])
        if log_type == "INFO":
            logger.info(f"Regular log message #{i} from thread {threading.current_thread().name}")
        elif log_type == "WARNING":
            logger.warning(f"Warning log message #{i} from thread {threading.current_thread().name}")
        else:
            logger.error(f"Error log message #{i} from thread {threading.current_thread().name}")
        time.sleep(random.uniform(0.1, 0.5))

def simulate_panel_rendering():
    """Simulate rendering Rich panels."""
    panel_types = ["info", "success", "warning", "error"]
    
    for i in range(5):
        panel_type = random.choice(panel_types)
        
        if panel_type == "info":
            async_console.info_panel(
                title=f"Information Panel #{i}", 
                content="This is an information panel with important details."
            )
        elif panel_type == "success":
            async_console.success_panel(
                title=f"Success Panel #{i}", 
                content="Operation completed successfully!"
            )
        elif panel_type == "warning":
            async_console.warning_panel(
                title=f"Warning Panel #{i}", 
                content="Be cautious about this operation."
            )
        else:
            async_console.error_panel(
                title=f"Error Panel #{i}", 
                content="Something went wrong! Check the logs for details."
            )
            
        time.sleep(random.uniform(0.2, 0.8))

def create_sample_table():
    """Create and display a sample Rich table."""
    table = Table(title="Sample Translation Results")
    
    table.add_column("Original", style="cyan")
    table.add_column("Translated", style="green")
    table.add_column("Model", style="magenta")
    table.add_column("Time (ms)", justify="right", style="dim")
    
    table.add_row(
        "Hello, world!", 
        "¡Hola, mundo!", 
        "NLLB-200-1.3B", 
        "245"
    )
    table.add_row(
        "How are you today?", 
        "¿Cómo estás hoy?", 
        "NLLB-200-1.3B", 
        "312"
    )
    table.add_row(
        "Machine translation is amazing", 
        "La traducción automática es increíble", 
        "NLLB-200-1.3B", 
        "420"
    )
    
    async_console.table(table)

def demonstrate_styled_text():
    """Demonstrate styled text rendering."""
    text = Text()
    text.append("CasaLingua ", style="bold cyan")
    text.append("now supports ", style="default")
    text.append("NLLB-200-1.3B", style="bold green")
    text.append(" for ", style="default")
    text.append("200 languages", style="bold yellow")
    text.append("!", style="default")
    
    async_console.print(text)
    async_console.rule("Text Styling Example")

def main():
    """Run the demonstration."""
    async_console.info_panel(
        "Async Console Demo", 
        "This example demonstrates the thread-safe asynchronous console with Rich."
    )
    
    # Create and start threads for logging
    log_threads = []
    for i in range(3):
        t = threading.Thread(
            target=simulate_concurrent_logging,
            name=f"LoggingThread-{i}",
            daemon=True
        )
        log_threads.append(t)
        t.start()
    
    # Create thread for panel rendering
    panel_thread = threading.Thread(
        target=simulate_panel_rendering,
        name="PanelThread",
        daemon=True
    )
    panel_thread.start()
    
    # Wait a moment to let other threads run
    time.sleep(2)
    
    # Demonstrate other features
    create_sample_table()
    time.sleep(1)
    demonstrate_styled_text()
    
    # Wait for all threads to finish
    for t in log_threads:
        t.join()
    panel_thread.join()
    
    async_console.success_panel(
        "Demo Complete", 
        "The async console demonstration has completed successfully."
    )
    
    # Allow time for final messages to be processed
    time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure clean shutdown
        async_console.stop()