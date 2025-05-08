"""
Asynchronous console handler for thread-safe output with Rich.
Provides queue-based rendering to prevent overlapping outputs.
"""
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union
from queue import Queue
import time

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.style import Style
from rich.theme import Theme

# Custom theme with CasaLingua colors
CASALINGUA_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "highlight": "bold magenta",
    "muted": "dim white"
})

class AsyncConsole:
    """Thread-safe asynchronous console for handling Rich output."""
    
    def __init__(self, theme: Theme = CASALINGUA_THEME):
        """
        Initialize the async console.
        
        Args:
            theme: Rich theme for styling console output
        """
        self.console = Console(theme=theme)
        self.message_queue = Queue()
        self._running = False
        self._processor_thread = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the message processor thread."""
        if self._running:
            return
            
        self._running = True
        self._processor_thread = threading.Thread(
            target=self._process_messages,
            daemon=True
        )
        self._processor_thread.start()
    
    def stop(self):
        """Stop the message processor thread."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=1.0)
            self._processor_thread = None
    
    def _process_messages(self):
        """Process messages from the queue and render them."""
        while self._running:
            try:
                if not self.message_queue.empty():
                    msg_type, content, kwargs = self.message_queue.get(block=False)
                    
                    with self._lock:
                        # Small delay to ensure any logging finishes
                        time.sleep(0.05)
                        
                        if msg_type == "panel":
                            self.console.print(Panel(**content, **kwargs))
                        elif msg_type == "text":
                            self.console.print(content, **kwargs)
                        elif msg_type == "table":
                            self.console.print(content, **kwargs)
                        elif msg_type == "rule":
                            self.console.rule(content, **kwargs)
                        elif msg_type == "log":
                            self.console.log(content, **kwargs)
                        
                        # Small delay after rendering to avoid overlap
                        time.sleep(0.05)
                    
                    self.message_queue.task_done()
                else:
                    # Don't burn CPU when queue is empty
                    time.sleep(0.01)
            except Exception as e:
                # Protect against any rendering errors
                print(f"Error in message processor: {e}")
                time.sleep(0.1)
    
    def panel(self, title: str, content: Union[str, Text], **kwargs):
        """
        Queue a panel for rendering.
        
        Args:
            title: Panel title
            content: Panel content (string or Rich Text)
            **kwargs: Additional panel options
        """
        self.message_queue.put((
            "panel", 
            {"title": title, "renderable": content}, 
            kwargs
        ))
    
    def print(self, content: Any, **kwargs):
        """
        Queue text for rendering.
        
        Args:
            content: Content to print
            **kwargs: Additional print options
        """
        self.message_queue.put(("text", content, kwargs))
    
    def table(self, table: Table, **kwargs):
        """
        Queue a table for rendering.
        
        Args:
            table: Rich Table to render
            **kwargs: Additional table options
        """
        self.message_queue.put(("table", table, kwargs))
    
    def rule(self, title: str = "", **kwargs):
        """
        Queue a horizontal rule for rendering.
        
        Args:
            title: Rule title
            **kwargs: Additional rule options
        """
        self.message_queue.put(("rule", title, kwargs))
    
    def log(self, message: str, **kwargs):
        """
        Queue a log message for rendering.
        
        Args:
            message: Log message
            **kwargs: Additional log options
        """
        self.message_queue.put(("log", message, kwargs))
    
    def info_panel(self, title: str, content: str):
        """Render an info panel with predefined styling."""
        self.panel(
            title=title, 
            content=content, 
            style="info", 
            border_style="info"
        )
    
    def success_panel(self, title: str, content: str):
        """Render a success panel with predefined styling."""
        self.panel(
            title=title, 
            content=content, 
            style="success", 
            border_style="success"
        )
    
    def warning_panel(self, title: str, content: str):
        """Render a warning panel with predefined styling."""
        self.panel(
            title=title, 
            content=content, 
            style="warning", 
            border_style="warning"
        )
    
    def error_panel(self, title: str, content: str):
        """Render an error panel with predefined styling."""
        self.panel(
            title=title, 
            content=content, 
            style="error", 
            border_style="error"
        )

# Create a singleton instance for import
async_console = AsyncConsole()

# Auto-start the processor thread
async_console.start()

def shutdown_console():
    """Shutdown the console message processor gracefully."""
    async_console.stop()