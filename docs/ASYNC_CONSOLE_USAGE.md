# Async Console Usage Guide

This guide explains how to use the `async_console` module for thread-safe console output in CasaLingua.

## Overview

The `AsyncConsole` class provides a queue-based approach to console output that prevents overlapping text when multiple threads are writing to the console simultaneously. It's particularly useful for applications that combine logging with Rich panels and other visual elements.

## Basic Usage

### 1. Import the async_console singleton

```python
from app.ui.async_console import async_console
```

### 2. Use the console methods

```python
# Display an info panel
async_console.info_panel("Important Notice", "This is important information")

# Display success message
async_console.success_panel("Operation Complete", "The operation was successful")

# Display a warning
async_console.warning_panel("Caution", "Be careful with this operation")

# Display an error
async_console.error_panel("Error Occurred", "Something went wrong")

# Print regular text
async_console.print("Regular text output")

# Print with styling
async_console.print("Styled text", style="bold green")
```

### 3. Advanced Rich components

```python
from rich.text import Text
from rich.table import Table

# Create and display styled text
text = Text()
text.append("CasaLingua ", style="bold cyan")
text.append("supports ", style="default")
text.append("NLLB", style="bold green")
async_console.print(text)

# Create and display a table
table = Table(title="Sample Results")
table.add_column("Column 1")
table.add_column("Column 2")
table.add_row("Value 1", "Value 2")
async_console.table(table)

# Add a horizontal rule
async_console.rule("Section Divider")
```

## Thread Safety

The AsyncConsole handles thread safety internally with a queue-based approach:

1. All output requests are queued instead of being rendered immediately
2. A background thread processes the queue and renders items sequentially
3. Thread locks ensure only one item is rendered at a time
4. Small timing delays prevent overlap with standard logging

This makes it safe to use from any thread without worrying about output corruption.

## Graceful Shutdown

When your application exits, make sure to stop the async console processor:

```python
from app.ui.async_console import async_console, shutdown_console

# In your application shutdown code:
shutdown_console()

# Or directly:
async_console.stop()
```

## Integration with Existing Code

To update existing code that uses the standard Rich Console:

```python
# Before:
from rich.console import Console
console = Console()
console.print(Panel(title="My Title", renderable="Content"))

# After:
from app.ui.async_console import async_console
async_console.panel(title="My Title", content="Content")
```

## Example

See the complete example in `examples/async_console_example.py`, which demonstrates:

- Concurrent logging from multiple threads
- Panels rendering alongside logs
- Tables and styled text
- Proper shutdown handling

Run the example with:

```bash
python examples/async_console_example.py
```

## Best Practices

1. Use the pre-styled panel methods (`info_panel`, `success_panel`, etc.) for consistent styling
2. Let the async_console handle all Rich visualization (panels, tables, etc.)
3. Standard logging can run alongside async_console without modification
4. Remember to call `shutdown_console()` when your application exits