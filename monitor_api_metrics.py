#!/usr/bin/env python3
"""
Monitor API metrics in real-time.
This script continuously sends requests to the API endpoints and prints the metrics.
"""
import os
import time
import json
import requests
import threading
import signal
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table

# API configuration
BASE_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": f"Bearer {os.environ.get('TEST_API_KEY', 'cslg_dev_test_key_12345')}"}

# Rich console for pretty output
console = Console()

# Global flag to control monitoring
running = True

def signal_handler(sig, frame):
    """Handle CTRL+C to gracefully exit"""
    global running
    console.print("\n[bold red]Stopping API metrics monitor...[/bold red]")
    running = False
    sys.exit(0)

def format_metrics(metrics):
    """Format metrics for display"""
    if not metrics:
        return "None"
    
    # Format memory as human-readable
    if "memory_usage" in metrics and metrics["memory_usage"]:
        memory = metrics["memory_usage"]
        if "before" in memory and memory["before"]:
            memory["before"]["total"] = f"{memory['before']['total'] / (1024**3):.2f} GB"
            memory["before"]["available"] = f"{memory['before']['available'] / (1024**3):.2f} GB"
            memory["before"]["used"] = f"{memory['before']['used'] / (1024**3):.2f} GB"
            memory["before"]["free"] = f"{memory['before']['free'] / (1024**3):.2f} GB"
        if "after" in memory and memory["after"]:
            memory["after"]["total"] = f"{memory['after']['total'] / (1024**3):.2f} GB"
            memory["after"]["available"] = f"{memory['after']['available'] / (1024**3):.2f} GB"
            memory["after"]["used"] = f"{memory['after']['used'] / (1024**3):.2f} GB"
            memory["after"]["free"] = f"{memory['after']['free'] / (1024**3):.2f} GB"
    
    # Format performance metrics
    if "performance_metrics" in metrics and metrics["performance_metrics"]:
        perf = metrics["performance_metrics"]
        if "throughput" in perf and perf["throughput"]:
            # Round values for better display
            if "tokens_per_second" in perf["throughput"]:
                perf["throughput"]["tokens_per_second"] = f"{perf['throughput']['tokens_per_second']:.2f}"
            if "chars_per_second" in perf["throughput"]:
                perf["throughput"]["chars_per_second"] = f"{perf['throughput']['chars_per_second']:.2f}"
            
    # Format cost
    if "operation_cost" in metrics and metrics["operation_cost"]:
        metrics["operation_cost"] = f"${metrics['operation_cost']:.6f}"
        
    return json.dumps(metrics, indent=2)

def test_language_detection():
    """Test language detection endpoint and print metrics"""
    url = f"{BASE_URL}/pipeline/detect"
    headers = {
        "Content-Type": "application/json",
        **AUTH_HEADER
    }
    data = {
        "text": "Hello, how are you today?",
        "detailed": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            
            # Extract metrics from response
            metrics = {}
            if "data" in response_json and response_json["data"]:
                data = response_json["data"]
                metrics["performance_metrics"] = data.get("performance_metrics")
                metrics["memory_usage"] = data.get("memory_usage")
                metrics["operation_cost"] = data.get("operation_cost")
                metrics["accuracy_score"] = data.get("accuracy_score")
                metrics["truth_score"] = data.get("truth_score")
            
            # Extract metrics from metadata
            metadata_metrics = {}
            if "metadata" in response_json and response_json["metadata"]:
                metadata = response_json["metadata"]
                metadata_metrics["performance_metrics"] = metadata.get("performance_metrics")
                metadata_metrics["memory_usage"] = metadata.get("memory_usage")
                metadata_metrics["operation_cost"] = metadata.get("operation_cost")
                metadata_metrics["accuracy_score"] = metadata.get("accuracy_score")
                metadata_metrics["truth_score"] = metadata.get("truth_score")
            
            # Return metrics for display
            return {
                "endpoint": "Language Detection",
                "status": response.status_code,
                "data_metrics": metrics,
                "metadata_metrics": metadata_metrics,
                "detected_language": response_json.get("data", {}).get("detected_language", "unknown"),
                "confidence": response_json.get("data", {}).get("confidence", 0.0)
            }
        else:
            return {
                "endpoint": "Language Detection",
                "status": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return {
            "endpoint": "Language Detection",
            "status": "Error",
            "error": str(e)
        }

def test_translation():
    """Test translation endpoint and print metrics"""
    url = f"{BASE_URL}/pipeline/translate"
    headers = {
        "Content-Type": "application/json",
        **AUTH_HEADER
    }
    data = {
        "text": "Hello, how are you today?",
        "source_language": "en",
        "target_language": "es"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            
            # Extract metrics from response
            metrics = {}
            if "data" in response_json and response_json["data"]:
                data = response_json["data"]
                metrics["performance_metrics"] = data.get("performance_metrics")
                metrics["memory_usage"] = data.get("memory_usage")
                metrics["operation_cost"] = data.get("operation_cost")
                metrics["accuracy_score"] = data.get("accuracy_score")
                metrics["truth_score"] = data.get("truth_score")
            
            # Extract metrics from metadata
            metadata_metrics = {}
            if "metadata" in response_json and response_json["metadata"]:
                metadata = response_json["metadata"]
                metadata_metrics["performance_metrics"] = metadata.get("performance_metrics")
                metadata_metrics["memory_usage"] = metadata.get("memory_usage")
                metadata_metrics["operation_cost"] = metadata.get("operation_cost")
                metadata_metrics["accuracy_score"] = metadata.get("accuracy_score")
                metadata_metrics["truth_score"] = metadata.get("truth_score")
            
            # Return metrics for display
            return {
                "endpoint": "Translation",
                "status": response.status_code,
                "data_metrics": metrics,
                "metadata_metrics": metadata_metrics,
                "translated_text": response_json.get("data", {}).get("translated_text", ""),
                "source_language": response_json.get("data", {}).get("source_language", "unknown"),
                "target_language": response_json.get("data", {}).get("target_language", "unknown")
            }
        else:
            return {
                "endpoint": "Translation",
                "status": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return {
            "endpoint": "Translation",
            "status": "Error",
            "error": str(e)
        }

def test_simplify():
    """Test simplification endpoint and print metrics"""
    url = f"{BASE_URL}/pipeline/simplify"
    headers = {
        "Content-Type": "application/json",
        **AUTH_HEADER
    }
    data = {
        "text": "The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms.",
        "language": "en",
        "target_level": "simple"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            
            # Extract metrics from metadata
            metadata_metrics = {}
            if "metadata" in response_json and response_json["metadata"]:
                metadata = response_json["metadata"]
                metadata_metrics["performance_metrics"] = metadata.get("performance_metrics")
                metadata_metrics["memory_usage"] = metadata.get("memory_usage")
                metadata_metrics["operation_cost"] = metadata.get("operation_cost")
                metadata_metrics["accuracy_score"] = metadata.get("accuracy_score")
                metadata_metrics["truth_score"] = metadata.get("truth_score")
            
            # Return metrics for display
            return {
                "endpoint": "Simplification",
                "status": response.status_code,
                "metadata_metrics": metadata_metrics,
                "simplified_text": response_json.get("data", {}).get("simplified_text", "")
            }
        else:
            return {
                "endpoint": "Simplification",
                "status": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return {
            "endpoint": "Simplification",
            "status": "Error",
            "error": str(e)
        }

def display_results(results):
    """Display results in a pretty format"""
    console.clear()
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header"),
        Layout(name="body")
    )
    
    # Header
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    layout["header"].update(
        Panel(
            f"[bold blue]API Metrics Monitor[/bold blue] - [green]{current_time}[/green]\n"
            f"[yellow]Press Ctrl+C to exit[/yellow]",
            border_style="blue"
        )
    )
    
    # Body content
    results_panel = []
    for result in results:
        if "error" in result:
            # Error case
            results_panel.append(
                Panel(
                    f"[bold red]Error: {result.get('error', 'Unknown error')}[/bold red]",
                    title=f"[bold]{result['endpoint']}[/bold] - Status: {result['status']}",
                    border_style="red"
                )
            )
        else:
            # Success case
            content = []
            
            # Add endpoint-specific data
            if result["endpoint"] == "Language Detection":
                content.append(f"[bold cyan]Detected Language:[/bold cyan] [green]{result.get('detected_language', 'unknown')}[/green]")
                content.append(f"[bold cyan]Confidence:[/bold cyan] [green]{result.get('confidence', 0.0):.4f}[/green]")
            elif result["endpoint"] == "Translation":
                content.append(f"[bold cyan]Source Language:[/bold cyan] [green]{result.get('source_language', 'unknown')}[/green]")
                content.append(f"[bold cyan]Target Language:[/bold cyan] [green]{result.get('target_language', 'unknown')}[/green]")
                content.append(f"[bold cyan]Translated Text:[/bold cyan] [green]{result.get('translated_text', '')}[/green]")
            elif result["endpoint"] == "Simplification":
                content.append(f"[bold cyan]Simplified Text:[/bold cyan] [green]{result.get('simplified_text', '')}[/green]")
            
            # Add metrics
            if "data_metrics" in result:
                data_metrics = result["data_metrics"]
                if any(data_metrics.values()):
                    content.append("\n[bold magenta]Data Metrics:[/bold magenta]")
                    
                    # Add performance metrics
                    if data_metrics.get("performance_metrics"):
                        perf = data_metrics["performance_metrics"]
                        content.append(f"[bold yellow]Performance:[/bold yellow]")
                        if "total_time" in perf:
                            content.append(f"  Total Time: {perf['total_time']:.4f}s")
                        if "throughput" in perf and perf["throughput"]:
                            throughput = perf["throughput"]
                            if "tokens_per_second" in throughput:
                                content.append(f"  Tokens/Second: {throughput['tokens_per_second']}")
                            if "chars_per_second" in throughput:
                                content.append(f"  Chars/Second: {throughput['chars_per_second']}")
                    
                    # Add memory usage
                    if data_metrics.get("memory_usage"):
                        content.append(f"[bold yellow]Memory Usage:[/bold yellow] {data_metrics['memory_usage'].get('difference', {}).get('used', 'N/A')}")
                    
                    # Add operation cost
                    if data_metrics.get("operation_cost"):
                        content.append(f"[bold yellow]Operation Cost:[/bold yellow] {data_metrics['operation_cost']}")
                    
                    # Add accuracy scores
                    if data_metrics.get("accuracy_score"):
                        content.append(f"[bold yellow]Accuracy Score:[/bold yellow] {data_metrics['accuracy_score']:.4f}")
                    
                    # Add truth scores
                    if data_metrics.get("truth_score"):
                        content.append(f"[bold yellow]Truth Score:[/bold yellow] {data_metrics['truth_score']:.4f}")
            
            # Add metadata metrics
            if "metadata_metrics" in result:
                metadata_metrics = result["metadata_metrics"]
                if any(metadata_metrics.values()):
                    content.append("\n[bold magenta]Metadata Metrics:[/bold magenta]")
                    
                    # Add performance metrics
                    if metadata_metrics.get("performance_metrics"):
                        perf = metadata_metrics["performance_metrics"]
                        content.append(f"[bold yellow]Performance:[/bold yellow]")
                        if "total_time" in perf:
                            content.append(f"  Total Time: {perf['total_time']:.4f}s")
                        if "throughput" in perf and perf["throughput"]:
                            throughput = perf["throughput"]
                            if "tokens_per_second" in throughput:
                                content.append(f"  Tokens/Second: {throughput['tokens_per_second']}")
                            if "chars_per_second" in throughput:
                                content.append(f"  Chars/Second: {throughput['chars_per_second']}")
                    
                    # Add memory usage
                    if metadata_metrics.get("memory_usage"):
                        content.append(f"[bold yellow]Memory Usage:[/bold yellow] {metadata_metrics['memory_usage'].get('difference', {}).get('used', 'N/A')}")
                    
                    # Add operation cost
                    if metadata_metrics.get("operation_cost"):
                        content.append(f"[bold yellow]Operation Cost:[/bold yellow] {metadata_metrics['operation_cost']}")
                    
                    # Add accuracy scores
                    if metadata_metrics.get("accuracy_score"):
                        content.append(f"[bold yellow]Accuracy Score:[/bold yellow] {metadata_metrics['accuracy_score']:.4f}")
                    
                    # Add truth scores
                    if metadata_metrics.get("truth_score"):
                        content.append(f"[bold yellow]Truth Score:[/bold yellow] {metadata_metrics['truth_score']:.4f}")
            
            results_panel.append(
                Panel(
                    "\n".join(content),
                    title=f"[bold]{result['endpoint']}[/bold] - Status: {result['status']}",
                    border_style="green"
                )
            )
    
    # Update body
    layout["body"].update(Panel(
        "\n".join([str(panel) for panel in results_panel]),
        title="[bold]Endpoint Results[/bold]",
        border_style="cyan"
    ))
    
    console.print(layout)

def main():
    """Main function to monitor API metrics"""
    # Register signal handler for CTRL+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set environment variables
    os.environ["CASALINGUA_ENV"] = "development"
    
    console.print("[bold blue]Starting API Metrics Monitor...[/bold blue]")
    console.print("[yellow]Press Ctrl+C to exit[/yellow]")
    
    # Monitor loop
    while running:
        # Run tests
        results = []
        results.append(test_language_detection())
        results.append(test_translation())
        results.append(test_simplify())
        
        # Display results
        display_results(results)
        
        # Wait before next check
        time.sleep(5)

if __name__ == "__main__":
    main()