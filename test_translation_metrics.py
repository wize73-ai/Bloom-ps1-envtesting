#!/usr/bin/env python3
"""
Test translation endpoint with enhanced metrics.
"""
import os
import sys
import json
import requests
from rich.console import Console
from rich.panel import Panel

# API configuration
BASE_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": f"Bearer {os.environ.get('TEST_API_KEY', 'cslg_dev_test_key_12345')}"}

# Rich console for pretty output
console = Console()

def test_translation():
    """Test translation endpoint and print metrics"""
    console.print("[bold blue]Testing Translation Endpoint[/bold blue]")
    
    url = f"{BASE_URL}/pipeline/translate"
    headers = {
        "Content-Type": "application/json",
        **AUTH_HEADER
    }
    data = {
        "text": "Hello, this is a test of the translation API.",
        "source_language": "en",
        "target_language": "es",
        "preserve_formatting": True
    }
    
    console.print(f"[yellow]Request:[/yellow]")
    console.print(f"URL: {url}")
    console.print(f"Headers: {json.dumps(headers, indent=2)}")
    console.print(f"Data: {json.dumps(data, indent=2)}")
    
    try:
        console.print("\n[yellow]Sending request...[/yellow]")
        response = requests.post(url, headers=headers, json=data)
        console.print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            
            # Print success response
            console.print(Panel(
                f"[green]Success![/green]\n\n"
                f"[bold cyan]Source Text:[/bold cyan] {response_json.get('data', {}).get('source_text', 'unknown')}\n"
                f"[bold cyan]Translated Text:[/bold cyan] {response_json.get('data', {}).get('translated_text', 'unknown')}\n"
                f"[bold cyan]Confidence:[/bold cyan] {response_json.get('data', {}).get('confidence', 0.0):.4f}",
                title="Response Data",
                border_style="green"
            ))
            
            # Print metrics
            data = response_json.get("data", {})
            console.print(Panel(
                f"[bold magenta]Performance Metrics:[/bold magenta]\n{json.dumps(data.get('performance_metrics'), indent=2)}\n\n"
                f"[bold magenta]Memory Usage:[/bold magenta]\n{json.dumps(data.get('memory_usage'), indent=2)}\n\n"
                f"[bold magenta]Operation Cost:[/bold magenta] {data.get('operation_cost')}\n\n"
                f"[bold magenta]Accuracy Score:[/bold magenta] {data.get('accuracy_score')}\n\n"
                f"[bold magenta]Truth Score:[/bold magenta] {data.get('truth_score')}",
                title="Enhanced Metrics",
                border_style="magenta"
            ))
            
            # Print full response for debugging
            console.print(Panel(
                json.dumps(response_json, indent=2),
                title="Full Response JSON",
                border_style="blue"
            ))
            
            return True
        else:
            # Print error response
            console.print(Panel(
                f"[red]Error:[/red] {response.text}",
                title=f"Error Response (Status: {response.status_code})",
                border_style="red"
            ))
            return False
    except Exception as e:
        console.print(Panel(
            f"[red]Exception:[/red] {str(e)}",
            title="Request Error",
            border_style="red"
        ))
        return False

def main():
    """Main function"""
    # Set environment variables
    os.environ["CASALINGUA_ENV"] = "development"
    
    console.print("[bold]Testing API Endpoint with Enhanced Metrics[/bold]")
    console.print("-" * 50)
    
    # Run test
    success = test_translation()
    
    # Print summary
    console.print("\n[bold]Test Summary[/bold]")
    console.print("-" * 50)
    if success:
        console.print("[green]✓ Test completed successfully[/green]")
    else:
        console.print("[red]✗ Test failed[/red]")
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())