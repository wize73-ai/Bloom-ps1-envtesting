#!/usr/bin/env python3
"""
Test the API with enhanced metrics.

This script tests the API endpoints to verify the enhanced metrics are included
in the response. It requires that you first start the API server.

Usage:
    1. First start the API server:
       python app/main.py
    
    2. Then in another terminal, run this script:
       python test_enhanced_api.py
"""

import os
import sys
import json
import requests
import time
from rich.console import Console
from rich.panel import Panel

# Configure API details
BASE_URL = "http://localhost:8000"
API_KEY = os.environ.get("TEST_API_KEY", "cslg_dev_test_key_12345")
AUTH_HEADER = {"Authorization": f"Bearer {API_KEY}"}

# Initialize rich console
console = Console()

def test_translation_metrics():
    """Test translation endpoint for enhanced metrics"""
    console.print("[bold blue]Testing Translation Endpoint for Enhanced Metrics[/bold blue]")
    
    url = f"{BASE_URL}/pipeline/translate"
    headers = {
        "Content-Type": "application/json",
        **AUTH_HEADER
    }
    data = {
        "text": "Hello, this is a test of the translation API with enhanced metrics.",
        "source_language": "en",
        "target_language": "es",
        "preserve_formatting": True
    }
    
    try:
        console.print("[yellow]Sending request to translation endpoint...[/yellow]")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            result = response_data.get("data", {})
            
            # Print translation result
            console.print(Panel(
                f"[green]Translation successful:[/green]\n"
                f"[cyan]Original:[/cyan] {result.get('source_text')}\n"
                f"[cyan]Translated:[/cyan] {result.get('translated_text')}",
                title="Translation Result",
                border_style="green"
            ))
            
            # Check for enhanced metrics
            has_performance_metrics = "performance_metrics" in result and result["performance_metrics"] is not None
            has_memory_usage = "memory_usage" in result and result["memory_usage"] is not None
            has_operation_cost = "operation_cost" in result and result["operation_cost"] is not None
            has_accuracy_score = "accuracy_score" in result
            has_truth_score = "truth_score" in result
            
            # Display enhanced metrics
            console.print(Panel(
                f"[bold]Enhanced Metrics in Response:[/bold]\n"
                f"Performance Metrics: {'[green]✓[/green]' if has_performance_metrics else '[red]✗[/red]'}\n"
                f"Memory Usage: {'[green]✓[/green]' if has_memory_usage else '[red]✗[/red]'}\n"
                f"Operation Cost: {'[green]✓[/green]' if has_operation_cost else '[red]✗[/red]'}\n"
                f"Accuracy Score: {'[green]✓[/green]' if has_accuracy_score else '[red]✗[/red]'}\n"
                f"Truth Score: {'[green]✓[/green]' if has_truth_score else '[red]✗[/red]'}",
                title="Enhanced Metrics Check",
                border_style="blue"
            ))
            
            # Show enhanced metrics details if present
            if has_performance_metrics:
                console.print(Panel(
                    json.dumps(result["performance_metrics"], indent=2),
                    title="Performance Metrics",
                    border_style="magenta"
                ))
            
            if has_memory_usage:
                console.print(Panel(
                    json.dumps(result["memory_usage"], indent=2),
                    title="Memory Usage",
                    border_style="magenta"
                ))
            
            if has_operation_cost:
                console.print(Panel(
                    f"Operation Cost: {result['operation_cost']}",
                    title="Cost Estimate",
                    border_style="magenta"
                ))
            
            if has_accuracy_score or has_truth_score:
                console.print(Panel(
                    f"Accuracy Score: {result.get('accuracy_score')}\n"
                    f"Truth Score: {result.get('truth_score')}",
                    title="Quality Metrics",
                    border_style="magenta"
                ))
            
            # Determine if test passed
            all_metrics_present = all([
                has_performance_metrics,
                has_memory_usage,
                has_operation_cost
            ])
            
            if all_metrics_present:
                console.print(Panel(
                    "[bold green]✓ All essential enhanced metrics are present in the API response![/bold green]",
                    border_style="green"
                ))
                return True
            else:
                console.print(Panel(
                    "[bold red]✗ Some essential enhanced metrics are missing from the API response![/bold red]",
                    border_style="red"
                ))
                return False
            
        else:
            console.print(Panel(
                f"[red]Error:[/red] {response.text}\n"
                f"Status code: {response.status_code}",
                title="API Error",
                border_style="red"
            ))
            return False
    except Exception as e:
        console.print(Panel(
            f"[red]Exception:[/red] {str(e)}",
            title="Request Error",
            border_style="red"
        ))
        console.print("[yellow]Make sure the API server is running on localhost:8000[/yellow]")
        return False

def test_language_detection_metrics():
    """Test language detection endpoint for enhanced metrics"""
    console.print("\n[bold blue]Testing Language Detection Endpoint for Enhanced Metrics[/bold blue]")
    
    url = f"{BASE_URL}/pipeline/detect"
    headers = {
        "Content-Type": "application/json",
        **AUTH_HEADER
    }
    data = {
        "text": "Hello, this is a test of the language detection API with enhanced metrics.",
        "detailed": True
    }
    
    try:
        console.print("[yellow]Sending request to language detection endpoint...[/yellow]")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            result = response_data.get("data", {})
            
            # Print detection result
            console.print(Panel(
                f"[green]Language detection successful:[/green]\n"
                f"[cyan]Detected Language:[/cyan] {result.get('detected_language')}\n"
                f"[cyan]Confidence:[/cyan] {result.get('confidence')}",
                title="Language Detection Result",
                border_style="green"
            ))
            
            # Check for enhanced metrics
            has_performance_metrics = "performance_metrics" in result and result["performance_metrics"] is not None
            has_memory_usage = "memory_usage" in result and result["memory_usage"] is not None
            has_operation_cost = "operation_cost" in result and result["operation_cost"] is not None
            has_accuracy_score = "accuracy_score" in result
            has_truth_score = "truth_score" in result
            
            # Display enhanced metrics
            console.print(Panel(
                f"[bold]Enhanced Metrics in Response:[/bold]\n"
                f"Performance Metrics: {'[green]✓[/green]' if has_performance_metrics else '[red]✗[/red]'}\n"
                f"Memory Usage: {'[green]✓[/green]' if has_memory_usage else '[red]✗[/red]'}\n"
                f"Operation Cost: {'[green]✓[/green]' if has_operation_cost else '[red]✗[/red]'}\n"
                f"Accuracy Score: {'[green]✓[/green]' if has_accuracy_score else '[red]✗[/red]'}\n"
                f"Truth Score: {'[green]✓[/green]' if has_truth_score else '[red]✗[/red]'}",
                title="Enhanced Metrics Check",
                border_style="blue"
            ))
            
            # Show enhanced metrics details if present
            if has_performance_metrics:
                console.print(Panel(
                    json.dumps(result["performance_metrics"], indent=2),
                    title="Performance Metrics",
                    border_style="magenta"
                ))
            
            if has_memory_usage:
                console.print(Panel(
                    json.dumps(result["memory_usage"], indent=2),
                    title="Memory Usage",
                    border_style="magenta"
                ))
            
            # Determine if test passed
            all_metrics_present = all([
                has_performance_metrics,
                has_memory_usage,
                has_operation_cost
            ])
            
            if all_metrics_present:
                console.print(Panel(
                    "[bold green]✓ All essential enhanced metrics are present in the API response![/bold green]",
                    border_style="green"
                ))
                return True
            else:
                console.print(Panel(
                    "[bold red]✗ Some essential enhanced metrics are missing from the API response![/bold red]",
                    border_style="red"
                ))
                return False
            
        else:
            console.print(Panel(
                f"[red]Error:[/red] {response.text}\n"
                f"Status code: {response.status_code}",
                title="API Error",
                border_style="red"
            ))
            return False
    except Exception as e:
        console.print(Panel(
            f"[red]Exception:[/red] {str(e)}",
            title="Request Error",
            border_style="red"
        ))
        console.print("[yellow]Make sure the API server is running on localhost:8000[/yellow]")
        return False

def main():
    """Run all tests"""
    console.print(Panel(
        "[bold]Testing API Enhanced Metrics Integration[/bold]\n\n"
        "This test verifies that the API responses include the enhanced metrics:\n"
        "- Performance metrics (timing, throughput)\n"
        "- Memory usage (system and GPU memory)\n"
        "- Operation cost (cost estimate)\n"
        "- Accuracy and truth scores (quality assessment)",
        title="Enhanced Metrics Test",
        border_style="cyan"
    ))
    
    # Give server time to start up if needed
    console.print("[yellow]Waiting 2 seconds to ensure server is ready...[/yellow]")
    time.sleep(2)
    
    # Run tests
    translation_passed = test_translation_metrics()
    detection_passed = test_language_detection_metrics()
    
    # Summary
    console.print("\n[bold]Test Summary[/bold]")
    console.print(f"Translation API: {'[green]✓ PASSED[/green]' if translation_passed else '[red]✗ FAILED[/red]'}")
    console.print(f"Language Detection API: {'[green]✓ PASSED[/green]' if detection_passed else '[red]✗ FAILED[/red]'}")
    
    all_passed = translation_passed and detection_passed
    if all_passed:
        console.print(Panel(
            "[bold green]All tests passed! Enhanced metrics are successfully integrated in API responses.[/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold red]Some tests failed! Enhanced metrics may not be fully integrated in API responses.[/bold red]",
            border_style="red"
        ))
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())