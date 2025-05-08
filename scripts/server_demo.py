#!/usr/bin/env python3
"""
CasaLingua Server Demo

This script tests the running CasaLingua server and demonstrates its capabilities:
- Health check
- Translation
- Text simplification
- Veracity auditing

The demo runs for approximately 2 minutes, hitting the actual server.
"""

import os
import sys
import time
import json
import random
import requests
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
import asyncio
import importlib

# Initialize console for pretty output
console = Console()

# API configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Sample texts for demonstration
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Learning a new language opens doors to new cultures and perspectives.",
    "The housing agreement must be signed by all tenants prior to occupancy.",
    "The patient should take this medication twice daily with food.",
    "Climate change is one of the most pressing challenges of our time.",
]

COMPLEX_SENTENCES = [
    "The aforementioned contractual obligations shall be considered null and void if the party of the first part fails to remit payment within the specified timeframe.",
    "Notwithstanding the provisions outlined in section 3.2, the tenant hereby acknowledges that the landlord retains the right to access the premises for inspection purposes given reasonable notice.",
    "The novel's byzantine plot structure, replete with labyrinthine narrative diversions and oblique character motivations, confounded even the most perspicacious readers.",
    "The acquisition of language proficiency necessitates consistent immersion in linguistic contexts that facilitate the assimilation of vocabulary and grammatical constructs.",
]

TARGET_LANGUAGES = ["es", "fr", "de"]
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese"
}

class CasaLinguaServerDemo:
    """CasaLingua server demo class"""
    
    def __init__(self, duration=120):
        """Initialize the demo with specified duration in seconds"""
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.server_healthy = False
    
    async def check_server_availability(self):
        """Check if the server is available and return True if it is"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
            return False
        except requests.RequestException:
            return False
    
    async def get_memory_metrics(self):
        """Get memory usage metrics from API and system"""
        # Get detailed memory metrics about loaded models
        try:
            response = requests.get(f"{API_BASE_URL}/health/metrics", timeout=5)
            if response.status_code == 200:
                metrics_data = response.json()
                return metrics_data
        except:
            pass
        
        # If no metrics endpoint exists, try different possible endpoints
        for endpoint in ["health/detailed", "health/memory", "admin/memory", "admin/metrics"]:
            try:
                response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if "system" in data and isinstance(data["system"], dict):
                        return data["system"]
                    elif "memory" in data and isinstance(data["memory"], dict):
                        return data["memory"]
                    return data
            except:
                continue
        
        # Last resort: Calculate process memory from OS
        try:
            import psutil
            process = psutil.Process(os.getpid())
            server_process = None
            
            # Try to find the Python process running the server
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] == 'python' and any('main.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                    server_process = proc
                    break
            
            if server_process:
                memory_info = server_process.memory_info()
                total_memory = psutil.virtual_memory()
                return {
                    "system_memory": {
                        "total": round(total_memory.total / (1024**3), 2),
                        "available": round(total_memory.available / (1024**3), 2),
                        "used": round((total_memory.total - total_memory.available) / (1024**3), 2)
                    },
                    "process_memory": round(memory_info.rss / (1024**3), 2)
                }
        except:
            pass
        
        return {}

    async def display_memory_table(self, metrics=None):
        """Display a formatted table of memory usage"""
        from rich.table import Table
        
        if not metrics:
            metrics = await self.get_memory_metrics()
        
        # Create and display memory table
        memory_table = Table(title="Memory Usage", box=box.ROUNDED)
        memory_table.add_column("Component", style="cyan")
        memory_table.add_column("Memory Used", style="magenta")
        memory_table.add_column("Memory Available", style="green")
        memory_table.add_column("Details", style="dim")
        
        # System memory
        system_memory = metrics.get("system_memory", {})
        if system_memory:
            total = system_memory.get("total", "Unknown")
            available = system_memory.get("available", "Unknown")
            used = system_memory.get("used", "Unknown")
            memory_table.add_row(
                "System Memory",
                f"{used}" if isinstance(used, str) else f"{used:.2f} GB",
                f"{available}" if isinstance(available, str) else f"{available:.2f} GB",
                f"Total: {total}" if isinstance(total, str) else f"Total: {total:.2f} GB"
            )
        
        # GPU memory if available
        gpu_memory = metrics.get("gpu_memory", {})
        if gpu_memory:
            total = gpu_memory.get("total", "Unknown")
            available = gpu_memory.get("available", "Unknown")
            used = gpu_memory.get("used", "Unknown")
            memory_table.add_row(
                "GPU Memory",
                f"{used}" if isinstance(used, str) else f"{used:.2f} GB",
                f"{available}" if isinstance(available, str) else f"{available:.2f} GB",
                f"Total: {total}" if isinstance(total, str) else f"Total: {total:.2f} GB"
            )
        
        # Model memory
        model_memory = metrics.get("model_memory", {})
        if model_memory:
            for model_name, memory_used in model_memory.items():
                memory_table.add_row(
                    f"Model: {model_name}",
                    f"{memory_used}" if isinstance(memory_used, str) else f"{memory_used:.2f} GB",
                    "-",
                    "Loaded in memory"
                )
        
        # Process memory
        process_memory = metrics.get("process_memory")
        if process_memory:
            memory_table.add_row(
                "Process Memory",
                f"{process_memory}" if isinstance(process_memory, str) else f"{process_memory:.2f} GB",
                "-",
                "Total server process memory"
            )
        
        # If we don't have good metrics, add fallback info
        if len(memory_table.rows) == 0:
            memory_info = metrics.get("memory_available")
            if memory_info:
                memory_table.add_row(
                    "System Memory",
                    "Unknown",
                    f"{memory_info}",
                    "Available for server use"
                )
            memory_table.add_row(
                "Model Memory",
                "See model info in health check",
                "-",
                "No detailed metrics available"
            )
        
        console.print(memory_table)

    async def check_health(self):
        """Check system health and display results"""
        console.print(Panel("[bold]CasaLingua System Health Check[/]", 
                           style="blue", box=box.ROUNDED))
        
        # Check if API is running
        api_status = "Unknown"
        detailed_data = {}
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Checking API health...", total=None)
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            
            if response.status_code == 200:
                api_status = "[bold green]Online[/]"
                self.server_healthy = True
                console.print("  [bold green]✓[/] API is online")
                
                health_data = response.json()
                
                # Check detailed health if available
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True,
                    ) as progress:
                        progress.add_task(description="Fetching detailed health information...", total=None)
                        detailed_response = requests.get(f"{API_BASE_URL}/health/detailed", timeout=5)
                    
                    if detailed_response.status_code == 200:
                        detailed_data = detailed_response.json()
                        
                        # Display model information
                        if "models" in detailed_data:
                            console.print(f"  Loaded Models: [bold cyan]{len(detailed_data['models'])}[/]")
                            for model in detailed_data["models"]:
                                model_name = model.get("name", "Unknown")
                                model_status = model.get("status", "unknown")
                                if model_status == "loaded":
                                    console.print(f"    ✓ [cyan]{model_name}[/]")
                                else:
                                    console.print(f"    ⚠ [yellow]{model_name}[/] ({model_status})")
                        
                        # Display system information
                        if "system" in detailed_data:
                            sys_info = detailed_data["system"]
                            console.print(f"  System Memory: [cyan]{sys_info.get('memory_available', 'Unknown')} available[/]")
                            console.print(f"  System Load: [cyan]{sys_info.get('cpu_usage', 'Unknown')}%[/]")
                    else:
                        console.print("  [yellow]⚠ Detailed health information not available[/]")
                
                except requests.RequestException:
                    console.print("  [yellow]⚠ Detailed health check failed[/]")
            else:
                console.print(f"  [bold red]✗ API health check failed ({response.status_code})[/]")
                api_status = f"[bold red]Error ({response.status_code})[/]"
        except requests.RequestException:
            console.print("  [bold red]✗ API is offline or unreachable[/]")
            api_status = "[bold red]Offline[/]"
        
        # Display memory metrics
        console.print("\n[bold]Memory Usage Metrics:[/]")
        try:
            await self.display_memory_table(detailed_data.get("system"))
        except Exception as e:
            console.print(f"[yellow]⚠ Could not display memory metrics: {str(e)}[/]")
        
        console.print("")
        await asyncio.sleep(2)  # Pause for readability

    async def demonstrate_translation(self):
        """Demonstrate translation capabilities using the API"""
        console.print(Panel("[bold]Translation Demonstration[/]", 
                           style="green", box=box.ROUNDED))
        
        # Select a random sentence
        text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Display source text
        console.print(f"  Source Text [bold yellow](English)[/]:")
        console.print(f"  [italic]\"{text}\"[/]")
        console.print("")
        
        # Prepare request
        payload = {
            "text": text,
            "source_language": "en",
            "target_language": target_lang,
            "preserve_formatting": True
        }
        
        # Call the translation API
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"Translating to {LANGUAGE_NAMES.get(target_lang)}...", total=None)
                response = requests.post(
                    f"{API_BASE_URL}/pipeline/translate", 
                    headers=HEADERS,
                    json=payload,
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                lang_name = LANGUAGE_NAMES.get(target_lang, target_lang.upper())
                translated_text = result.get("data", {}).get("translated_text", "Translation failed")
                model_used = result.get("data", {}).get("model_used", "Unknown")
                process_time = result.get("metadata", {}).get("process_time", 0)
                
                console.print(f"  Translated Text [bold yellow]({lang_name})[/]:")
                console.print(f"  [italic]\"{translated_text}\"[/]")
                console.print(f"  Model Used: [bold cyan]{model_used}[/]")
                console.print(f"  Processing Time: [bold cyan]{process_time:.2f}s[/]")
            else:
                console.print(f"  [bold red]✗ Translation failed ({response.status_code})[/]")
                try:
                    error_data = response.json()
                    console.print(f"  Error: {error_data.get('message', 'Unknown error')}")
                except:
                    console.print(f"  Error: {response.text}")
                
        except requests.RequestException as e:
            console.print(f"  [bold red]✗ Translation request failed: {str(e)}[/]")
        
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def demonstrate_simplification(self):
        """Demonstrate text simplification capabilities using the API"""
        console.print(Panel("[bold]Text Simplification Demonstration[/]", 
                           style="magenta", box=box.ROUNDED))
        
        # Select a complex sentence
        text = random.choice(COMPLEX_SENTENCES)
        
        # Display source text
        console.print(f"  Complex Text:")
        console.print(f"  [italic]\"{text}\"[/]")
        console.print("")
        
        # Prepare request
        payload = {
            "text": text,
            "target_grade_level": "5"  # Simplify to 5th grade reading level
        }
        
        # Call the simplification API
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Simplifying text to 5th grade level...", total=None)
                response = requests.post(
                    f"{API_BASE_URL}/pipeline/simplify", 
                    headers=HEADERS,
                    json=payload,
                    timeout=15
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                simplified_text = result.get("data", {}).get("simplified_text", "Simplification failed")
                process_time = result.get("metadata", {}).get("process_time", 0)
                
                console.print(f"  Simplified Text [bold yellow](5th grade level)[/]:")
                console.print(f"  [italic]\"{simplified_text}\"[/]")
                console.print(f"  Processing Time: [bold cyan]{process_time:.2f}s[/]")
            else:
                console.print(f"  [bold red]✗ Simplification failed ({response.status_code})[/]")
                try:
                    error_data = response.json()
                    console.print(f"  Error: {error_data.get('message', 'Unknown error')}")
                except:
                    console.print(f"  Error: {response.text}")
                
        except requests.RequestException as e:
            console.print(f"  [bold red]✗ Simplification request failed: {str(e)}[/]")
        
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def demonstrate_veracity_audit(self):
        """Demonstrate veracity auditing capabilities using the API"""
        console.print(Panel("[bold]Veracity Audit Demonstration[/]", 
                           style="yellow", box=box.ROUNDED))
        
        # Select a sentence to translate then audit
        source_text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Display source text
        console.print(f"  Source Text [bold](English)[/]:")
        console.print(f"  [italic]\"{source_text}\"[/]")
        
        # Perform translation first
        translated_text = ""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"Translating to {LANGUAGE_NAMES.get(target_lang)}...", total=None)
                response = requests.post(
                    f"{API_BASE_URL}/pipeline/translate", 
                    headers=HEADERS,
                    json={
                        "text": source_text,
                        "source_language": "en",
                        "target_language": target_lang
                    },
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                translated_text = result.get("data", {}).get("translated_text", "")
                console.print(f"  Translated Text:")
                console.print(f"  [italic]\"{translated_text}\"[/]")
                console.print("")
            else:
                console.print(f"  [bold red]✗ Translation for audit failed[/]")
                return
                
        except requests.RequestException:
            console.print(f"  [bold red]✗ Translation request for audit failed[/]")
            return
        
        # Now perform veracity audit
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Performing veracity audit on translation...", total=None)
                response = requests.post(
                    f"{API_BASE_URL}/verify", 
                    headers=HEADERS,
                    json={
                        "source_text": source_text,
                        "translation": translated_text,
                        "source_language": "en",
                        "target_language": target_lang
                    },
                    timeout=15
                )
            
            if response.status_code == 200:
                result = response.json()
                audit_data = result.get("data", {})
                
                # Display audit results
                score = audit_data.get("score", 0.0)
                issues = audit_data.get("issues", [])
                process_time = result.get("metadata", {}).get("process_time", 0)
                
                console.print(f"  Veracity Score: [bold cyan]{score:.2f}[/]")
                console.print(f"  Issues Found: [bold cyan]{len(issues)}[/]")
                
                for issue in issues:
                    severity = issue.get("severity", "info")
                    if severity == "critical":
                        color = "red"
                    elif severity == "warning":
                        color = "yellow"
                    else:
                        color = "blue"
                    
                    console.print(f"    ⚠ [bold {color}]{issue.get('type')}[/]: {issue.get('message')}")
                
                if not issues:
                    console.print(f"    ✓ [bold green]No issues found - translation verified[/]")
                
                console.print(f"  Processing Time: [bold cyan]{process_time:.2f}s[/]")
            else:
                console.print(f"  [bold red]✗ Veracity check failed ({response.status_code})[/]")
                try:
                    error_data = response.json()
                    console.print(f"  Error: {error_data.get('message', 'Unknown error')}")
                except:
                    console.print(f"  Error: {response.text}")
                
        except requests.RequestException as e:
            console.print(f"  [bold red]✗ Veracity audit request failed: {str(e)}[/]")
        
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def show_memory_snapshot(self):
        """Display current memory snapshot"""
        console.print("")
        console.print(Panel("[bold]Memory Usage Snapshot[/]", style="yellow", box=box.ROUNDED))
        await self.display_memory_table()
        console.print("")

    async def run_demo(self):
        """Run the full demonstration sequence"""
        console.clear()
        console.rule("[bold blue]CasaLingua Server Demo[/]")
        console.print("[bold cyan]Testing the running CasaLingua server[/]")
        console.print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print("")
        
        # Check if server is available
        server_available = await self.check_server_availability()
        if not server_available:
            console.print("[bold red]ERROR: CasaLingua server is not accessible![/]")
            console.print("Please make sure the server is running at http://localhost:8000")
            console.print("Exiting demo.")
            return 1
        
        # Set start and end times
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration
        
        # First run health check
        await self.check_health()
        
        if not self.server_healthy:
            console.print("[bold red]Server health check failed. Exiting demo.[/]")
            return 1
        
        # Main demo loop
        demo_sequence = [
            self.demonstrate_translation,
            self.demonstrate_simplification,
            self.demonstrate_veracity_audit
        ]
        
        sequence_index = 0
        memory_check_counter = 0
        while time.time() < self.end_time:
            # Run the next demo in sequence
            await demo_sequence[sequence_index]()
            
            # Show memory usage after every 3 operations
            memory_check_counter += 1
            if memory_check_counter >= 3:
                await self.show_memory_snapshot()
                memory_check_counter = 0
            
            # Move to next demo
            sequence_index = (sequence_index + 1) % len(demo_sequence)
            
            # Show remaining time
            remaining = int(self.end_time - time.time())
            if remaining > 0:
                console.print(f"[dim]Demo will continue for approximately {remaining} more seconds...[/]")
                console.print("")
            
            # Short delay between demonstrations
            await asyncio.sleep(1)
        
        # Final memory snapshot
        console.print("[bold]Final Memory Usage:[/]")
        await self.show_memory_snapshot()
        
        # Completion message
        console.rule("[bold green]Demo Complete[/]")
        console.print("[bold]Thank you for exploring CasaLingua's capabilities![/]")
        console.print("")
        return 0

async def main():
    """Main function to run the demo"""
    try:
        demo = CasaLinguaServerDemo(duration=120)  # 2-minute demo
        return await demo.run_demo()
    except KeyboardInterrupt:
        console.print("[bold red]Demo interrupted by user[/]")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/]")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)