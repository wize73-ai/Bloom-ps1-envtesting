#!/usr/bin/env python3
"""
CasaLingua API Demo Script

This standalone script demonstrates the main features of CasaLingua using the API:
- Health check
- Translation
- Text simplification
- Veracity auditing

The demo runs for approximately 2 minutes showing the various capabilities.

Usage:
    python casalingua_api_demo.py
"""

import os
import sys
import time
import json
import random
import requests
import datetime
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

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

TARGET_LANGUAGES = ["es", "fr", "de", "it", "zh"]
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese"
}

class CasaLinguaApiDemo:
    """CasaLingua API interactive demo class"""
    
    def __init__(self, duration=120):
        """Initialize the demo with specified duration in seconds"""
        self.duration = duration
        self.start_time = None
        self.end_time = None
        
    async def check_health(self):
        """Check system health and display results"""
        console.print(Panel("[bold]CasaLingua System Health Check[/]", 
                           style="blue", box=box.ROUNDED))
        
        # Check basic health
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Checking API health...", total=None)
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            
            if response.status_code == 200:
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
                
        except requests.RequestException:
            console.print("  [bold red]✗ API is offline or unreachable[/]")
        
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
                progress.add_task(description="Translating...", total=None)
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
                
                console.print(f"  Translated Text [bold yellow]({lang_name})[/]:")
                console.print(f"  [italic]\"{translated_text}\"[/]")
                console.print(f"  Model Used: [bold cyan]{model_used}[/]")
            else:
                console.print(f"  [bold red]✗ Translation failed ({response.status_code})[/]")
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
                progress.add_task(description="Simplifying text...", total=None)
                response = requests.post(
                    f"{API_BASE_URL}/pipeline/simplify", 
                    headers=HEADERS,
                    json=payload,
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                simplified_text = result.get("data", {}).get("simplified_text", "Simplification failed")
                
                console.print(f"  Simplified Text [bold yellow](5th grade level)[/]:")
                console.print(f"  [italic]\"{simplified_text}\"[/]")
            else:
                # Fallback to a simulated simplification if API fails
                console.print(f"  [yellow]⚠ Simplification API not available, showing simulated result[/]")
                simplified = text.replace("aforementioned", "mentioned")
                simplified = simplified.replace("null and void", "canceled")
                simplified = simplified.replace("byzantine", "complex")
                simplified = simplified.replace("labyrinthine", "confusing")
                simplified = simplified.replace("perspicacious", "smart")
                simplified = simplified.replace("necessitates", "needs")
                simplified = simplified.replace("assimilation", "learning")
                console.print(f"  Simplified Text [bold yellow](simulated)[/]:")
                console.print(f"  [italic]\"{simplified}\"[/]")
                
        except requests.RequestException as e:
            console.print(f"  [bold red]✗ Simplification request failed: {str(e)}[/]")
            # Fallback to a simulated simplification
            simplified = "This text would be simplified to be easier to read."
            console.print(f"  Simplified Text [bold yellow](simulated)[/]:")
            console.print(f"  [italic]\"{simplified}\"[/]")
        
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
                progress.add_task(description="Translating for audit...", total=None)
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
                progress.add_task(description="Performing veracity audit...", total=None)
                response = requests.post(
                    f"{API_BASE_URL}/pipeline/verify", 
                    headers=HEADERS,
                    json={
                        "source_text": source_text,
                        "translated_text": translated_text,
                        "source_language": "en",
                        "target_language": target_lang
                    },
                    timeout=15
                )
            
            if response.status_code == 200:
                result = response.json()
                audit_data = result.get("data", {})
                
                # Display audit results
                score = audit_data.get("verification_score", 0.0)
                issues = audit_data.get("issues", [])
                
                console.print(f"  Veracity Score: [bold cyan]{score:.2f}[/]")
                console.print(f"  Issues Found: [bold cyan]{len(issues)}[/]")
                
                for issue in issues:
                    severity = issue.get("severity", "info")
                    if severity == "high":
                        color = "red"
                    elif severity == "medium":
                        color = "yellow"
                    else:
                        color = "blue"
                    
                    console.print(f"    ⚠ [bold {color}]{issue.get('type')}[/]: {issue.get('description')}")
                
                if not issues:
                    console.print(f"    ✓ [bold green]No issues found - translation verified[/]")
            else:
                # Fallback to a simulated audit if API fails
                console.print(f"  [yellow]⚠ Veracity API not available, showing simulated result[/]")
                console.print(f"  Veracity Score: [bold cyan]0.95[/] (simulated)")
                console.print(f"    ✓ [bold green]Simulated verification passed[/]")
                
        except requests.RequestException as e:
            console.print(f"  [bold red]✗ Veracity audit request failed: {str(e)}[/]")
            # Fallback to a simulated audit
            console.print(f"  Veracity Score: [bold cyan]0.92[/] (simulated)")
            console.print(f"    ✓ [bold green]Simulated verification passed[/]")
        
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def run_demo(self):
        """Run the full demonstration sequence"""
        console.clear()
        console.rule("[bold blue]CasaLingua Interactive Demo[/]")
        console.print("[bold cyan]A demonstration of language AI capabilities[/]")
        console.print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print("")
        
        # Set start and end times
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration
        
        # Main demo loop
        demo_sequence = [
            self.check_health,
            self.demonstrate_translation,
            self.demonstrate_simplification,
            self.demonstrate_veracity_audit
        ]
        
        sequence_index = 0
        while time.time() < self.end_time:
            # Run the next demo in sequence
            await demo_sequence[sequence_index]()
            
            # Move to next demo
            sequence_index = (sequence_index + 1) % len(demo_sequence)
            
            # Show remaining time
            remaining = int(self.end_time - time.time())
            if remaining > 0:
                console.print(f"[dim]Demo will continue for approximately {remaining} more seconds...[/]")
                console.print("")
            
            # Short delay between demonstrations
            await asyncio.sleep(1)
        
        # Completion message
        console.rule("[bold green]Demo Complete[/]")
        console.print("[bold]Thank you for exploring CasaLingua's capabilities![/]")
        console.print("")

async def main():
    """Main function to run the demo"""
    try:
        demo = CasaLinguaApiDemo(duration=120)  # 2-minute demo
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("[bold red]Demo interrupted by user[/]")
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())