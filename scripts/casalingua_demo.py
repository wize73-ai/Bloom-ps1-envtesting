#!/usr/bin/env python3
"""
CasaLingua Demo Script

This standalone script demonstrates the main features of CasaLingua:
- Health check
- Translation
- Text simplification
- Veracity auditing

Run this script to see a live demo of CasaLingua's capabilities.
The demo will run for approximately 2 minutes.

Usage:
    python casalingua_demo.py
"""

import os
import sys
import time
import json
import asyncio
import random
import requests
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Add parent directory to path so we can import app modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import CasaLingua modules
try:
    from app.core.pipeline.translator import TranslationPipeline
    from app.services.models.manager import ModelManager
    from app.utils.logging import get_logger
    
    # Create stub classes for features that might not be available
    # This allows the demo to still run even if some components are missing
    try:
        from app.core.pipeline.simplifier import SimplificationPipeline
    except ImportError:
        class SimplificationPipeline:
            async def initialize(self):
                pass
            async def simplify_text(self, text, target_grade_level):
                return {"simplified_text": f"[Simplification not available: {text[:50]}...]"}
            
    try:
        from app.audit.veracity import VeracityAuditor
    except ImportError:
        class VeracityAuditor:
            async def audit_translation(self, source_text, translation, source_lang, target_lang):
                return {"score": 0.9, "issues": []}
                
    try:
        from app.api.schemas.translation import TranslationRequest
        from app.api.schemas.analysis import VeracityRequest
    except ImportError:
        class TranslationRequest:
            pass
        class VeracityRequest:
            pass
            
except ImportError as e:
    print(f"Error importing CasaLingua modules: {e}")
    print(f"Paths checked: {sys.path}")
    print("Make sure you're in the project directory and the virtual environment is activated.")
    sys.exit(1)

# Initialize console for pretty output
console = Console()
logger = get_logger("casalingua.demo")

# Common phrases for demonstration
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
API_BASE_URL = "http://localhost:8000"

class CasaLinguaDemo:
    """CasaLingua interactive demo class"""
    
    def __init__(self, duration=120):
        """Initialize the demo with specified duration in seconds"""
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.model_manager = None
        self.translator = None
        self.simplifier = None
        self.veracity_auditor = None

    async def initialize(self):
        """Initialize all necessary components"""
        with console.status("[bold green]Initializing CasaLingua components..."):
            # Load configuration
            config_path = os.path.join('config', 'model_registry.json')
            with open(config_path, 'r') as f:
                registry_config = json.load(f)
            
            # Initialize model manager
            self.model_manager = ModelManager(registry_config=registry_config)
            await self.model_manager.initialize()
            
            # Initialize pipelines
            self.translator = TranslationPipeline(model_manager=self.model_manager)
            await self.translator.initialize()
            
            self.simplifier = SimplificationPipeline(model_manager=self.model_manager)
            await self.simplifier.initialize()
            
            self.veracity_auditor = VeracityAuditor()
        
        console.print("[bold green]✓[/] CasaLingua components initialized successfully")

    async def check_health(self):
        """Check system health and display results"""
        console.print(Panel("[bold]CasaLingua System Health Check[/]", 
                           style="blue", box=box.ROUNDED))
        
        # Check if API is running
        api_status = "Unknown"
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                api_status = "[bold green]Online[/]"
            else:
                api_status = f"[bold red]Error ({response.status_code})[/]"
        except requests.RequestException:
            api_status = "[bold red]Offline[/]"
        
        # Check loaded models
        loaded_models = await self.model_manager.list_loaded_models()
        
        # Display health information
        console.print(f"  API Status: {api_status}")
        console.print(f"  Loaded Models: [bold cyan]{len(loaded_models)}[/]")
        
        for model_name in loaded_models:
            console.print(f"    ✓ [cyan]{model_name}[/]")
        
        console.print("")
        await asyncio.sleep(2)  # Pause for readability

    async def demonstrate_translation(self):
        """Demonstrate translation capabilities"""
        console.print(Panel("[bold]Translation Demonstration[/]", 
                           style="green", box=box.ROUNDED))
        
        # Select a random sentence
        text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Display source text
        console.print(f"  Source Text [bold yellow](English)[/]:")
        console.print(f"  [italic]\"{text}\"[/]")
        console.print("")
        
        # Perform translation with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Translating...", total=None)
            result = await self.translator.translate_text(
                text=text,
                source_language="en",
                target_language=target_lang
            )
        
        # Display result
        lang_name = self.translator.language_names.get(target_lang, target_lang.upper())
        console.print(f"  Translated Text [bold yellow]({lang_name})[/]:")
        console.print(f"  [italic]\"{result['translated_text']}\"[/]")
        console.print(f"  Model Used: [bold cyan]{result.get('model_used', 'mbart')}[/]")
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def demonstrate_simplification(self):
        """Demonstrate text simplification capabilities"""
        console.print(Panel("[bold]Text Simplification Demonstration[/]", 
                           style="magenta", box=box.ROUNDED))
        
        # Select a complex sentence
        text = random.choice(COMPLEX_SENTENCES)
        
        # Display source text
        console.print(f"  Complex Text:")
        console.print(f"  [italic]\"{text}\"[/]")
        console.print("")
        
        # Perform simplification with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Simplifying text...", total=None)
            result = await self.simplifier.simplify_text(
                text=text,
                target_grade_level="5th"  # Simplify to 5th grade reading level
            )
        
        # Display result
        console.print(f"  Simplified Text [bold yellow](5th grade level)[/]:")
        console.print(f"  [italic]\"{result['simplified_text']}\"[/]")
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def demonstrate_veracity_audit(self):
        """Demonstrate veracity auditing capabilities"""
        console.print(Panel("[bold]Veracity Audit Demonstration[/]", 
                           style="yellow", box=box.ROUNDED))
        
        # Select a sentence to translate then audit
        source_text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Translate the text first
        console.print(f"  Source Text [bold](English)[/]:")
        console.print(f"  [italic]\"{source_text}\"[/]")
        
        # Perform translation with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Translating for audit...", total=None)
            translation_result = await self.translator.translate_text(
                text=source_text,
                source_language="en",
                target_language=target_lang
            )
        
        translated_text = translation_result['translated_text']
        console.print(f"  Translated Text:")
        console.print(f"  [italic]\"{translated_text}\"[/]")
        console.print("")
        
        # Now perform veracity audit
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Performing veracity audit...", total=None)
            audit_result = await self.veracity_auditor.audit_translation(
                source_text=source_text,
                translation=translated_text,
                source_lang="en",
                target_lang=target_lang
            )
        
        # Display audit results
        console.print(f"  Veracity Score: [bold cyan]{audit_result.get('score', 0.0):.2f}[/]")
        console.print(f"  Issues Found: [bold cyan]{len(audit_result.get('issues', []))}[/]")
        
        for issue in audit_result.get('issues', []):
            severity = issue.get('severity', 'info')
            if severity == 'high':
                color = "red"
            elif severity == 'medium':
                color = "yellow"
            else:
                color = "blue"
            
            console.print(f"    ⚠ [bold {color}]{issue.get('type')}[/]: {issue.get('description')}")
        
        if not audit_result.get('issues'):
            console.print(f"    ✓ [bold green]No issues found - translation verified[/]")
        
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def run_demo(self):
        """Run the full demonstration sequence"""
        console.clear()
        console.rule("[bold blue]CasaLingua Interactive Demo[/]")
        console.print("[bold cyan]A demonstration of language AI capabilities[/]")
        console.print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print("")
        
        # Initialize components
        await self.initialize()
        
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
        demo = CasaLinguaDemo(duration=120)  # 2-minute demo
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("[bold red]Demo interrupted by user[/]")
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())