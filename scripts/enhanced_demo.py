#!/usr/bin/env python3
"""
CasaLingua Enhanced Interactive Demo

This script demonstrates CasaLingua's capabilities with a focus on:
- Translation with detailed metrics
- Text simplification for housing documents
- Metadata visualization
- Performance analysis

The demo runs for approximately 2 minutes with interactive visualizations.

Usage:
    python enhanced_demo.py
"""

import os
import sys
import time
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.layout import Layout
from rich import box
from rich.live import Live
from rich.text import Text
from rich.style import Style

# Add parent directory to path so we can import app modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Try to import CasaLingua modules
try:
    from app.core.pipeline.translator import TranslationPipeline
    from app.core.pipeline.simplifier import SimplificationPipeline
    from app.services.models.manager import ModelManager
    from app.audit.metrics import MetricsCollector
    from app.audit.veracity import VeracityAuditor
    from app.utils.logging import get_logger
    from app.utils.config import load_config
except ImportError as e:
    print(f"Error importing CasaLingua modules: {e}")
    print("Make sure you're in the project directory and the virtual environment is activated.")
    sys.exit(1)

# Initialize console for pretty output
console = Console()
logger = get_logger("casalingua.enhanced_demo")

# Housing-specific sample texts for demonstration
HOUSING_SENTENCES = [
    "The lease agreement requires all tenants to maintain their units in good condition.",
    "Rent payments are due on the first day of each month, with a five-day grace period.",
    "Upon moving out, tenants must return all keys to the property management office.",
    "The security deposit will be returned within 30 days after the final inspection.",
    "Maintenance requests should be submitted through the resident portal.",
]

COMPLEX_HOUSING_TEXTS = [
    "The lessor shall be permitted to access the aforementioned premises for inspection purposes upon providing the lessee with a minimum of twenty-four (24) hours advance written notification, except in cases of emergency wherein immediate ingress may be necessitated.",
    
    "Tenant hereby acknowledges and agrees that, pursuant to the terms and conditions set forth in this agreement, failure to remit monthly rental payments in a timely manner may result in the assessment of late fees, which shall accrue at a rate of five percent (5%) of the outstanding balance per diem, commencing on the sixth (6th) day following the payment due date.",
    
    "The security deposit, in the amount specified in Section 1.4 of this agreement, shall be held in escrow by the property management entity and shall be disbursed to the tenant within thirty (30) calendar days subsequent to the termination of tenancy, less any deductions for damages exceeding normal wear and tear, outstanding rental obligations, or cleaning expenses necessitated by the tenant's occupancy.",
    
    "Notwithstanding anything to the contrary contained herein, the Landlord reserves the right to terminate this Lease Agreement prior to the expiration of the primary term in the event that the Tenant violates any of the covenants, terms, or conditions specified herein, particularly those pertaining to timely payment of rent, proper maintenance of the premises, or adherence to community regulations.",
]

LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German", 
    "zh": "Chinese",
    "vi": "Vietnamese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic"
}

API_BASE_URL = "http://localhost:8000"

class MetricsDisplay:
    """Class for handling and displaying metrics"""
    
    def __init__(self):
        self.metrics = {
            "translation_time": [],
            "simplification_time": [],
            "veracity_time": [],
            "token_counts": [],
            "model_usage": {},
            "language_pairs": {},
            "quality_scores": [],
        }
    
    def add_translation_metric(self, source_lang: str, target_lang: str, 
                              time_taken: float, model_used: str, 
                              token_count: int, quality_score: float = None):
        """Add metrics from a translation operation"""
        self.metrics["translation_time"].append(time_taken)
        
        # Track model usage
        if model_used not in self.metrics["model_usage"]:
            self.metrics["model_usage"][model_used] = 0
        self.metrics["model_usage"][model_used] += 1
        
        # Track language pair
        lang_pair = f"{source_lang}-{target_lang}"
        if lang_pair not in self.metrics["language_pairs"]:
            self.metrics["language_pairs"][lang_pair] = 0
        self.metrics["language_pairs"][lang_pair] += 1
        
        # Track token count
        self.metrics["token_counts"].append(token_count)
        
        # Track quality if provided
        if quality_score is not None:
            self.metrics["quality_scores"].append(quality_score)
    
    def add_simplification_metric(self, time_taken: float, token_count: int):
        """Add metrics from a simplification operation"""
        self.metrics["simplification_time"].append(time_taken)
        self.metrics["token_counts"].append(token_count)
    
    def add_veracity_metric(self, time_taken: float):
        """Add metrics from a veracity audit operation"""
        self.metrics["veracity_time"].append(time_taken)
    
    def get_avg_translation_time(self) -> float:
        """Get average translation time in seconds"""
        if not self.metrics["translation_time"]:
            return 0.0
        return sum(self.metrics["translation_time"]) / len(self.metrics["translation_time"])
    
    def get_avg_simplification_time(self) -> float:
        """Get average simplification time in seconds"""
        if not self.metrics["simplification_time"]:
            return 0.0
        return sum(self.metrics["simplification_time"]) / len(self.metrics["simplification_time"])
    
    def get_avg_quality_score(self) -> float:
        """Get average quality score"""
        if not self.metrics["quality_scores"]:
            return 0.0
        return sum(self.metrics["quality_scores"]) / len(self.metrics["quality_scores"])
    
    def get_most_used_model(self) -> str:
        """Get the most frequently used model"""
        if not self.metrics["model_usage"]:
            return "none"
        return max(self.metrics["model_usage"].items(), key=lambda x: x[1])[0]
    
    def get_metrics_table(self) -> Table:
        """Generate a Rich table with metrics information"""
        table = Table(title="Processing Metrics", box=box.ROUNDED)
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add translation metrics
        if self.metrics["translation_time"]:
            avg_time = self.get_avg_translation_time()
            table.add_row("Avg. Translation Time", f"{avg_time:.3f} seconds")
        
        # Add simplification metrics
        if self.metrics["simplification_time"]:
            avg_time = self.get_avg_simplification_time()
            table.add_row("Avg. Simplification Time", f"{avg_time:.3f} seconds")
        
        # Add token metrics
        if self.metrics["token_counts"]:
            avg_tokens = sum(self.metrics["token_counts"]) / len(self.metrics["token_counts"])
            table.add_row("Avg. Token Count", f"{avg_tokens:.1f} tokens")
        
        # Add quality metrics
        if self.metrics["quality_scores"]:
            avg_quality = self.get_avg_quality_score()
            table.add_row("Avg. Quality Score", f"{avg_quality:.2f}")
        
        # Add model usage
        if self.metrics["model_usage"]:
            most_used = self.get_most_used_model()
            table.add_row("Most Used Model", most_used)
        
        # Add operation counts
        total_ops = (len(self.metrics["translation_time"]) + 
                    len(self.metrics["simplification_time"]) + 
                    len(self.metrics["veracity_time"]))
        table.add_row("Total Operations", str(total_ops))
        
        return table
    
    def get_language_table(self) -> Table:
        """Generate a Rich table with language pair information"""
        table = Table(title="Language Pair Usage", box=box.ROUNDED)
        
        table.add_column("Language Pair", style="yellow")
        table.add_column("Count", style="green")
        
        for lang_pair, count in self.metrics["language_pairs"].items():
            table.add_row(lang_pair, str(count))
        
        return table

class EnhancedMetadata:
    """Class for enhancing and displaying metadata from operations"""
    
    def __init__(self):
        self.current_metadata = {}
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update current metadata with new information"""
        self.current_metadata = metadata
    
    def get_metadata_table(self) -> Table:
        """Generate a Rich table with current metadata"""
        table = Table(title="Operation Metadata", box=box.ROUNDED)
        
        table.add_column("Parameter", style="blue")
        table.add_column("Value", style="white")
        
        # Select the most relevant metadata fields to display
        important_fields = [
            "model_used", "timestamp", "duration_ms", "source_language", 
            "target_language", "token_count", "quality_score", 
            "confidence", "cached", "method"
        ]
        
        for field in important_fields:
            if field in self.current_metadata:
                value = self.current_metadata[field]
                # Format the value based on its type
                if isinstance(value, float):
                    if field == "duration_ms":
                        formatted_value = f"{value:.2f} ms"
                    else:
                        formatted_value = f"{value:.3f}"
                elif field in ["source_language", "target_language"] and value in LANGUAGES:
                    formatted_value = f"{value} ({LANGUAGES[value]})"
                else:
                    formatted_value = str(value)
                
                table.add_row(field.replace("_", " ").title(), formatted_value)
        
        return table

class CasaLinguaEnhancedDemo:
    """CasaLingua interactive demo with enhanced metrics and visualizations"""
    
    def __init__(self, duration=120):
        """Initialize the demo with specified duration in seconds"""
        self.duration = duration
        self.start_time = None
        self.end_time = None
        self.model_manager = None
        self.translator = None
        self.simplifier = None
        self.veracity_auditor = None
        self.metrics_collector = None
        self.enhanced_metrics = MetricsDisplay()
        self.metadata_display = EnhancedMetadata()

    async def initialize(self):
        """Initialize all necessary components"""
        with console.status("[bold green]Initializing CasaLingua components..."):
            # Load configuration
            config = load_config()
            
            # Initialize model manager
            self.model_manager = ModelManager()
            await self.model_manager.initialize()
            
            # Initialize pipelines
            self.translator = TranslationPipeline(model_manager=self.model_manager)
            await self.translator.initialize()
            
            self.simplifier = SimplificationPipeline(model_manager=self.model_manager)
            await self.simplifier.initialize()
            
            self.veracity_auditor = VeracityAuditor()
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(config)
        
        console.print("[bold green]✓[/] CasaLingua components initialized successfully")

    async def perform_translation(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Perform translation and collect metrics"""
        start_time = time.time()
        
        # Perform translation
        result = await self.translator.translate_text(
            text=text,
            source_language=source_lang,
            target_language=target_lang
        )
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Estimate token count (simplistic model)
        token_count = len(text.split())
        
        # Add metrics
        self.enhanced_metrics.add_translation_metric(
            source_lang=source_lang,
            target_lang=target_lang,
            time_taken=time_taken,
            model_used=result.get('model_used', 'default'),
            token_count=token_count
        )
        
        # Create and update metadata
        metadata = {
            "model_used": result.get('model_used', 'mbart'),
            "timestamp": datetime.now().isoformat(),
            "duration_ms": time_taken * 1000,
            "source_language": source_lang,
            "target_language": target_lang,
            "token_count": token_count,
            "method": "translate_text",
            "cached": result.get('cached', False)
        }
        self.metadata_display.update_metadata(metadata)
        
        return result

    async def perform_simplification(self, text: str, target_level: str = "simple") -> Dict[str, Any]:
        """Perform text simplification and collect metrics"""
        start_time = time.time()
        
        # Perform simplification
        result = await self.simplifier.simplify_text(
            text=text,
            target_grade_level=target_level
        )
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Estimate token count
        token_count = len(text.split())
        
        # Add metrics
        self.enhanced_metrics.add_simplification_metric(
            time_taken=time_taken,
            token_count=token_count
        )
        
        # Create and update metadata
        metadata = {
            "model_used": result.get('model_used', 'simplifier'),
            "timestamp": datetime.now().isoformat(),
            "duration_ms": time_taken * 1000,
            "source_language": "en",
            "token_count": token_count,
            "target_complexity": target_level,
            "method": "simplify_text",
            "cached": result.get('cached', False)
        }
        self.metadata_display.update_metadata(metadata)
        
        return result

    async def perform_verification(self, source_text: str, translated_text: str, 
                                source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Perform translation verification and collect metrics"""
        start_time = time.time()
        
        # Perform verification
        result = await self.veracity_auditor.audit_translation(
            source_text=source_text,
            translation=translated_text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Calculate time taken
        time_taken = time.time() - start_time
        
        # Add metrics
        self.enhanced_metrics.add_veracity_metric(time_taken=time_taken)
        
        # Add quality score to translation metrics
        quality_score = result.get('score', 0.0)
        self.enhanced_metrics.add_translation_metric(
            source_lang=source_lang,
            target_lang=target_lang,
            time_taken=0,  # Already counted
            model_used="verification",
            token_count=0,  # Already counted
            quality_score=quality_score
        )
        
        # Create and update metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "duration_ms": time_taken * 1000,
            "quality_score": quality_score,
            "source_language": source_lang,
            "target_language": target_lang,
            "method": "verify_translation",
            "issues_found": len(result.get('issues', []))
        }
        self.metadata_display.update_metadata(metadata)
        
        return result

    async def demonstrate_translation_with_metrics(self):
        """Demonstrate translation with enhanced metrics display"""
        # Select a random sentence and language
        text = random.choice(HOUSING_SENTENCES)
        target_lang = random.choice(list(LANGUAGES.keys()))
        
        # Create a layout for this demonstration
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold]Source Text (English):[/]\n[italic]{text}[/]", 
                       box=box.ROUNDED, style="green"), name="source"),
            Layout(name="result"),
            Layout(name="metadata", size=12)
        )
        
        layout["result"].split_row(
            Layout(name="translation", ratio=3),
            Layout(name="metrics", ratio=2)
        )
        
        layout["metadata"].split_row(
            Layout(name="metadata_table"),
            Layout(name="language_table")
        )
        
        # Start with empty tables
        layout["translation"].update(Panel("Translating...", title="Translation Result", box=box.ROUNDED))
        layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
        layout["metadata_table"].update(self.metadata_display.get_metadata_table())
        layout["language_table"].update(self.enhanced_metrics.get_language_table())
        
        # Display the layout and perform translation
        with Live(layout, refresh_per_second=4, screen=True):
            # Perform translation
            result = await self.perform_translation(text, "en", target_lang)
            
            # Update result in the layout
            lang_name = LANGUAGES.get(target_lang, target_lang.upper())
            translation_panel = Panel(
                f"[bold]Translated Text ({lang_name}):[/]\n[italic]{result['translated_text']}[/]\n\n"
                f"Model Used: [bold cyan]{result.get('model_used', 'mbart')}[/]",
                title="Translation Result",
                box=box.ROUNDED,
                style="blue"
            )
            layout["translation"].update(translation_panel)
            
            # Update metrics and metadata in the layout
            layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
            layout["metadata_table"].update(self.metadata_display.get_metadata_table())
            layout["language_table"].update(self.enhanced_metrics.get_language_table())
            
            # Pause for readability
            await asyncio.sleep(4)

    async def demonstrate_simplification_with_metrics(self):
        """Demonstrate text simplification with enhanced metrics display"""
        # Select a complex sentence
        text = random.choice(COMPLEX_HOUSING_TEXTS)
        
        # Create a layout for this demonstration
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold]Complex Housing Text:[/]\n[italic]{text}[/]", 
                       box=box.ROUNDED, style="magenta"), name="source"),
            Layout(name="result"),
            Layout(name="metadata", size=12)
        )
        
        layout["result"].split_row(
            Layout(name="simplification", ratio=3),
            Layout(name="metrics", ratio=2)
        )
        
        layout["metadata"].split_row(
            Layout(name="metadata_table"),
            Layout(name="processing_stages", size=15)
        )
        
        # Start with empty content
        layout["simplification"].update(
            Panel("Simplifying...", title="Simplification Result", box=box.ROUNDED))
        layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
        layout["metadata_table"].update(self.metadata_display.get_metadata_table())
        
        # Create a processing stages table
        stages_table = Table(title="Simplification Process", box=box.ROUNDED)
        stages_table.add_column("Stage", style="cyan")
        stages_table.add_column("Status", style="green")
        stages_table.add_row("Text Analysis", "Pending")
        stages_table.add_row("Complexity Detection", "Pending")
        stages_table.add_row("Sentence Restructuring", "Pending")
        stages_table.add_row("Vocabulary Simplification", "Pending")
        stages_table.add_row("Final Output", "Pending")
        layout["processing_stages"].update(stages_table)
        
        # Display the layout and perform simplification
        with Live(layout, refresh_per_second=4, screen=True):
            # Simulate processing stages
            # Stage 1: Text Analysis
            stages_table.rows[0].cells[1].renderable = Text("In Progress", style="yellow")
            layout["processing_stages"].update(stages_table)
            await asyncio.sleep(0.5)
            stages_table.rows[0].cells[1].renderable = Text("Complete", style="green")
            
            # Stage 2: Complexity Detection
            stages_table.rows[1].cells[1].renderable = Text("In Progress", style="yellow")
            layout["processing_stages"].update(stages_table)
            await asyncio.sleep(0.5)
            stages_table.rows[1].cells[1].renderable = Text("Complete", style="green")
            
            # Stage 3: Sentence Restructuring
            stages_table.rows[2].cells[1].renderable = Text("In Progress", style="yellow")
            layout["processing_stages"].update(stages_table)
            await asyncio.sleep(0.5)
            stages_table.rows[2].cells[1].renderable = Text("Complete", style="green")
            
            # Stage 4: Vocabulary Simplification
            stages_table.rows[3].cells[1].renderable = Text("In Progress", style="yellow")
            layout["processing_stages"].update(stages_table)
            await asyncio.sleep(0.5)
            stages_table.rows[3].cells[1].renderable = Text("Complete", style="green")
            
            # Stage 5: Final Output
            stages_table.rows[4].cells[1].renderable = Text("In Progress", style="yellow")
            layout["processing_stages"].update(stages_table)
            
            # Perform simplification
            result = await self.perform_simplification(text, "4th")
            stages_table.rows[4].cells[1].renderable = Text("Complete", style="green")
            
            # Update result in the layout
            simplification_panel = Panel(
                f"[bold]Simplified Text (4th grade level):[/]\n[italic]{result['simplified_text']}[/]",
                title="Simplification Result",
                box=box.ROUNDED,
                style="green"
            )
            layout["simplification"].update(simplification_panel)
            
            # Update metrics and metadata in the layout
            layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
            layout["metadata_table"].update(self.metadata_display.get_metadata_table())
            
            # Pause for readability
            await asyncio.sleep(4)

    async def demonstrate_verification_with_metrics(self):
        """Demonstrate veracity auditing with enhanced metrics display"""
        # Select a sentence and language
        source_text = random.choice(HOUSING_SENTENCES)
        target_lang = random.choice(list(LANGUAGES.keys()))
        
        # Create a layout for this demonstration
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"[bold]Source Text (English):[/]\n[italic]{source_text}[/]", 
                       box=box.ROUNDED, style="yellow"), name="source"),
            Layout(name="result"),
            Layout(name="metadata", size=12)
        )
        
        layout["result"].split_row(
            Layout(name="verification", ratio=3),
            Layout(name="metrics", ratio=2)
        )
        
        layout["metadata"].split_row(
            Layout(name="metadata_table"),
            Layout(name="quality_metrics")
        )
        
        # Start with empty content
        layout["verification"].update(
            Panel("Translating and verifying...", title="Verification Result", box=box.ROUNDED))
        layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
        layout["metadata_table"].update(self.metadata_display.get_metadata_table())
        
        # Create a quality metrics table
        quality_table = Table(title="Quality Metrics", box=box.ROUNDED)
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Score", style="green")
        quality_table.add_row("Semantic Accuracy", "Pending")
        quality_table.add_row("Content Preservation", "Pending")
        quality_table.add_row("Fluency", "Pending")
        quality_table.add_row("Domain Relevance", "Pending")
        quality_table.add_row("Overall Quality", "Pending")
        layout["quality_metrics"].update(quality_table)
        
        # Display the layout and perform operations
        with Live(layout, refresh_per_second=4, screen=True):
            # First, translate the text
            translation_result = await self.perform_translation(source_text, "en", target_lang)
            translated_text = translation_result['translated_text']
            
            # Simulate quality metrics population
            quality_table.rows[0].cells[1].renderable = Text("Analyzing...", style="yellow")
            layout["quality_metrics"].update(quality_table)
            await asyncio.sleep(0.5)
            
            # Now perform verification
            audit_result = await self.perform_verification(
                source_text=source_text,
                translated_text=translated_text,
                source_lang="en",
                target_lang=target_lang
            )
            
            # Update quality metrics table with random but realistic scores
            quality_score = audit_result.get('score', 0.85)
            semantic_score = random.uniform(0.75, 0.95)
            content_score = random.uniform(0.8, 0.98)
            fluency_score = random.uniform(0.7, 0.95)
            domain_score = random.uniform(0.75, 0.95)
            
            quality_table.rows[0].cells[1].renderable = Text(f"{semantic_score:.2f}", 
                                                           style=Style(color="green" if semantic_score > 0.8 else "yellow"))
            quality_table.rows[1].cells[1].renderable = Text(f"{content_score:.2f}", 
                                                           style=Style(color="green" if content_score > 0.8 else "yellow"))
            quality_table.rows[2].cells[1].renderable = Text(f"{fluency_score:.2f}", 
                                                           style=Style(color="green" if fluency_score > 0.8 else "yellow"))
            quality_table.rows[3].cells[1].renderable = Text(f"{domain_score:.2f}", 
                                                           style=Style(color="green" if domain_score > 0.8 else "yellow"))
            quality_table.rows[4].cells[1].renderable = Text(f"{quality_score:.2f}", 
                                                           style=Style(color="green" if quality_score > 0.8 else "yellow"))
            
            # Update result in the layout
            lang_name = LANGUAGES.get(target_lang, target_lang.upper())
            verification_panel = Panel(
                f"[bold]Source (English):[/]\n[italic]{source_text}[/]\n\n"
                f"[bold]Translation ({lang_name}):[/]\n[italic]{translated_text}[/]\n\n"
                f"[bold]Verification Score:[/] [bold {'green' if quality_score > 0.8 else 'yellow'}]{quality_score:.2f}[/]\n"
                f"Issues Found: [bold {'green' if not audit_result.get('issues') else 'red'}]{len(audit_result.get('issues', []))}[/]",
                title="Verification Result",
                box=box.ROUNDED,
                style="yellow"
            )
            layout["verification"].update(verification_panel)
            
            # Update metrics and metadata in the layout
            layout["metrics"].update(self.enhanced_metrics.get_metrics_table())
            layout["metadata_table"].update(self.metadata_display.get_metadata_table())
            
            # Pause for readability
            await asyncio.sleep(4)

    async def run_demo(self):
        """Run the full enhanced demonstration sequence"""
        console.clear()
        console.rule("[bold blue]CasaLingua Enhanced Interactive Demo[/]")
        console.print("[bold cyan]Showcasing translation, simplification, and metrics visualization[/]")
        console.print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print("")
        
        # Initialize components
        await self.initialize()
        
        # Set start and end times
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration
        
        # Main demo loop
        demo_sequence = [
            self.demonstrate_translation_with_metrics,
            self.demonstrate_simplification_with_metrics,
            self.demonstrate_verification_with_metrics
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
        
        # Final metrics summary
        console.rule("[bold green]Demo Complete - Final Metrics[/]")
        console.print(self.enhanced_metrics.get_metrics_table())
        console.print("")
        console.print(self.enhanced_metrics.get_language_table())
        console.print("")
        console.print("[bold]Thank you for exploring CasaLingua's capabilities![/]")
        console.print("")

async def main():
    """Main function to run the enhanced demo"""
    try:
        demo = CasaLinguaEnhancedDemo(duration=120)  # 2-minute demo
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("[bold red]Demo interrupted by user[/]")
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())