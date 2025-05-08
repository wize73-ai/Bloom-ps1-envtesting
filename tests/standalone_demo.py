#!/usr/bin/env python3
"""
CasaLingua Standalone Demo

This is a completely standalone demo that simulates CasaLingua's core features.
It doesn't require any API server or complex dependencies.

The demo shows:
- Health status simulation
- Translation examples
- Text simplification examples
- Veracity audit simulation

Run this for a 2-minute interactive demo of CasaLingua's capabilities.
"""

import os
import sys
import time
import json
import random
import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
import asyncio

# Initialize console for pretty output
console = Console()

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

# Translation examples (pre-defined for demo purposes)
TRANSLATIONS = {
    "es": {
        "The quick brown fox jumps over the lazy dog.": "El zorro marrón rápido salta sobre el perro perezoso.",
        "Learning a new language opens doors to new cultures and perspectives.": "Aprender un nuevo idioma abre puertas a nuevas culturas y perspectivas.",
        "The housing agreement must be signed by all tenants prior to occupancy.": "El contrato de vivienda debe ser firmado por todos los inquilinos antes de la ocupación.",
        "The patient should take this medication twice daily with food.": "El paciente debe tomar este medicamento dos veces al día con alimentos.",
        "Climate change is one of the most pressing challenges of our time.": "El cambio climático es uno de los desafíos más urgentes de nuestro tiempo."
    },
    "fr": {
        "The quick brown fox jumps over the lazy dog.": "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Learning a new language opens doors to new cultures and perspectives.": "Apprendre une nouvelle langue ouvre des portes vers de nouvelles cultures et perspectives.",
        "The housing agreement must be signed by all tenants prior to occupancy.": "Le contrat de logement doit être signé par tous les locataires avant l'occupation.",
        "The patient should take this medication twice daily with food.": "Le patient doit prendre ce médicament deux fois par jour avec de la nourriture.",
        "Climate change is one of the most pressing challenges of our time.": "Le changement climatique est l'un des défis les plus pressants de notre époque."
    },
    "de": {
        "The quick brown fox jumps over the lazy dog.": "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Learning a new language opens doors to new cultures and perspectives.": "Das Erlernen einer neuen Sprache öffnet Türen zu neuen Kulturen und Perspektiven.",
        "The housing agreement must be signed by all tenants prior to occupancy.": "Der Wohnungsvertrag muss von allen Mietern vor dem Einzug unterzeichnet werden.",
        "The patient should take this medication twice daily with food.": "Der Patient sollte dieses Medikament zweimal täglich mit dem Essen einnehmen.",
        "Climate change is one of the most pressing challenges of our time.": "Der Klimawandel ist eine der drängendsten Herausforderungen unserer Zeit."
    }
}

# Simplified versions of complex sentences
SIMPLIFIED_SENTENCES = {
    "The aforementioned contractual obligations shall be considered null and void if the party of the first part fails to remit payment within the specified timeframe.": 
        "The contract will be canceled if the first party does not pay on time.",
    
    "Notwithstanding the provisions outlined in section 3.2, the tenant hereby acknowledges that the landlord retains the right to access the premises for inspection purposes given reasonable notice.": 
        "Despite what section 3.2 says, the tenant agrees that the landlord can enter to inspect the property if they give proper notice.",
    
    "The novel's byzantine plot structure, replete with labyrinthine narrative diversions and oblique character motivations, confounded even the most perspicacious readers.": 
        "The book's complex plot, with many twists and unclear character motives, confused even the smartest readers.",
    
    "The acquisition of language proficiency necessitates consistent immersion in linguistic contexts that facilitate the assimilation of vocabulary and grammatical constructs.": 
        "To learn a language well, you need to regularly practice in situations that help you learn new words and grammar."
}

TARGET_LANGUAGES = ["es", "fr", "de"]
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German"
}

class CasaLinguaStandaloneDemo:
    """CasaLingua standalone demo class"""
    
    def __init__(self, duration=120):
        """Initialize the demo with specified duration in seconds"""
        self.duration = duration
        self.start_time = None
        self.end_time = None
    
    async def simulate_loading(self, message="Loading...", duration=1.0):
        """Simulate a loading process"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(description=message, total=None)
            await asyncio.sleep(duration)
    
    async def check_health(self):
        """Simulate system health check"""
        console.print(Panel("[bold]CasaLingua System Health Check[/]", 
                           style="blue", box=box.ROUNDED))
        
        # Simulate checking API health
        await self.simulate_loading("Checking API health...", 0.7)
        console.print("  [bold green]✓[/] API is online")
        
        # Simulate checking model status
        await self.simulate_loading("Checking model status...", 1.0)
        
        # Display simulated model information
        models = [
            {"name": "MBART Translation", "status": "loaded"},
            {"name": "Simplifier (BART)", "status": "loaded"},
            {"name": "Language Detection (XLM-R)", "status": "loaded"},
            {"name": "Veracity Auditor", "status": "loaded"}
        ]
        
        console.print(f"  Loaded Models: [bold cyan]{len(models)}[/]")
        for model in models:
            model_name = model.get("name", "Unknown")
            model_status = model.get("status", "unknown")
            if model_status == "loaded":
                console.print(f"    ✓ [cyan]{model_name}[/]")
        
        # Display simulated system information
        console.print(f"  System Memory: [cyan]24.0 GB available[/]")
        console.print(f"  System Load: [cyan]12%[/]")
        
        console.print("")
        await asyncio.sleep(2)  # Pause for readability

    async def demonstrate_translation(self):
        """Demonstrate translation capabilities"""
        console.print(Panel("[bold]Translation Demonstration[/]", 
                           style="green", box=box.ROUNDED))
        
        # Select a random sentence and language
        text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Display source text
        console.print(f"  Source Text [bold yellow](English)[/]:")
        console.print(f"  [italic]\"{text}\"[/]")
        console.print("")
        
        # Simulate translation process
        await self.simulate_loading(f"Translating to {LANGUAGE_NAMES.get(target_lang)}...", 1.5)
        
        # Get predetermined translation or fallback to a simple one
        translated_text = TRANSLATIONS.get(target_lang, {}).get(text)
        if not translated_text:
            # Simple fallback translation simulation
            translated_text = f"[Translation to {target_lang}]: {text[:30]}..."
        
        # Display result
        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang.upper())
        console.print(f"  Translated Text [bold yellow]({lang_name})[/]:")
        console.print(f"  [italic]\"{translated_text}\"[/]")
        console.print(f"  Model Used: [bold cyan]mbart (facebook/mbart-large-50-many-to-many-mmt)[/]")
        console.print(f"  Translation Time: [bold cyan]1.2s[/]")
        
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
        
        # Simulate simplification process
        await self.simulate_loading("Simplifying text to 5th grade reading level...", 1.8)
        
        # Get predetermined simplified version or generate a simple one
        simplified_text = SIMPLIFIED_SENTENCES.get(text)
        if not simplified_text:
            # Simple fallback simplification simulation
            simplified_text = text.replace("aforementioned", "mentioned")
            simplified_text = simplified_text.replace("null and void", "canceled")
            simplified_text = simplified_text.replace("byzantine", "complex")
            simplified_text = simplified_text.replace("labyrinthine", "confusing")
            simplified_text = simplified_text.replace("perspicacious", "smart")
            simplified_text = simplified_text.replace("necessitates", "needs")
            simplified_text = simplified_text.replace("assimilation", "learning")
        
        # Display result
        console.print(f"  Simplified Text [bold yellow](5th grade level)[/]:")
        console.print(f"  [italic]\"{simplified_text}\"[/]")
        console.print(f"  Model Used: [bold cyan]BART (facebook/bart-large-cnn)[/]")
        console.print(f"  Reading Level Change: [bold cyan]College → Grade 5[/]")
        
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def demonstrate_veracity_audit(self):
        """Demonstrate veracity auditing capabilities"""
        console.print(Panel("[bold]Veracity Audit Demonstration[/]", 
                           style="yellow", box=box.ROUNDED))
        
        # Select a random sentence and language for translation
        source_text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Display source text
        console.print(f"  Source Text [bold](English)[/]:")
        console.print(f"  [italic]\"{source_text}\"[/]")
        
        # Simulate translation process
        await self.simulate_loading(f"Translating to {LANGUAGE_NAMES.get(target_lang)}...", 1.0)
        
        # Get predetermined translation or fallback
        translated_text = TRANSLATIONS.get(target_lang, {}).get(source_text)
        if not translated_text:
            translated_text = f"[Translation to {target_lang}]: {source_text[:30]}..."
        
        console.print(f"  Translated Text:")
        console.print(f"  [italic]\"{translated_text}\"[/]")
        console.print("")
        
        # Simulate veracity audit process
        await self.simulate_loading("Performing veracity audit of translation...", 2.0)
        
        # Randomly decide if there will be issues (for demo variety)
        has_issues = random.choice([True, False])
        
        # Display audit results
        score = 0.97 if not has_issues else random.uniform(0.70, 0.85)
        console.print(f"  Veracity Score: [bold cyan]{score:.2f}[/]")
        
        if has_issues:
            # Show some random issues
            issues = [
                {"type": "Meaning Shift", "severity": "medium", "description": "A subtle change in meaning was detected in part of the translation"},
                {"type": "Number Mismatch", "severity": "high", "description": "A numerical value was translated incorrectly"},
                {"type": "Formality Level", "severity": "low", "description": "The translation uses a different formality level than the source"}
            ]
            selected_issues = random.sample(issues, k=random.randint(1, 2))
            
            console.print(f"  Issues Found: [bold cyan]{len(selected_issues)}[/]")
            
            for issue in selected_issues:
                severity = issue.get("severity", "info")
                if severity == "high":
                    color = "red"
                elif severity == "medium":
                    color = "yellow"
                else:
                    color = "blue"
                
                console.print(f"    ⚠ [bold {color}]{issue.get('type')}[/]: {issue.get('description')}")
        else:
            console.print(f"  Issues Found: [bold cyan]0[/]")
            console.print(f"    ✓ [bold green]No issues found - translation verified[/]")
        
        console.print("")
        await asyncio.sleep(3)  # Pause for readability

    async def run_demo(self):
        """Run the full demonstration sequence"""
        console.clear()
        console.rule("[bold blue]CasaLingua Interactive Demo[/]")
        console.print("[bold cyan]A demonstration of language AI capabilities[/]")
        console.print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Demo mode: [bold yellow]STANDALONE SIMULATION[/]")
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
        demo = CasaLinguaStandaloneDemo(duration=120)  # 2-minute demo
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("[bold red]Demo interrupted by user[/]")
    except Exception as e:
        console.print(f"[bold red]Error during demo: {str(e)}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())