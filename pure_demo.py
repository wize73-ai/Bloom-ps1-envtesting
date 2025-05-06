#!/usr/bin/env python3
"""
CasaLingua Pure Demo

This is a completely standalone demo that simulates CasaLingua's core features
using only Python standard library and rich for formatting.

NO MODEL DOWNLOADS - 100% SIMULATION
"""

import os
import sys
import time
import random
import datetime

# Check if rich is installed, if not use plain text
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' package for better visuals: pip install rich")

# Set up console
if HAS_RICH:
    console = Console()
else:
    # Create a simple console replacement
    class SimpleConsole:
        def print(self, text, **kwargs):
            # Strip out rich formatting
            text = text.replace("[bold]", "").replace("[/]", "")
            text = text.replace("[italic]", "").replace("[cyan]", "")
            text = text.replace("[yellow]", "").replace("[green]", "")
            text = text.replace("[red]", "").replace("[blue]", "")
            text = text.replace("[magenta]", "").replace("[dim]", "")
            print(text)
            
        def rule(self, text=""):
            width = 80
            print("\n" + "-" * width)
            if text:
                print(text.center(width))
                print("-" * width)
                
        def clear(self):
            if os.name == 'nt':  # For Windows
                os.system('cls')
            else:  # For Linux/Mac
                os.system('clear')
    
    console = SimpleConsole()

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

class CasaLinguaDemo:
    """CasaLingua demo class"""
    
    def __init__(self, duration=120):
        """Initialize the demo with specified duration in seconds"""
        self.duration = duration
        self.start_time = None
        self.end_time = None
    
    def simulate_loading(self, message, duration=1.0):
        """Simulate a loading process"""
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(description=message, total=None)
                time.sleep(duration)
        else:
            print(f"{message}...")
            time.sleep(duration)
            print("Done!")
    
    def show_panel(self, title, style="blue"):
        """Show a panel with title"""
        if HAS_RICH:
            console.print(Panel(f"[bold]{title}[/]", style=style, box=box.ROUNDED))
        else:
            console.print(f"\n--- {title} ---\n")
    
    def check_health(self):
        """Simulate system health check"""
        self.show_panel("CasaLingua System Health Check", "blue")
        
        # Simulate checking API health
        self.simulate_loading("Checking API health", 0.7)
        console.print("  ✓ API is online")
        
        # Simulate checking model status
        self.simulate_loading("Checking model status", 1.0)
        
        # Display simulated model information
        models = [
            {"name": "MBART Translation", "status": "loaded"},
            {"name": "Simplifier (BART)", "status": "loaded"},
            {"name": "Language Detection (XLM-R)", "status": "loaded"},
            {"name": "Veracity Auditor", "status": "loaded"}
        ]
        
        console.print(f"  Loaded Models: {len(models)}")
        for model in models:
            model_name = model.get("name", "Unknown")
            console.print(f"    ✓ {model_name}")
        
        # Display simulated system information
        console.print(f"  System Memory: 24.0 GB available")
        console.print(f"  System Load: 12%")
        
        console.print("")
        time.sleep(2)  # Pause for readability

    def demonstrate_translation(self):
        """Demonstrate translation capabilities"""
        self.show_panel("Translation Demonstration", "green")
        
        # Select a random sentence and language
        text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Display source text
        console.print(f"  Source Text (English):")
        console.print(f"  \"{text}\"")
        console.print("")
        
        # Simulate translation process
        self.simulate_loading(f"Translating to {LANGUAGE_NAMES.get(target_lang)}", 1.5)
        
        # Get predetermined translation or fallback to a simple one
        translated_text = TRANSLATIONS.get(target_lang, {}).get(text)
        if not translated_text:
            # Simple fallback translation simulation
            translated_text = f"[Translation to {target_lang}]: {text[:30]}..."
        
        # Display result
        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang.upper())
        console.print(f"  Translated Text ({lang_name}):")
        console.print(f"  \"{translated_text}\"")
        console.print(f"  Model Used: mbart (facebook/mbart-large-50-many-to-many-mmt)")
        console.print(f"  Translation Time: 1.2s")
        
        console.print("")
        time.sleep(3)  # Pause for readability

    def demonstrate_simplification(self):
        """Demonstrate text simplification capabilities"""
        self.show_panel("Text Simplification Demonstration", "magenta")
        
        # Select a complex sentence
        text = random.choice(COMPLEX_SENTENCES)
        
        # Display source text
        console.print(f"  Complex Text:")
        console.print(f"  \"{text}\"")
        console.print("")
        
        # Simulate simplification process
        self.simulate_loading("Simplifying text to 5th grade reading level", 1.8)
        
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
        console.print(f"  Simplified Text (5th grade level):")
        console.print(f"  \"{simplified_text}\"")
        console.print(f"  Model Used: BART (facebook/bart-large-cnn)")
        console.print(f"  Reading Level Change: College → Grade 5")
        
        console.print("")
        time.sleep(3)  # Pause for readability

    def demonstrate_veracity_audit(self):
        """Demonstrate veracity auditing capabilities"""
        self.show_panel("Veracity Audit Demonstration", "yellow")
        
        # Select a random sentence and language for translation
        source_text = random.choice(SAMPLE_SENTENCES)
        target_lang = random.choice(TARGET_LANGUAGES)
        
        # Display source text
        console.print(f"  Source Text (English):")
        console.print(f"  \"{source_text}\"")
        
        # Simulate translation process
        self.simulate_loading(f"Translating to {LANGUAGE_NAMES.get(target_lang)}", 1.0)
        
        # Get predetermined translation or fallback
        translated_text = TRANSLATIONS.get(target_lang, {}).get(source_text)
        if not translated_text:
            translated_text = f"[Translation to {target_lang}]: {source_text[:30]}..."
        
        console.print(f"  Translated Text:")
        console.print(f"  \"{translated_text}\"")
        console.print("")
        
        # Simulate veracity audit process
        self.simulate_loading("Performing veracity audit of translation", 2.0)
        
        # Randomly decide if there will be issues (for demo variety)
        has_issues = random.choice([True, False])
        
        # Display audit results
        score = 0.97 if not has_issues else random.uniform(0.70, 0.85)
        console.print(f"  Veracity Score: {score:.2f}")
        
        if has_issues:
            # Show some random issues
            issues = [
                {"type": "Meaning Shift", "severity": "medium", "description": "A subtle change in meaning was detected in part of the translation"},
                {"type": "Number Mismatch", "severity": "high", "description": "A numerical value was translated incorrectly"},
                {"type": "Formality Level", "severity": "low", "description": "The translation uses a different formality level than the source"}
            ]
            selected_issues = random.sample(issues, k=random.randint(1, 2))
            
            console.print(f"  Issues Found: {len(selected_issues)}")
            
            for issue in selected_issues:
                severity = issue.get("severity", "info")
                icon = "⚠"
                console.print(f"    {icon} {issue.get('type')}: {issue.get('description')}")
        else:
            console.print(f"  Issues Found: 0")
            console.print(f"    ✓ No issues found - translation verified")
        
        console.print("")
        time.sleep(3)  # Pause for readability

    def run_demo(self):
        """Run the full demonstration sequence"""
        console.clear()
        console.rule("CasaLingua Interactive Demo")
        console.print("A demonstration of language AI capabilities")
        console.print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Demo mode: STANDALONE SIMULATION")
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
            demo_sequence[sequence_index]()
            
            # Move to next demo
            sequence_index = (sequence_index + 1) % len(demo_sequence)
            
            # Show remaining time
            remaining = int(self.end_time - time.time())
            if remaining > 0:
                console.print(f"Demo will continue for approximately {remaining} more seconds...")
                console.print("")
            
            # Short delay between demonstrations
            time.sleep(1)
        
        # Completion message
        console.rule("Demo Complete")
        console.print("Thank you for exploring CasaLingua's capabilities!")
        console.print("")

def main():
    """Main function to run the demo"""
    try:
        demo = CasaLinguaDemo(duration=120)  # 2-minute demo
        demo.run_demo()
        return 0
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())