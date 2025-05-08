#!/usr/bin/env python3
"""
Test script for enhanced translation prompts in CasaLingua

This script demonstrates the use of the TranslationPromptEnhancer
to improve translation quality through better prompt engineering.
"""

import asyncio
import sys
import time
import os
from pathlib import Path

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary components
from app.utils.config import load_config
from app.services.models.loader import ModelLoader
from app.services.hardware.detector import HardwareDetector
from app.services.models.manager import EnhancedModelManager
from app.services.models.translation_prompt_enhancer import TranslationPromptEnhancer
from app.api.schemas.translation import TranslationRequest, TranslationResult

# Import UI components for nice output
from app.ui.console import Console

# Initialize console for pretty output
console = Console()

async def test_enhanced_translation_prompts():
    """Test enhanced translation prompts with different models and options"""
    console.print("\n[bold blue]===== CasaLingua Enhanced Translation Prompts Test =====[/bold blue]")
    
    # Load configuration
    config = load_config()
    console.print(f"Environment: {config.get('environment', 'development')}")
    
    # Detect hardware
    console.print("\n[cyan]Detecting hardware...[/cyan]")
    hardware_detector = HardwareDetector(config)
    hardware_info = hardware_detector.detect_all()
    
    console.print(f"Processor: {hardware_info.get('processor_type', 'unknown')}")
    console.print(f"GPU available: {hardware_info.get('has_gpu', False)}")
    if hardware_info.get('has_gpu', False):
        console.print(f"GPU: {hardware_info.get('gpu_name', 'unknown')}")
    
    # Create model loader and manager
    console.print("\n[cyan]Initializing model manager...[/cyan]")
    model_loader = ModelLoader(config)
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    # Initialize the translation prompt enhancer
    prompt_enhancer = TranslationPromptEnhancer(config)
    
    # Define test cases with various parameters
    test_cases = [
        {
            "name": "Basic Spanish to English",
            "text": "Estoy muy feliz de conocerte hoy. El clima es hermoso y espero que estés bien.",
            "source_language": "es",
            "target_language": "en",
            "domain": None,
            "formality": None,
            "model_types": ["mbart", "mt5"]
        },
        {
            "name": "Legal Spanish to English",
            "text": "El inquilino debe pagar el depósito de seguridad antes de ocupar la vivienda, según lo establecido en el contrato.",
            "source_language": "es",
            "target_language": "en",
            "domain": "legal",
            "formality": "formal",
            "model_types": ["mbart", "mt5"]
        },
        {
            "name": "Casual English to Spanish",
            "text": "Hey, how's it going? Do you want to grab lunch later today?",
            "source_language": "en",
            "target_language": "es",
            "domain": "casual",
            "formality": "informal",
            "model_types": ["mbart", "mt5"]
        },
        {
            "name": "Technical English to Spanish",
            "text": "The database migration will require updating all connection strings in the configuration file.",
            "source_language": "en",
            "target_language": "es",
            "domain": "technical",
            "formality": "neutral",
            "model_types": ["mt5"]
        }
    ]
    
    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        console.print(f"\n[bold]Test Case {i}: {test_case['name']}[/bold]")
        console.print(f"[bold cyan]Source ({test_case['source_language']}):[/bold cyan] {test_case['text']}")
        
        # Process with each model type
        for model_type in test_case["model_types"]:
            console.print(f"\n[bold magenta]Testing with {model_type} model:[/bold magenta]")
            
            # Create basic input data
            input_data = {
                "text": test_case["text"],
                "source_language": test_case["source_language"],
                "target_language": test_case["target_language"],
                "parameters": {}
            }
            
            # Add domain and formality if specified
            if test_case["domain"]:
                input_data["parameters"]["domain"] = test_case["domain"]
                console.print(f"Domain: {test_case['domain']}")
            
            if test_case["formality"]:
                input_data["parameters"]["formality"] = test_case["formality"]
                console.print(f"Formality: {test_case['formality']}")
            
            # Generate enhanced prompts and parameters (we'll just print these, not actually run the model)
            enhanced_input = prompt_enhancer.enhance_translation_input(input_data, model_type)
            
            # Display enhanced prompt for MT5
            if model_type == "mt5":
                console.print("\n[bold cyan]Enhanced MT5 Prompt:[/bold cyan]")
                console.print(enhanced_input["text"])
            
            # Display enhanced parameters for MBART
            if model_type == "mbart":
                console.print("\n[bold cyan]Enhanced MBART Parameters:[/bold cyan]")
                if "generation_kwargs" in enhanced_input.get("parameters", {}):
                    for key, value in enhanced_input["parameters"]["generation_kwargs"].items():
                        console.print(f"  {key}: {value}")
            
            # Run the actual model if directed
            if config.get("run_models", False) and model_manager.is_initialized:
                try:
                    console.print("\n[bold cyan]Attempting model execution...[/bold cyan]")
                    
                    # Try to load the model
                    model_id = f"{model_type}_translation"
                    model_data = await model_manager.load_model(model_id)
                    
                    if model_data and model_data.get("model"):
                        # Run the translation
                        result = await model_manager.run_model(
                            model_id,
                            "process",
                            enhanced_input
                        )
                        
                        if result and "result" in result:
                            translated_text = result["result"]
                            console.print(f"\n[bold green]Translation Result:[/bold green] {translated_text}")
                        else:
                            console.print("[bold red]No translation result returned from model[/bold red]")
                    else:
                        console.print(f"[bold yellow]Model {model_id} could not be loaded, skipping execution[/bold yellow]")
                    
                except Exception as e:
                    console.print(f"[bold red]Error running model: {str(e)}[/bold red]")
                
    console.print("\n[bold blue]===== Enhanced Translation Prompts Test Complete =====[/bold blue]")

# Demonstrate the use of the prompt enhancer without running models
def demonstrate_prompt_enhancement():
    """Demonstrate prompt enhancement without loading or running actual models"""
    console.print("\n[bold]Demonstrating Prompt Enhancement[/bold]")
    
    # Initialize the prompt enhancer
    prompt_enhancer = TranslationPromptEnhancer()
    
    # Test cases for demonstration
    test_texts = [
        ("Estoy muy feliz de conocerte hoy.", "es", "en", None, None),
        ("This legal document requires careful consideration.", "en", "es", "legal", "formal"),
        ("Hey, how's it going? Want to grab lunch?", "en", "es", "casual", "informal"),
        ("The patient exhibits symptoms of acute bronchitis.", "en", "es", "medical", "formal")
    ]
    
    console.print("\n[bold cyan]MT5 Enhanced Prompts:[/bold cyan]")
    for text, src, tgt, domain, formality in test_texts:
        enhanced = prompt_enhancer.enhance_mt5_prompt(text, src, tgt, domain, formality)
        console.print(f"\n[bold]{src} → {tgt}{' ('+domain+')' if domain else ''}{' ('+formality+')' if formality else ''}[/bold]")
        console.print(f"Original: {text}")
        console.print(f"Enhanced: {enhanced}")
    
    console.print("\n[bold cyan]MBART Generation Parameters:[/bold cyan]")
    for _, src, tgt, domain, formality in test_texts:
        params = prompt_enhancer.get_mbart_generation_params(src, tgt, domain, formality)
        console.print(f"\n[bold]{src} → {tgt}{' ('+domain+')' if domain else ''}{' ('+formality+')' if formality else ''}[/bold]")
        for key, value in params.items():
            console.print(f"  {key}: {value}")

if __name__ == "__main__":
    # Check whether to run with actual models
    if "--run-models" in sys.argv:
        asyncio.run(test_enhanced_translation_prompts())
    else:
        # Just demonstrate the prompt enhancement without models
        demonstrate_prompt_enhancement()