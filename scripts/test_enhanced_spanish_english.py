#!/usr/bin/env python3
"""
Test script for enhanced Spanish to English translation in CasaLingua

This script tests the enhanced translation prompts specifically for 
the Spanish to English language pair, which historically had issues.
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
from app.services.models.wrapper import TranslationModelWrapper, ModelInput
from app.api.schemas.translation import TranslationRequest
from app.ui.console import Console

# Initialize console for nice output
console = Console()

async def test_enhanced_spanish_english():
    """Test enhanced Spanish to English translation with various options"""
    console.print("\n[bold blue]===== CasaLingua Enhanced Spanish-English Translation Test =====[/bold blue]")
    
    # Load configuration
    config = load_config()
    
    # Detect hardware
    hardware_detector = HardwareDetector(config)
    hardware_info = hardware_detector.detect_all()
    
    # Create model loader and manager
    model_loader = ModelLoader(config)
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    # Test cases with problematic Spanish to English examples
    test_cases = [
        {
            "name": "Baseline Test Case",
            "text": "Estoy muy feliz de conocerte hoy. El clima es hermoso y espero que estés bien.",
            "domain": None,
            "formality": None,
            "enhance_prompts": True
        },
        {
            "name": "Legal Domain",
            "text": "El inquilino debe pagar el depósito de seguridad antes de ocupar la vivienda, según lo establecido en el contrato.",
            "domain": "legal",
            "formality": "formal",
            "enhance_prompts": True
        },
        {
            "name": "Technical Domain",
            "text": "El servidor procesa las solicitudes en paralelo para mejorar el rendimiento del sistema.",
            "domain": "technical",
            "formality": "neutral",
            "enhance_prompts": True
        },
        {
            "name": "Casual Conversation",
            "text": "¡Hola! ¿Cómo estás? ¿Quieres salir a comer algo más tarde?",
            "domain": "casual",
            "formality": "informal",
            "enhance_prompts": True
        },
        {
            "name": "Without Enhancement (Control)",
            "text": "Estoy muy feliz de conocerte hoy. El clima es hermoso y espero que estés bien.",
            "domain": None,
            "formality": None,
            "enhance_prompts": False
        },
    ]
    
    # Try to load the translation model
    try:
        console.print("[bold cyan]Loading translation model...[/bold cyan]")
        model_data = await model_manager.load_model("translation")
        
        if model_data and model_data.get("model"):
            model = model_data["model"]
            tokenizer = model_data.get("tokenizer")
            
            console.print(f"[bold green]Model loaded successfully: {type(model).__name__}[/bold green]")
            
            # Get model type for prompt enhancement
            model_type = "mt5"
            if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                model_name = model.config._name_or_path.lower()
                if "mbart" in model_name:
                    model_type = "mbart"
            
            console.print(f"Detected model type: {model_type}")
            
            # Process test cases
            for i, test_case in enumerate(test_cases, 1):
                console.print(f"\n[bold]Test Case {i}: {test_case['name']}[/bold]")
                console.print(f"[bold cyan]Spanish:[/bold cyan] '{test_case['text']}'")
                
                # Create wrapper for translation
                wrapper_config = config.get("translation", {}).copy()
                
                # Add test case parameters
                wrapper_config["domain"] = test_case["domain"]
                wrapper_config["formality"] = test_case["formality"]
                
                # Create translation wrapper
                wrapper = TranslationModelWrapper(model, tokenizer, wrapper_config)
                
                # Create input data
                input_data = ModelInput(
                    text=test_case["text"],
                    source_language="es",
                    target_language="en",
                    parameters={
                        "enhance_prompts": test_case["enhance_prompts"],
                        "domain": test_case["domain"],
                        "formality": test_case["formality"]
                    }
                )
                
                # Process translation
                console.print(f"[bold]Processing with {'enhanced' if test_case['enhance_prompts'] else 'standard'} prompts...[/bold]")
                try:
                    start_time = time.time()
                    output = await wrapper.process(input_data)
                    process_time = time.time() - start_time
                    
                    result = output.result
                    if isinstance(result, list) and result:
                        result = result[0]
                    
                    console.print(f"[bold green]English Translation:[/bold green] '{result}'")
                    console.print(f"[bold]Processing time:[/bold] {process_time:.3f}s")
                    
                    # Check for problematic Czech words
                    czech_words = ["Jsem", "velmi", "že", "vás", "poznávám", "dnes", "Těší", "mě", "rád"]
                    has_czech = any(word in result for word in czech_words)
                    
                    if has_czech:
                        console.print("[bold red]ERROR: Translation contains Czech words![/bold red]")
                    else:
                        console.print("[bold green]SUCCESS: No Czech words detected[/bold green]")
                    
                    # Check if Spanish words remain in the result
                    spanish_words = ["muy", "feliz", "conocerte", "hoy", "clima", "hermoso", "espero", "que", "estés", "bien"]
                    has_spanish = any(word in result for word in spanish_words)
                    
                    if has_spanish:
                        console.print("[bold yellow]WARNING: Translation may contain untranslated Spanish words[/bold yellow]")
                    
                    # Add a separator between test cases
                    console.print("-" * 50)
                    
                except Exception as e:
                    console.print(f"[bold red]Error in translation: {str(e)}[/bold red]")
                    import traceback
                    console.print(traceback.format_exc())
        else:
            console.print("[bold red]Could not load translation model[/bold red]")
    
    except Exception as e:
        console.print(f"[bold red]Error during model loading: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    
    console.print("\n[bold blue]===== Enhanced Spanish-English Translation Test Complete =====[/bold blue]")

if __name__ == "__main__":
    asyncio.run(test_enhanced_spanish_english())