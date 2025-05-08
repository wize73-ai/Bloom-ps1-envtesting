#!/usr/bin/env python3
"""
Test script for model-specific translation prompt enhancement

This script demonstrates the advanced model-aware translation prompt enhancement
in CasaLingua, showing how different models are handled with tailored prompts.
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from typing import Dict, Any, List

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary components
from app.utils.config import load_config
from app.services.models.loader import ModelLoader
from app.services.hardware.detector import HardwareDetector
from app.services.models.manager import EnhancedModelManager
from app.services.models.wrapper import TranslationModelWrapper, ModelInput
from app.services.models.translation_prompt_enhancer import TranslationPromptEnhancer
from app.ui.console import Console

# Initialize console for pretty output
console = Console()

async def test_model_specific_translation():
    """Test model-specific translation prompt enhancements"""
    console.print("\n[bold blue]===== CasaLingua Model-Specific Translation Enhancement Test =====[/bold blue]")
    
    # Create a prompt enhancer 
    prompt_enhancer = TranslationPromptEnhancer()
    
    # Define test models with different capabilities and instruction styles
    test_models = [
        {"name": "mt5-base", "type": "mt5", "instruction_style": "detailed"},
        {"name": "mt5-small", "type": "mt5", "instruction_style": "detailed"},
        {"name": "mbart-large-50-many-to-many-mmt", "type": "mbart", "instruction_style": "minimal"}
    ]
    
    # Define test language pairs with varying difficulty
    test_language_pairs = [
        {"source": "es", "target": "en", "description": "Spanish to English (challenging)"},
        {"source": "en", "target": "fr", "description": "English to French (straightforward)"},
        {"source": "en", "target": "zh", "description": "English to Chinese (very different)"}
    ]
    
    # Define test domains
    test_domains = [
        {"name": "legal", "text": "El inquilino debe pagar el depósito de seguridad antes de ocupar la vivienda."},
        {"name": "technical", "text": "Configure the database connection parameters in the configuration file."},
        {"name": "casual", "text": "Hey, how's it going? Want to grab lunch later today?"}
    ]
    
    # Define formality levels
    test_formality_levels = ["formal", "informal", "neutral"]
    
    # Test model-specific MT5 prompt generation
    console.print("\n[bold cyan]Testing Model-Specific MT5 Prompt Generation[/bold cyan]")
    
    for model in test_models:
        if "mt5" not in model["type"]:
            continue
            
        console.print(f"\n[bold]Model: {model['name']} (instruction style: {model['instruction_style']})[/bold]")
        
        for language_pair in test_language_pairs:
            source_lang = language_pair["source"]
            target_lang = language_pair["target"]
            console.print(f"\n  [bold cyan]Language Pair: {language_pair['description']}[/bold cyan]")
            
            # Test different domains
            for domain in test_domains:
                console.print(f"\n    [bold]Domain: {domain['name']}[/bold]")
                
                # Generate domain-specific MT5 prompt
                enhanced_prompt = prompt_enhancer.enhance_mt5_prompt(
                    text=domain["text"],
                    source_lang=source_lang,
                    target_lang=target_lang,
                    domain=domain["name"],
                    formality="neutral",
                    parameters={"model_name": model["name"]}
                )
                
                console.print(f"    [cyan]Text:[/cyan] {domain['text']}")
                console.print(f"    [green]Enhanced Prompt:[/green] {enhanced_prompt[:300]}...")
                console.print(f"    [dim]Prompt Length: {len(enhanced_prompt.split())} words[/dim]")
                
                # Show model-specific characteristics
                capabilities = prompt_enhancer.get_model_capabilities(model["name"])
                proficiency = prompt_enhancer.get_language_pair_proficiency(
                    model["name"], source_lang, target_lang
                )
                domain_quality = prompt_enhancer.get_domain_quality_for_model(
                    model["name"], domain["name"]
                )
                
                console.print(f"    [magenta]Model Capabilities:[/magenta] {', '.join(capabilities.get('strengths', []))}")
                console.print(f"    [magenta]Language Pair Proficiency:[/magenta] {proficiency}/10.0")
                console.print(f"    [magenta]Domain Quality:[/magenta] {domain_quality}/10.0")
    
    # Test model-specific MBART generation parameters
    console.print("\n[bold cyan]Testing Model-Specific MBART Generation Parameters[/bold cyan]")
    
    for model in test_models:
        if "mbart" not in model["type"]:
            continue
            
        console.print(f"\n[bold]Model: {model['name']} (instruction style: {model['instruction_style']})[/bold]")
        
        for language_pair in test_language_pairs:
            source_lang = language_pair["source"]
            target_lang = language_pair["target"]
            console.print(f"\n  [bold cyan]Language Pair: {language_pair['description']}[/bold cyan]")
            
            # Test different domains and formality levels
            for domain in test_domains:
                for formality in test_formality_levels:
                    console.print(f"\n    [bold]Domain: {domain['name']}, Formality: {formality}[/bold]")
                    
                    # Generate model-specific MBART parameters
                    gen_params = prompt_enhancer.get_mbart_generation_params(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        domain=domain["name"],
                        formality=formality,
                        parameters={"model_name": model["name"]}
                    )
                    
                    console.print(f"    [green]Generation Parameters:[/green]")
                    for key, value in gen_params.items():
                        console.print(f"      {key}: {value}")
                    
                    # Generate MBART prompt prefix if needed
                    prefix = prompt_enhancer.create_domain_prompt_prefix(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        domain=domain["name"],
                        formality=formality,
                        model_name=model["name"]
                    )
                    
                    console.print(f"    [green]Prompt Prefix:[/green] {prefix[:200]}...")
    
    # Test end-to-end translation input enhancement
    console.print("\n[bold cyan]Testing End-to-End Translation Input Enhancement[/bold cyan]")
    
    for model in test_models:
        console.print(f"\n[bold]Model: {model['name']} (type: {model['type']})[/bold]")
        
        for language_pair in test_language_pairs:
            source_lang = language_pair["source"]
            target_lang = language_pair["target"]
            
            for domain in test_domains:
                console.print(f"\n  [bold cyan]Translating {source_lang} to {target_lang} ({domain['name']})[/bold cyan]")
                
                # Create input data
                input_data = {
                    "text": domain["text"],
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "parameters": {
                        "domain": domain["name"],
                        "formality": "neutral",
                        "model_name": model["name"],
                        "enhance_prompts": True
                    }
                }
                
                # Enhance input data for this model
                enhanced_input = prompt_enhancer.enhance_translation_input(
                    input_data=input_data,
                    model_type=model["type"]
                )
                
                # Show the enhanced input
                console.print(f"  [dim]Original Text:[/dim] {domain['text']}")
                
                # Show enhanced text or parameters depending on model type
                if "mt5" in model["type"]:
                    console.print(f"  [green]Enhanced Prompt:[/green] {enhanced_input['text'][:200]}...")
                elif "mbart" in model["type"]:
                    # Show generation parameters
                    if "parameters" in enhanced_input and "generation_kwargs" in enhanced_input["parameters"]:
                        console.print("  [green]Enhanced Generation Parameters:[/green]")
                        for key, value in enhanced_input["parameters"]["generation_kwargs"].items():
                            console.print(f"    {key}: {value}")
                    
                    # Show enhanced text if prefix was added
                    if enhanced_input.get("text") != domain["text"]:
                        console.print(f"  [green]Enhanced Text:[/green] {enhanced_input['text'][:200]}...")
                
                # Show enhanced metadata for all model types
                if "parameters" in enhanced_input:
                    console.print("  [magenta]Enhanced Metadata:[/magenta]")
                    for key, value in enhanced_input["parameters"].items():
                        if key != "generation_kwargs":
                            console.print(f"    {key}: {value}")

async def demonstrate_model_selection():
    """Demonstrate model selection based on language pair and domain"""
    console.print("\n[bold blue]===== CasaLingua Model Selection Demonstration =====[/bold blue]")
    
    # Create a prompt enhancer 
    prompt_enhancer = TranslationPromptEnhancer()
    
    # Show model proficiencies across language pairs
    console.print("\n[bold cyan]Model Proficiencies Across Language Pairs[/bold cyan]")
    
    language_pairs = ["en-es", "es-en", "en-fr", "fr-en", "en-de", "de-en", "en-zh", "zh-en"]
    models = ["mbart-large-50-many-to-many-mmt", "mt5-base"]
    
    # Create a proficiency table
    console.print("\n[bold]Language Pair Proficiencies (0-10 scale)[/bold]")
    console.print(f"{'Language Pair':<15} | {'MBART':<10} | {'MT5':<10} | {'Best Model':<15}")
    console.print("-" * 55)
    
    for lang_pair in language_pairs:
        source, target = lang_pair.split("-")
        mbart_score = prompt_enhancer.get_language_pair_proficiency(models[0], source, target)
        mt5_score = prompt_enhancer.get_language_pair_proficiency(models[1], source, target)
        
        best_model = "MBART" if mbart_score > mt5_score else "MT5"
        if abs(mbart_score - mt5_score) < 0.3:
            best_model = "Either"
            
        console.print(f"{lang_pair:<15} | {mbart_score:<10.1f} | {mt5_score:<10.1f} | {best_model:<15}")
    
    # Show domain preferences for models
    console.print("\n[bold cyan]Model Performance by Domain[/bold cyan]")
    
    domains = ["legal", "medical", "technical", "casual", "housing_legal"]
    
    console.print("\n[bold]Domain Quality by Model (0-10 scale)[/bold]")
    console.print(f"{'Domain':<15} | {'MBART':<10} | {'MT5':<10} | {'Best Model':<15}")
    console.print("-" * 55)
    
    for domain in domains:
        mbart_score = prompt_enhancer.get_domain_quality_for_model(models[0], domain)
        mt5_score = prompt_enhancer.get_domain_quality_for_model(models[1], domain)
        
        best_model = "MBART" if mbart_score > mt5_score else "MT5"
        if abs(mbart_score - mt5_score) < 0.3:
            best_model = "Either"
            
        console.print(f"{domain:<15} | {mbart_score:<10.1f} | {mt5_score:<10.1f} | {best_model:<15}")
    
    # Show recommended model for language pair + domain combinations
    console.print("\n[bold cyan]Recommended Models for Language Pair + Domain Combinations[/bold cyan]")
    
    test_cases = [
        {"source": "es", "target": "en", "domain": "legal"},
        {"source": "es", "target": "en", "domain": "casual"},
        {"source": "en", "target": "fr", "domain": "technical"},
        {"source": "zh", "target": "en", "domain": "medical"}
    ]
    
    console.print("\n[bold]Recommended Models for Specific Tasks[/bold]")
    console.print(f"{'Language Pair':<15} | {'Domain':<15} | {'Recommended Model':<25} | {'Instruction Style':<20}")
    console.print("-" * 80)
    
    for case in test_cases:
        source_lang = case["source"]
        target_lang = case["target"]
        domain = case["domain"]
        lang_pair = f"{source_lang}-{target_lang}"
        
        # Calculate language pair proficiency scores
        mbart_lang_score = prompt_enhancer.get_language_pair_proficiency(models[0], source_lang, target_lang)
        mt5_lang_score = prompt_enhancer.get_language_pair_proficiency(models[1], source_lang, target_lang)
        
        # Calculate domain quality scores
        mbart_domain_score = prompt_enhancer.get_domain_quality_for_model(models[0], domain)
        mt5_domain_score = prompt_enhancer.get_domain_quality_for_model(models[1], domain)
        
        # Combine scores with weights
        mbart_combined = (mbart_lang_score * 0.7) + (mbart_domain_score * 0.3)
        mt5_combined = (mt5_lang_score * 0.7) + (mt5_domain_score * 0.3)
        
        # Determine recommended model
        if mbart_combined > mt5_combined:
            recommended_model = "mbart-large-50-many-to-many-mmt"
            instruction_style = prompt_enhancer.get_model_instruction_style(recommended_model)
        else:
            recommended_model = "mt5-base"
            instruction_style = prompt_enhancer.get_model_instruction_style(recommended_model)
            
        console.print(f"{lang_pair:<15} | {domain:<15} | {recommended_model:<25} | {instruction_style:<20}")
        
    console.print("\n[bold blue]===== Model Selection Demonstration Complete =====[/bold blue]")

if __name__ == "__main__":
    if "--model-selection" in sys.argv:
        asyncio.run(demonstrate_model_selection())
    else:
        asyncio.run(test_model_specific_translation())