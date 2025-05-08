#!/usr/bin/env python3
# Test NLLB translation functionality
# This script tests the NLLB model with different language pairs

import os
import sys
import time
import json
import asyncio
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import necessary modules
from app.utils.config import load_config
from app.services.models.loader import ModelLoader

# Explicit import of the NLLB language mapping
sys.path.append(str(project_root / "app" / "core" / "pipeline"))
from nllb_language_mapping import get_nllb_code

async def test_translation():
    """Test NLLB translation functionality."""
    print("🔍 Testing NLLB translation...")
    
    # Load configuration
    config = load_config()
    
    # Create model loader
    loader = ModelLoader(config=config)
    
    # Load registry configuration
    registry_path = project_root / "config" / "model_registry.json"
    with open(registry_path, "r") as f:
        registry = json.load(f)
    
    # Load NLLB model
    print("⏳ Loading NLLB translation model...")
    loader.model_config = registry
    translation_model = await loader.load_model("translation")
    
    if translation_model is None:
        print("❌ Failed to load NLLB translation model")
        return
    
    # Extract model and tokenizer
    model = translation_model.get("model")
    tokenizer = translation_model.get("tokenizer")
    
    if model is None or tokenizer is None:
        print("❌ Model or tokenizer is None")
        return
    
    print(f"✅ Successfully loaded NLLB model: {model.__class__.__name__}")
    print(f"✅ Device: {next(model.parameters()).device}")
    
    # Test translations
    test_translations = [
        {"text": "Hello, how are you?", "source": "en", "target": "es"},
        {"text": "This is a test of the NLLB translation model.", "source": "en", "target": "fr"},
        {"text": "I hope this works correctly.", "source": "en", "target": "de"},
        {"text": "The quick brown fox jumps over the lazy dog.", "source": "en", "target": "es"}
    ]
    
    for i, test in enumerate(test_translations):
        text = test["text"]
        source = test["source"]
        target = test["target"]
        
        # Convert to NLLB format
        source_nllb = get_nllb_code(source)
        target_nllb = get_nllb_code(target)
        
        print(f"\n🔄 Test {i+1}: Translating from {source} ({source_nllb}) to {target} ({target_nllb})\n")
        print(f"Input: {text}")
        
        # Set source language
        tokenizer.src_lang = source_nllb
        
        # Prepare the input
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        
        # Force target language
        forced_bos_token_id = tokenizer.lang_code_to_id[target_nllb]
        
        # Generate translation
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )
        translation_time = time.time() - start_time
        
        # Decode translation
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(f"Output: {translation}")
        print(f"Translation completed in {translation_time:.3f} seconds")

if __name__ == "__main__":
    asyncio.run(test_translation())
