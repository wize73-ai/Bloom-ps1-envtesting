#!/usr/bin/env python3
"""
Test for MT5-based Spanish to English translation fallback.

This script tests Spanish to English translation using the MT5 fallback
approach instead of MBART, which has issues on Apple Silicon hardware.
"""

import os
import sys
import logging
import torch
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mt5_spanish_to_english():
    """Test Spanish to English translation using MT5 model."""
    from app.services.models.wrapper import TranslationModelWrapper, ModelInput
    from app.services.models.loader import get_model_loader
    
    # Test text
    test_text = "Estoy muy feliz de conocerte hoy."
    logger.info(f"\n=== Testing Spanish->English with MT5 fallback: '{test_text}' ===")
    
    # Get the model loader
    loader = get_model_loader()
    
    # Load translation model
    logger.info("Loading translation model...")
    model_info = loader.load_model("translation")
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    config = model_info["config"]
    
    # Check which model was loaded
    if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
        model_name = model.config._name_or_path
        logger.info(f"Loaded model: {model_name}")
        if "mbart" in model_name.lower():
            logger.warning("⚠️ MBART model loaded - should be using MT5 for Spanish->English!")
        elif "mt5" in model_name.lower():
            logger.info("✅ MT5 model loaded correctly")
        else:
            logger.info(f"Other model type loaded: {model_name}")
    
    # Create wrapper
    logger.info("Creating TranslationModelWrapper...")
    wrapper = TranslationModelWrapper(model, tokenizer, config)
    
    # Create model input for Spanish to English
    input_data = ModelInput(
        text=test_text,
        source_language="es",
        target_language="en"
    )
    
    # Process translation
    logger.info("Running translation...")
    result = asyncio.run(wrapper.process(input_data))
    
    # Check result
    logger.info(f"Translation result: '{result.result}'")
    
    # Check if translation is available and not an error
    success = (
        result.result and
        result.result != "Translation unavailable" and
        "unavailable" not in result.result.lower() and
        "error" not in result.result.lower() and
        len(result.result) > 5
    )
    
    if success:
        logger.info("✅ Translation successful!")
        return True, result.result
    else:
        logger.error("❌ Translation failed or returned error message")
        return False, result.result

def main():
    """Run the MT5 Spanish to English translation test."""
    logger.info("=== Testing MT5 Fallback for Spanish to English Translation ===")
    
    # Test with MT5 fallback
    success, result = test_mt5_spanish_to_english()
    
    if success:
        logger.info("\n✅✅✅ MT5 FALLBACK FOR SPANISH TO ENGLISH WORKING SUCCESSFULLY ✅✅✅")
        logger.info(f"Translation result: '{result}'")
        return 0
    else:
        logger.error("\n❌❌❌ MT5 FALLBACK FOR SPANISH TO ENGLISH FAILED ❌❌❌")
        logger.error(f"Error result: '{result}'")
        return 1

if __name__ == "__main__":
    sys.exit(main())
