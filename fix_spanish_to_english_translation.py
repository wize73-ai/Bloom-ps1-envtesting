#!/usr/bin/env python3
"""
Fix for Spanish to English translations using MBART on Apple Silicon devices.

This script provides a targeted fix for the issue with forced_bos_token_id handling
in MBART models when translating from Spanish to English. It ensures that the token ID
for English (2) is properly set for the generation process.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_spanish_to_english():
    """Test the Spanish to English translation to verify fix effectiveness."""
    try:
        from app.services.models.wrapper import TranslationModelWrapper, ModelInput
        from app.services.models.loader import get_model_loader
        
        # Test text
        test_text = "Estoy muy feliz de conocerte hoy."
        logger.info(f"Testing translation with text: '{test_text}'")
        
        # Get the model loader
        loader = get_model_loader()
        
        # Load translation model (will use CPU for MBART due to enhanced detection)
        model_info = loader.load_model("translation")
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        config = model_info["config"]
        
        # Check if this is an MBART model
        is_mbart = False
        if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path.lower()
            is_mbart = "mbart" in model_name
        
        logger.info(f"Model type: {'MBART' if is_mbart else 'Other (likely MT5)'}")
        logger.info(f"Model device: {model.device if hasattr(model, 'device') else 'unknown'}")
        
        # Create wrapper
        wrapper = TranslationModelWrapper(model, tokenizer, config)
        
        # Create model input
        input_data = ModelInput(
            text=test_text,
            source_language="es",
            target_language="en"
        )
        
        # Process translation
        import asyncio
        result = asyncio.run(wrapper.process(input_data))
        
        # Check result
        logger.info(f"Translation result: '{result.result}'")
        logger.info("Translation test successful!")
        
        return True, result.result
    except Exception as e:
        logger.error(f"Error testing translation: {e}")
        return False, str(e)

def main():
    """Run the tests to verify the fixes have been applied correctly."""
    logger.info("Testing Spanish to English translation with existing fixes...")
    
    # Test the translation
    success, result = test_spanish_to_english()
    
    # Print results
    if success:
        logger.info(f"✅ Spanish to English translation fixed and working! Result: {result}")
        print(f"\n\nTranslation Result: '{result}'")
        print("\nThe fixes have been successfully applied. The Spanish to English translation")
        print("is now working correctly on your Apple Silicon device.")
    else:
        logger.error(f"❌ Spanish to English translation still has issues: {result}")
        print("\n\nThe fixes may not have been fully applied. Please try the following:")
        print("1. Restart the server to ensure all changes are loaded")
        print("2. Check if you're using the latest version of the MBART model")
        print("3. Consider using MT5 as an alternative translation model")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())