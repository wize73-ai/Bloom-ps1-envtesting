#!/usr/bin/env python3
"""
Test for MBART Translation Wrapper Fix

This script tests the fix for Spanish to English translation in the MBART model wrapper.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
from app.services.models.wrapper import TranslationModelWrapper, ModelInput
from app.services.models.model_manager import ModelManager

async def test_spanish_to_english_translation():
    """Test Spanish to English translation with the fixed wrapper"""
    logger.info("Starting test for Spanish to English translation")
    
    # Initialize model manager and load models
    logger.info("Initializing model manager")
    manager = ModelManager()
    await manager.initialize()
    
    # Wait for models to be loaded
    logger.info("Waiting for models to be loaded")
    await asyncio.sleep(2)
    
    # Get the mbart translation model
    mbart_model_name = "mbart_translation"  # This should match what's in your config
    logger.info(f"Loading {mbart_model_name} model")
    model_wrapper = await manager.get_model(mbart_model_name)
    
    if not model_wrapper:
        logger.error(f"Failed to load {mbart_model_name} model")
        return
    
    # Test data
    spanish_texts = [
        "Hola, estoy muy feliz de conocerte hoy.",
        "Esta es una prueba de traducción del español al inglés.",
        "Necesito que esto se traduzca correctamente."
    ]
    
    # Process each text
    logger.info("Testing Spanish to English translations")
    
    for spanish_text in spanish_texts:
        # Create model input with Spanish source and English target
        input_data = ModelInput(
            text=spanish_text,
            source_language="es",
            target_language="en",
            parameters={"domain": "general", "formality": "neutral"}
        )
        
        # Process through the wrapper
        logger.info(f"Translating: {spanish_text}")
        try:
            result = await model_wrapper.process(input_data)
            translation = result.result
            
            # Output the result
            logger.info(f"Original (es): {spanish_text}")
            logger.info(f"Translation (en): {translation}")
            
            # Check if the result is still in Spanish (indicating the bug)
            if any(word in translation.lower() for word in ["estoy", "soy", "hola", "esta", "es", "una", "prueba", "del", "necesito", "que", "esto", "traduzca"]):
                logger.error("❌ FAILED: Translation appears to still be in Spanish!")
            else:
                logger.info("✅ SUCCESS: Translation appears to be in English")
            
            # Log metrics if available
            if hasattr(result, 'metadata') and result.metadata:
                logger.info(f"Metadata: {result.metadata}")
            
            logger.info("-" * 40)
        except Exception as e:
            logger.error(f"Error processing translation: {e}", exc_info=True)
    
    # Clean up
    await manager.shutdown()
    logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_spanish_to_english_translation())