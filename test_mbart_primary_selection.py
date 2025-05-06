#!/usr/bin/env python3
"""
Test script to verify MBART is used as the primary translation model by default
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mbart_primary")

# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Import the necessary components
from app.services.models.manager import ModelManager
from app.core.pipeline.translator import TranslationPipeline
from app.api.schemas.translation import TranslationRequest

async def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a file"""
    with open(config_path, 'r') as f:
        return json.load(f)

async def setup_model_manager() -> ModelManager:
    """Set up and initialize the model manager"""
    # Load model registry configuration
    registry_config = await load_config(os.path.join('config', 'model_registry.json'))
    
    # Create model manager
    model_manager = ModelManager(registry_config=registry_config)
    
    # Initialize model manager
    await model_manager.initialize()
    
    return model_manager

async def test_default_model_selection():
    """Test that MBART is selected by default for translation"""
    logger.info("Testing default model selection")
    
    # Setup model manager
    model_manager = await setup_model_manager()
    
    # Setup translation pipeline
    translator = TranslationPipeline(model_manager=model_manager)
    await translator.initialize()
    
    # Test cases
    test_cases = [
        {"text": "The quick brown fox jumps over the lazy dog.", "source": "en", "target": "es"},
        {"text": "Machine learning is transforming how we approach translation problems.", "source": "en", "target": "fr"},
        {"text": "This test verifies MBART is used by default.", "source": "en", "target": "de"}
    ]
    
    for i, test in enumerate(test_cases):
        logger.info(f"Test case {i+1}: {test['source']} to {test['target']}")
        logger.info(f"Source text: {test['text']}")
        
        # Perform translation - should use MBART by default
        result = await translator.translate_text(
            text=test["text"],
            source_language=test["source"],
            target_language=test["target"],
            model_id=None  # Default model selection
        )
        
        # Verify MBART is used
        logger.info(f"Translation: {result['translated_text']}")
        logger.info(f"Model used: {result.get('model_used', 'unknown')}")
        
        # Check if MBART was used (should be set to "mbart" in the model_used field)
        if result.get('model_used') == "mbart" or result.get('primary_model') == "mbart":
            logger.info("✅ Success: MBART was correctly used as the primary model")
        else:
            logger.error("❌ Error: MBART was not used as expected")
            logger.error(f"Model info: {result.get('model_used', 'unknown')}, Primary: {result.get('primary_model', 'unknown')}")
    
    logger.info("Testing with explicit MT5 model selection - should still use MBART")
    
    # Create request with explicit MT5 model selection
    test = test_cases[0]
    result = await translator.translate_text(
        text=test["text"],
        source_language=test["source"],
        target_language=test["target"],
        model_id="mt5_translation"  # Explicitly requesting MT5
    )
    
    # Verify MBART is still used (via override in translate_text)
    logger.info(f"Translation with explicit MT5 request: {result['translated_text']}")
    logger.info(f"Model used: {result.get('model_used', 'unknown')}")
    
    if result.get('model_used') == "mbart" or result.get('primary_model') == "mbart":
        logger.info("✅ Success: MBART was correctly used even when MT5 was explicitly requested")
    else:
        logger.error("❌ Error: MBART was not used when MT5 was explicitly requested")
        logger.error(f"Model info: {result.get('model_used', 'unknown')}, Primary: {result.get('primary_model', 'unknown')}")

async def main():
    # Run tests
    logger.info("Starting tests for MBART primary model selection")
    try:
        await test_default_model_selection()
        logger.info("All tests completed")
        return 0
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)