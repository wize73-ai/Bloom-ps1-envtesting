#!/usr/bin/env python3
"""
Enhanced wrapper test script
Tests the improved TranslationModelWrapper with various types of models
"""

import os
import sys
import torch
import logging
from typing import List, Dict, Any
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhanced_wrapper_test")

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

# Import the necessary modules
try:
    from app.services.models.wrapper import TranslationModelWrapper, ModelInput, ModelType
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MT5ForConditionalGeneration, MT5TokenizerFast
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

async def test_with_diverse_models():
    """Test the enhanced model wrapper with different model types"""
    
    logger.info("Testing enhanced TranslationModelWrapper with diverse models")
    
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog."
    source_lang = "en"
    target_lang = "es"
    
    # Keep track of results
    results = []
    
    # Test various model types
    try:
        # 1. Test with MBART model
        logger.info("Testing with MBART model...")
        try:
            # Load MBART model
            mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            
            # Create wrapper
            mbart_wrapper = TranslationModelWrapper(
                model=mbart_model,
                tokenizer=mbart_tokenizer,
                config={"task": ModelType.TRANSLATION}
            )
            
            # Test translation
            mbart_input = ModelInput(
                text=test_text,
                source_language=source_lang,
                target_language=target_lang,
                parameters={
                    "mbart_source_lang": mbart_wrapper._get_mbart_language_code(source_lang),
                    "mbart_target_lang": mbart_wrapper._get_mbart_language_code(target_lang)
                }
            )
            
            mbart_result = mbart_wrapper.process(mbart_input)
            logger.info(f"MBART result: {mbart_result.result}")
            
            results.append({
                "model": "MBART",
                "text": test_text,
                "translation": mbart_result.result,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"MBART test failed: {e}")
            results.append({
                "model": "MBART",
                "text": test_text,
                "error": str(e),
                "success": False
            })
        
        # 2. Test with MT5 model
        logger.info("Testing with MT5 model...")
        try:
            # Load MT5 model
            mt5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
            mt5_tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-small")
            
            # Create wrapper
            mt5_wrapper = TranslationModelWrapper(
                model=mt5_model,
                tokenizer=mt5_tokenizer,
                config={"task": ModelType.TRANSLATION}
            )
            
            # Test translation
            mt5_input = ModelInput(
                text=test_text,
                source_language=source_lang,
                target_language=target_lang
            )
            
            mt5_result = mt5_wrapper.process(mt5_input)
            logger.info(f"MT5 result: {mt5_result.result}")
            
            results.append({
                "model": "MT5",
                "text": test_text,
                "translation": mt5_result.result,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"MT5 test failed: {e}")
            results.append({
                "model": "MT5",
                "text": test_text,
                "error": str(e),
                "success": False
            })
        
        # 3. Test with a custom model with no standard interface
        logger.info("Testing with custom model...")
        try:
            # Create a minimal custom model class
            class CustomTranslationModel:
                def __init__(self):
                    self.name = "CustomModel"
                
                # No standard interfaces like generate, forward, etc.
                
            custom_model = CustomTranslationModel()
            
            # Create wrapper
            custom_wrapper = TranslationModelWrapper(
                model=custom_model,
                tokenizer=None,
                config={"task": ModelType.TRANSLATION}
            )
            
            # Test translation
            custom_input = ModelInput(
                text=test_text,
                source_language=source_lang,
                target_language=target_lang
            )
            
            # This should try multiple fallback approaches and eventually
            # throw a controlled "Unsupported translation model" error
            try:
                custom_result = custom_wrapper.process(custom_input)
                logger.info(f"Custom model result: {custom_result.result}")
                
                results.append({
                    "model": "CustomModel",
                    "text": test_text,
                    "translation": custom_result.result,
                    "success": True
                })
            except Exception as e:
                logger.info(f"Expected error from custom model: {e}")
                results.append({
                    "model": "CustomModel",
                    "text": test_text,
                    "error": str(e),
                    "success": False
                })
        
        except Exception as e:
            logger.error(f"Custom model test setup failed: {e}")
            results.append({
                "model": "CustomModel",
                "text": test_text,
                "error": str(e),
                "success": False
            })
    
    except Exception as e:
        logger.error(f"Overall test failed: {e}")
    
    # Print results
    logger.info("\nTest Results Summary:")
    logger.info("=====================")
    for result in results:
        if result.get("success", False):
            logger.info(f"Model: {result['model']} - SUCCESS")
            logger.info(f"  Source: {result['text']}")
            logger.info(f"  Translation: {result['translation']}")
        else:
            logger.info(f"Model: {result['model']} - FAILED")
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
        logger.info("")
        
    return results

if __name__ == "__main__":
    asyncio.run(test_with_diverse_models())