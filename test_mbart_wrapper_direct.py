#!/usr/bin/env python3
"""
Direct Test for MBART Spanish->English Translation Fix

This script directly tests the TranslationModelWrapper implementation
without requiring the full model loading infrastructure.
"""

import os
import sys
import logging
import torch
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
from app.services.models.wrapper import TranslationModelWrapper, ModelInput

class MockMBartModel:
    """Mock MBART model for testing"""
    
    def __init__(self):
        self.config = type('obj', (object,), {
            '_name_or_path': 'facebook/mbart-large-50-many-to-many-mmt',
            'name_or_path': 'facebook/mbart-large-50-many-to-many-mmt'
        })
    
    def to(self, device):
        """Move model to device"""
        return self
    
    def generate(self, **kwargs):
        """Mock generate method that returns token IDs for English text"""
        # If we're translating from Spanish to English (forced_bos_token_id=2 for English)
        if kwargs.get('forced_bos_token_id') == 2:
            # Return token IDs for "I am very happy to meet you today"
            return torch.tensor([[0, 2, 57, 33, 141, 589, 54, 129, 1025, 5, 3]])
        else:
            # Return token IDs in Spanish if not properly set
            # "Hola, estoy muy feliz de conocerte hoy"
            return torch.tensor([[0, 8, 154, 67, 413, 943, 24, 87, 1201, 14, 3]])

class MockMBartTokenizer:
    """Mock MBART tokenizer for testing"""
    
    def __init__(self):
        self.lang_code_to_id = {
            'en_XX': 2,  # English token ID
            'es_XX': 8,  # Spanish token ID
            'fr_XX': 6,  # French token ID
            'de_XX': 4   # German token ID
        }
    
    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None, src_lang=None):
        """Mock tokenizer call"""
        # Return mock token IDs
        return {
            'input_ids': torch.tensor([[0, 8, 154, 67, 413, 943, 24, 87, 1201, 14]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        }
    
    def batch_decode(self, token_ids, skip_special_tokens=None):
        """Mock decoder that returns text based on forced_bos_token_id"""
        # Check if it's English translation
        if token_ids[0][1].item() == 2:
            return ["I am very happy to meet you today."]
        else:
            return ["Hola, estoy muy feliz de conocerte hoy."]

def test_spanish_to_english_translation():
    """Test the Spanish to English translation with the wrapper fix"""
    logger.info("Starting direct test for Spanish to English translation")
    
    # Create mock model and tokenizer
    model = MockMBartModel()
    tokenizer = MockMBartTokenizer()
    
    # Create the wrapper with our mocks
    wrapper = TranslationModelWrapper(
        model=model,
        tokenizer=tokenizer,
        config={"device": "cpu", "max_length": 128}
    )
    
    # Test input in Spanish
    spanish_text = "Hola, estoy muy feliz de conocerte hoy."
    input_data = ModelInput(
        text=spanish_text,
        source_language="es",
        target_language="en",
        parameters={"domain": "general"}
    )
    
    # Test the preprocessing
    preprocessed = wrapper._preprocess(input_data)
    
    # Verify preprocessing set the correct language codes and detected MBART
    logger.info(f"Preprocessed data - is_mbart: {preprocessed.get('is_mbart', False)}")
    logger.info(f"Preprocessed data - is_special_lang_pair: {preprocessed.get('is_special_lang_pair', False)}")
    logger.info(f"Preprocessed data - source_lang_code: {preprocessed.get('source_lang_code', None)}")
    logger.info(f"Preprocessed data - target_lang_code: {preprocessed.get('target_lang_code', None)}")
    
    # Verify forced_bos_token_id is set to 2 (for English) in the inputs
    logger.info(f"forced_bos_token_id in inputs: {preprocessed['inputs'].get('forced_bos_token_id', None)}")
    
    # Test the inference
    inference_result = wrapper._run_inference(preprocessed)
    logger.info(f"Inference result: {inference_result}")
    
    # Test the postprocessing
    output = wrapper._postprocess(inference_result, input_data)
    
    # Check the translation result
    translation = output.result
    logger.info(f"Spanish input: {spanish_text}")
    logger.info(f"English output: {translation}")
    
    # Determine if the fix works
    is_english = "estoy" not in translation.lower() and "hola" not in translation.lower()
    if is_english:
        logger.info("✅ SUCCESS: Translation appears to be in English")
    else:
        logger.error("❌ FAILED: Translation appears to still be in Spanish!")
    
    return translation

if __name__ == "__main__":
    result = test_spanish_to_english_translation()
    print("\nTest Result:", result)
    print("\n✅ Test complete")