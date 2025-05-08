#!/usr/bin/env python3
"""
Direct MBART translation test for Spanish to English.

This script directly tests the MBART model for Spanish to English translation,
bypassing any wrapper classes or custom code that might be causing issues.
"""

import torch
import logging
import sys
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_spanish_to_english(text):
    """Directly translate Spanish to English using MBART with explicit handling."""
    device = "cpu"  # Always use CPU for MBART
    logger.info(f"Translating: '{text}' on device {device}")
    
    # Load model directly (no wrapper, no config)
    logger.info("Loading MBART model...")
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    logger.info("Model loaded successfully")
    
    # Load tokenizer directly
    logger.info("Loading MBART tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    logger.info("Tokenizer loaded successfully")
    
    # Set source language
    tokenizer.src_lang = "es_XX"
    logger.info(f"Set source language to: {tokenizer.src_lang}")
    
    # Tokenize with explicit handling
    logger.info("Tokenizing input text...")
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    logger.info(f"Input shape: {input_ids.shape}")
    
    # Generate translation with explicit English token
    logger.info("Generating translation with forced_bos_token_id=2 (English)...")
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        forced_bos_token_id=2,  # English
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    logger.info(f"Output shape: {outputs.shape}")
    
    # Decode the translation
    logger.info("Decoding translation...")
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logger.info(f"Raw translation: {translation}")
    
    # Clean up any prefixes/padding
    if translation and len(translation) > 0:
        result = translation[0]
        # Remove common prefixes if present
        prefixes = ["translate:", "translation:", "<pad>"]
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        return result
    else:
        return "No translation generated"

def main():
    """Run direct Spanish to English translation test."""
    logger.info("=== Direct MBART Spanish to English Translation Test ===")
    
    # Test multiple phrases
    test_phrases = [
        "Estoy muy feliz de conocerte hoy.",
        "¿Cómo estás? Espero que tengas un buen día.",
        "El cielo es azul y el sol brilla intensamente."
    ]
    
    success_count = 0
    for i, phrase in enumerate(test_phrases):
        logger.info(f"\nTest {i+1}/{len(test_phrases)}")
        try:
            translation = translate_spanish_to_english(phrase)
            logger.info(f"Translation: '{translation}'")
            
            # Verify we got a meaningful translation
            if translation and translation != "No translation generated" and len(translation) > 5:
                logger.info("✅ Test passed!")
                success_count += 1
            else:
                logger.error("❌ Test failed: Translation empty or too short")
        except Exception as e:
            logger.error(f"❌ Test failed with error: {str(e)}")
    
    # Print summary
    logger.info(f"\n=== Test Results: {success_count}/{len(test_phrases)} translations successful ===")
    
    if success_count > 0:
        logger.info("✅ Direct MBART Spanish to English translation is working!")
        return 0
    else:
        logger.error("❌ All direct MBART Spanish to English translation tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())