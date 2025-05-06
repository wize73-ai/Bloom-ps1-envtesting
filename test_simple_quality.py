#!/usr/bin/env python
"""
Simple test script to verify translation quality with upgraded models
"""

import os
import sys
import time
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("simple_quality_test")

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

def test_translation_quality():
    """Test translation quality with upgraded MT5 and MBART models using wrappers"""
    # Import here to avoid circular imports
    from app.services.models.wrapper import TranslationModelWrapper
    
    # Test cases
    test_cases = [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "source": "en",
            "target": "es",
            "description": "Simple sentence (EN->ES)"
        },
        {
            "text": "The new machine learning models demonstrate unprecedented levels of accuracy when trained on large, diverse datasets.",
            "source": "en",
            "target": "fr",
            "description": "Technical content (EN->FR)"
        },
        {
            "text": "Despite the challenges, the team managed to complete the project ahead of schedule and under budget, which impressed the stakeholders.",
            "source": "en",
            "target": "de",
            "description": "Complex sentence (EN->DE)"
        },
        {
            "text": "Her smile was as bright as the morning sun, warming everyone's hearts in the room.",
            "source": "en",
            "target": "it",
            "description": "Figurative language (EN->IT)"
        }
    ]
    
    # Init models
    logger.info("Testing MT5 model: google/mt5-base")
    mt5_wrapper = TranslationModelWrapper(model_type="translation", device="cpu")
    
    logger.info("Testing MBART model: facebook/mbart-large-50-many-to-many-mmt")
    mbart_wrapper = TranslationModelWrapper(model_type="mbart_translation", device="cpu")
    
    # Test both models on all test cases
    results = {"mt5": {}, "mbart": {}}
    
    for test_case in test_cases:
        text = test_case["text"]
        source = test_case["source"]
        target = test_case["target"]
        description = test_case["description"]
        
        logger.info(f"\nTesting: {description}")
        logger.info(f"Original ({source}): {text}")
        
        # Translate with MT5
        mt5_start = time.time()
        mt5_result = mt5_wrapper.translate(text, source, target)
        mt5_time = time.time() - mt5_start
        
        logger.info(f"MT5 (google/mt5-base) [{mt5_time:.2f}s]: {mt5_result}")
        results["mt5"][description] = {
            "original": text,
            "translation": mt5_result,
            "time": mt5_time
        }
        
        # Translate with MBART
        mbart_start = time.time()
        mbart_result = mbart_wrapper.translate(text, source, target)
        mbart_time = time.time() - mbart_start
        
        logger.info(f"MBART (facebook/mbart-large-50-many-to-many-mmt) [{mbart_time:.2f}s]: {mbart_result}")
        results["mbart"][description] = {
            "original": text,
            "translation": mbart_result,
            "time": mbart_time
        }
    
    # Save results to file
    with open("model_quality_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nQuality test results saved to model_quality_results.json")
    
    # Calculate and display averages
    mt5_times = [results["mt5"][desc]["time"] for desc in results["mt5"]]
    mbart_times = [results["mbart"][desc]["time"] for desc in results["mbart"]]
    
    logger.info(f"\nAverage translation times:")
    logger.info(f"MT5 (google/mt5-base): {sum(mt5_times)/len(mt5_times):.2f}s")
    logger.info(f"MBART (facebook/mbart-large-50-many-to-many-mmt): {sum(mbart_times)/len(mbart_times):.2f}s")

def main():
    """Main test function"""
    test_translation_quality()

if __name__ == "__main__":
    main()