#!/usr/bin/env python
"""
Test script to verify translation quality with upgraded models using wrappers
"""

import os
import sys
import time
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wrapper_quality_test")

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

def test_translation_quality():
    """Test translation quality with upgraded MT5 and MBART models using direct model loading and wrappers"""
    from app.services.models.loader import get_model_loader
    from app.services.models.wrapper import create_model_wrapper, ModelInput
    
    # Initialize model loader
    logger.info("Initializing model loader...")
    loader = get_model_loader()
    
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
    
    # Load models
    mt5_info = loader.load_model("translation", device="cpu")
    mbart_info = loader.load_model("mbart_translation", device="cpu")
    
    # Create wrappers
    mt5_wrapper = create_model_wrapper(
        "translation", 
        mt5_info["model"], 
        mt5_info["tokenizer"],
        {"task": "translation", "device": "cpu"}
    )
    
    mbart_wrapper = create_model_wrapper(
        "mbart_translation", 
        mbart_info["model"], 
        mbart_info["tokenizer"],
        {"task": "mbart_translation", "device": "cpu"}
    )
    
    logger.info(f"Using MT5 model: {mt5_info['config'].model_name}")
    logger.info(f"Using MBART model: {mbart_info['config'].model_name}")
    
    # Test both models on all test cases
    results = {"mt5": {}, "mbart": {}}
    
    for test_case in test_cases:
        text = test_case["text"]
        source = test_case["source"]
        target = test_case["target"]
        description = test_case["description"]
        
        logger.info(f"\nTesting: {description}")
        logger.info(f"Original ({source}): {text}")
        
        # Create input objects
        mt5_input = ModelInput(text=text, source_language=source, target_language=target)
        mbart_input = ModelInput(text=text, source_language=source, target_language=target)
        
        # Translate with MT5
        mt5_start = time.time()
        mt5_output = mt5_wrapper.process(mt5_input)
        mt5_time = time.time() - mt5_start
        
        # Extract MT5 result
        mt5_translation = mt5_output.result
        
        logger.info(f"MT5 ({mt5_info['config'].model_name}) [{mt5_time:.2f}s]: {mt5_translation}")
        results["mt5"][description] = {
            "original": text,
            "translation": mt5_translation,
            "time": mt5_time
        }
        
        # Translate with MBART
        mbart_start = time.time()
        mbart_output = mbart_wrapper.process(mbart_input)
        mbart_time = time.time() - mbart_start
        
        # Extract MBART result
        mbart_translation = mbart_output.result
        
        logger.info(f"MBART ({mbart_info['config'].model_name}) [{mbart_time:.2f}s]: {mbart_translation}")
        results["mbart"][description] = {
            "original": text,
            "translation": mbart_translation,
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
    logger.info(f"MT5 ({mt5_info['config'].model_name}): {sum(mt5_times)/len(mt5_times):.2f}s")
    logger.info(f"MBART ({mbart_info['config'].model_name}): {sum(mbart_times)/len(mbart_times):.2f}s")

if __name__ == "__main__":
    test_translation_quality()