#!/usr/bin/env python
"""
Test script to verify translation quality with upgraded models
"""

import os
import sys
import time
import logging
import argparse
import json
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_quality_test")

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

def test_translation_quality():
    """Test translation quality with upgraded MT5 and MBART models"""
    # Import here to avoid circular imports
    from services.models.loader import get_model_loader
    
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
    
    # Initialize model loader
    logger.info("Initializing model loader...")
    loader = get_model_loader()
    
    # Load models
    logger.info("Loading MT5 model...")
    mt5_result = loader.load_model("translation")
    mt5_model = mt5_result["model"]
    mt5_tokenizer = mt5_result["tokenizer"]
    logger.info(f"Loaded MT5 model: {mt5_result['config'].model_name}")
    
    logger.info("Loading MBART model...")
    mbart_result = loader.load_model("mbart_translation")
    mbart_model = mbart_result["model"]
    mbart_tokenizer = mbart_result["tokenizer"]
    logger.info(f"Loaded MBART model: {mbart_result['config'].model_name}")
    
    # Test both models on all test cases
    results = {"mt5": {}, "mbart": {}}
    
    for test_case in test_cases:
        text = test_case["text"]
        source = test_case["source"]
        target = test_case["target"]
        description = test_case["description"]
        
        logger.info(f"\nTesting: {description}")
        logger.info(f"Original ({source}): {text}")
        
        # Prepare MT5 input
        mt5_input = f"{source} to {target}: {text}"
        mt5_inputs = mt5_tokenizer(mt5_input, return_tensors="pt")
        if torch.cuda.is_available():
            mt5_inputs = {k: v.cuda() for k, v in mt5_inputs.items()}
        
        # Generate MT5 translation
        mt5_start = time.time()
        mt5_outputs = mt5_model.generate(
            **mt5_inputs,
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
        mt5_time = time.time() - mt5_start
        mt5_translation = mt5_tokenizer.decode(mt5_outputs[0], skip_special_tokens=True)
        
        logger.info(f"MT5 ({mt5_result['config'].model_name}) [{mt5_time:.2f}s]: {mt5_translation}")
        results["mt5"][description] = {
            "original": text,
            "translation": mt5_translation,
            "time": mt5_time
        }
        
        # Prepare MBART input
        mbart_tokenizer.src_lang = f"{source}_XX"
        mbart_inputs = mbart_tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            mbart_inputs = {k: v.cuda() for k, v in mbart_inputs.items()}
        
        # Generate MBART translation
        mbart_start = time.time()
        mbart_outputs = mbart_model.generate(
            **mbart_inputs,
            forced_bos_token_id=mbart_tokenizer.lang_code_to_id[f"{target}_XX"],
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
        mbart_time = time.time() - mbart_start
        mbart_translation = mbart_tokenizer.decode(mbart_outputs[0], skip_special_tokens=True)
        
        logger.info(f"MBART ({mbart_result['config'].model_name}) [{mbart_time:.2f}s]: {mbart_translation}")
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
    logger.info(f"MT5 ({mt5_result['config'].model_name}): {sum(mt5_times)/len(mt5_times):.2f}s")
    logger.info(f"MBART ({mbart_result['config'].model_name}): {sum(mbart_times)/len(mbart_times):.2f}s")

def main():
    """Main test function"""
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    test_translation_quality()

if __name__ == "__main__":
    main()