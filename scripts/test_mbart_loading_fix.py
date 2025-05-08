#!/usr/bin/env python3
"""
Test script to verify MBART model loading fixes
"""

import os
import sys
import json
import logging
import torch
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mbart_loading_test")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
try:
    from app.services.models.loader import ModelLoader, load_registry_config
    from app.services.models.wrapper import TranslationModelWrapper, ModelInput
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def main():
    """Test MBART model loading"""
    logger.info("Testing MBART model loading")
    
    # Load model registry config
    logger.info("Loading model registry config")
    registry_config = load_registry_config()
    
    # Create simple hardware info
    hardware_info = {
        "gpus": []
    }
    
    # Check for CUDA
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            gpu_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_available": torch.cuda.get_device_properties(i).total_memory * 0.8  # Assume 80% available
            }
            hardware_info["gpus"].append(gpu_info)
        logger.info(f"Found {device_count} CUDA devices")
    
    # Create loader
    logger.info("Creating ModelLoader")
    loader = ModelLoader(config=None, hardware_info=hardware_info)
    
    # Try to load MBART model
    logger.info("Attempting to load MBART translation model")
    try:
        model_info = loader.load_model("mbart_translation")
        logger.info(f"Successfully loaded MBART model: {model_info['status']}")
        
        # Now test the model with a simple example
        if model_info["model"] and model_info["tokenizer"]:
            # Create a TranslationModelWrapper
            wrapper = TranslationModelWrapper(
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                config={"task": "mbart_translation"}
            )
            
            # Test translation
            test_input = ModelInput(
                text="Hello, how are you?",
                source_language="en",
                target_language="es"
            )
            
            # Run translation
            logger.info("Testing translation...")
            output = wrapper.process(test_input)
            
            logger.info(f"Translation result: {output.result}")
            return 0
        else:
            logger.error("MBART model or tokenizer is None")
            return 1
            
    except Exception as e:
        logger.error(f"Failed to load MBART model: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())