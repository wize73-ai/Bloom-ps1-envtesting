#!/usr/bin/env python
"""
Test script to verify the transformer import fixes in loader.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))

def test_model_loader():
    """Test the model loader with fix for 'UnboundLocalError: local variable 'AutoModel' referenced before assignment'"""
    from services.models.loader import get_model_loader, ModelConfig

    logger.info("Initializing model loader...")
    loader = get_model_loader()
    logger.info("Model loader initialized successfully")

    # Test loading various model types
    tests = [
        {"model_type": "language_detection", "expect_success": True},
        {"model_type": "translation", "expect_success": True},
        {"model_type": "rag_generator", "expect_success": True},
    ]

    for test in tests:
        model_type = test["model_type"]
        expect_success = test["expect_success"]
        
        try:
            logger.info(f"Testing loading of {model_type} model...")
            result = loader.load_model(model_type)
            
            if result["status"] == "loaded":
                logger.info(f"✓ Successfully loaded {model_type} model")
                if not expect_success:
                    logger.warning(f"Expected failure for {model_type} but it succeeded")
            else:
                logger.warning(f"× Model {model_type} not fully loaded, status: {result['status']}")
                if expect_success:
                    logger.warning(f"Expected success for {model_type} but it's not fully loaded")
                
        except Exception as e:
            logger.error(f"× Error loading {model_type}: {e}")
            if expect_success:
                logger.error(f"Expected success for {model_type} but it failed")
            else:
                logger.info(f"✓ Expected failure for {model_type} and it failed as expected")

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test transformer import fixes")
    parser.add_argument("--test", choices=["loader"], default="loader", help="Test to run")
    args = parser.parse_args()
    
    if args.test == "loader":
        test_model_loader()
    
    logger.info("Tests completed")

if __name__ == "__main__":
    main()