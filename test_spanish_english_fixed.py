#!/usr/bin/env python3
"""
Test script for Spanish to English translation after applying the fixes.

This script tests the Spanish to English translation directly using the API endpoint
and also by directly using the model wrapper to ensure both paths work correctly.
"""

import os
import sys
import json
import logging
import time
import argparse
import asyncio
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_translation_endpoint():
    """Test the translation endpoint for Spanish to English translation."""
    server_url = "http://localhost:5000"
    endpoint = f"{server_url}/pipeline/translate"
    test_texts = [
        "Estoy muy feliz de conocerte hoy.",
        "Hola, ¿cómo estás? Espero que tengas un buen día.",
        "El tiempo está muy agradable hoy en la ciudad."
    ]
    
    results = []
    success_count = 0
    
    for i, text in enumerate(test_texts):
        try:
            # Prepare the request
            payload = {
                "text": text,
                "source_language": "es",
                "target_language": "en"
            }
            
            # Make the request
            logger.info(f"Testing translation endpoint with text {i+1}: '{text}'")
            response = requests.post(endpoint, json=payload)
            
            # Check the response
            if response.status_code == 200:
                result = response.json()
                translation = result.get("result", "")
                logger.info(f"Translation result: '{translation}'")
                results.append({
                    "source": text,
                    "translation": translation,
                    "success": True
                })
                success_count += 1
            else:
                error = response.text
                logger.error(f"API error: {error}")
                results.append({
                    "source": text,
                    "error": error,
                    "success": False
                })
        except Exception as e:
            logger.error(f"Error testing endpoint: {e}")
            results.append({
                "source": text,
                "error": str(e),
                "success": False
            })
    
    # Return results
    return success_count == len(test_texts), {
        "success_rate": f"{success_count}/{len(test_texts)}",
        "results": results
    }

async def test_translation_wrapper():
    """Test the translation wrapper directly for Spanish to English translation."""
    try:
        # Import required modules
        from app.services.models.wrapper import TranslationModelWrapper, ModelInput
        from app.services.models.loader import get_model_loader
        
        # Test texts
        test_texts = [
            "Estoy muy feliz de conocerte hoy.",
            "Hola, ¿cómo estás? Espero que tengas un buen día.",
            "El tiempo está muy agradable hoy en la ciudad."
        ]
        
        results = []
        success_count = 0
        
        # Get the model loader
        loader = get_model_loader()
        
        # Load translation model
        logger.info("Loading translation model...")
        model_info = loader.load_model("translation")
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        config = model_info["config"]
        
        # Check if this is an MBART model
        is_mbart = False
        if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path.lower()
            is_mbart = "mbart" in model_name
        
        device = model.device if hasattr(model, "device") else "unknown"
        logger.info(f"Model type: {'MBART' if is_mbart else 'Other (likely MT5)'}, Device: {device}")
        
        # Create wrapper
        wrapper = TranslationModelWrapper(model, tokenizer, config)
        
        # Test each text
        for i, text in enumerate(test_texts):
            try:
                # Create model input
                input_data = ModelInput(
                    text=text,
                    source_language="es",
                    target_language="en"
                )
                
                # Process translation
                logger.info(f"Testing wrapper with text {i+1}: '{text}'")
                result = await wrapper.process(input_data)
                
                # Check result
                translation = result.result
                logger.info(f"Translation result: '{translation}'")
                results.append({
                    "source": text,
                    "translation": translation,
                    "success": True
                })
                success_count += 1
            except Exception as e:
                logger.error(f"Error with wrapper translation: {e}")
                results.append({
                    "source": text,
                    "error": str(e),
                    "success": False
                })
        
        # Return results
        return success_count == len(test_texts), {
            "success_rate": f"{success_count}/{len(test_texts)}",
            "is_mbart": is_mbart,
            "device": str(device),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in wrapper test: {e}")
        return False, {"error": str(e)}

def main():
    """Run both translation tests."""
    parser = argparse.ArgumentParser(description="Test Spanish to English translation")
    parser.add_argument("--endpoint", action="store_true", help="Test the API endpoint")
    parser.add_argument("--wrapper", action="store_true", help="Test the model wrapper directly")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    # Determine which tests to run
    run_endpoint = args.endpoint or args.all
    run_wrapper = args.wrapper or args.all
    
    # If no specific tests were selected, run all
    if not run_endpoint and not run_wrapper:
        run_endpoint = True
        run_wrapper = True
    
    # Results dictionary
    results = {}
    overall_success = True
    
    # Test API endpoint if selected
    if run_endpoint:
        logger.info("\n====== Testing Translation API Endpoint ======")
        endpoint_success, endpoint_results = test_translation_endpoint()
        results["endpoint_test"] = endpoint_results
        overall_success = overall_success and endpoint_success
        
        logger.info(f"API Endpoint Test: {'✅ PASSED' if endpoint_success else '❌ FAILED'}")
        logger.info(f"Success rate: {endpoint_results['success_rate']}")
    
    # Test model wrapper directly if selected
    if run_wrapper:
        logger.info("\n====== Testing Translation Model Wrapper ======")
        wrapper_success, wrapper_results = asyncio.run(test_translation_wrapper())
        results["wrapper_test"] = wrapper_results
        overall_success = overall_success and wrapper_success
        
        logger.info(f"Model Wrapper Test: {'✅ PASSED' if wrapper_success else '❌ FAILED'}")
        if 'success_rate' in wrapper_results:
            logger.info(f"Success rate: {wrapper_results['success_rate']}")
            if 'is_mbart' in wrapper_results:
                logger.info(f"Model type: {'MBART' if wrapper_results['is_mbart'] else 'Other (likely MT5)'}")
                logger.info(f"Device: {wrapper_results['device']}")
    
    # Write results to file
    with open("spanish_english_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print overall result
    logger.info("\n====== Test Summary ======")
    logger.info(f"Overall Test Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    logger.info(f"Detailed results saved to spanish_english_test_results.json")
    
    # Output formatted results for reading
    print("\n\n===== TEST RESULTS =====")
    print(f"Overall: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if run_endpoint:
        print("\nAPI Endpoint Test:")
        print(f"  Result: {'✅ PASSED' if endpoint_success else '❌ FAILED'}")
        print(f"  Success rate: {endpoint_results['success_rate']}")
        
        if 'results' in endpoint_results:
            for i, result in enumerate(endpoint_results['results']):
                print(f"\n  Text {i+1}:")
                print(f"    Source: {result['source']}")
                if result['success']:
                    print(f"    Translation: {result['translation']}")
                else:
                    print(f"    Error: {result.get('error', 'Unknown error')}")
    
    if run_wrapper:
        print("\nModel Wrapper Test:")
        print(f"  Result: {'✅ PASSED' if wrapper_success else '❌ FAILED'}")
        
        if 'success_rate' in wrapper_results:
            print(f"  Success rate: {wrapper_results['success_rate']}")
            if 'is_mbart' in wrapper_results:
                print(f"  Model type: {'MBART' if wrapper_results['is_mbart'] else 'Other (likely MT5)'}")
                print(f"  Device: {wrapper_results['device']}")
            
            if 'results' in wrapper_results:
                for i, result in enumerate(wrapper_results['results']):
                    print(f"\n  Text {i+1}:")
                    print(f"    Source: {result['source']}")
                    if result['success']:
                        print(f"    Translation: {result['translation']}")
                    else:
                        print(f"    Error: {result.get('error', 'Unknown error')}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())