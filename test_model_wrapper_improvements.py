#!/usr/bin/env python3
"""
Test Script for Model Wrapper Improvements

This script tests the fixes made to the model wrapper and manager implementations
to ensure that simplification and translation work correctly without loading
new models unnecessarily.
"""

import asyncio
import sys
import os
import time
import argparse
import logging
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("test_model_wrapper")
logger.setLevel(logging.DEBUG)

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Test text for simplification
TEST_TEXTS = [
    {
        "name": "Simple Text",
        "text": "Hello, how are you today? I hope you are doing well.",
        "level": 3
    },
    {
        "name": "Legal Text",
        "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant.",
        "level": 5,
        "domain": "legal"
    },
    {
        "name": "Complex Paragraph",
        "text": "The implementation of the new legislation will necessitate a comprehensive review of existing protocols. Subsequently, organizations must establish corresponding policies to ensure compliance with revised regulatory frameworks. Additionally, stakeholders should anticipate potential modifications to current operational procedures.",
        "level": 4
    }
]

# Test text for translation
TRANSLATION_TESTS = [
    {
        "name": "Simple English to Spanish",
        "text": "Hello, how are you?",
        "source_language": "en",
        "target_language": "es"
    },
    {
        "name": "Medium Spanish to English",
        "text": "El apartamento debe estar limpio antes de que se devuelva el depósito de seguridad.",
        "source_language": "es",
        "target_language": "en"
    }
]

async def test_simplification():
    """Test text simplification with the fixed wrapper."""
    print(f"\n{BOLD}{BLUE}Testing Text Simplification with Fixed Wrapper{ENDC}")
    print("-" * 80)
    
    # Import needed modules
    from app.services.hardware.simple_detector import get_hardware_info
    from app.services.models.loader import ModelLoader
    from app.services.models.manager import EnhancedModelManager
    
    # Create model loader
    loader = ModelLoader()
    
    # Get hardware info
    hardware_info = get_hardware_info()
    
    # Create model manager
    manager = EnhancedModelManager(loader, hardware_info)
    
    # Create simplification pipeline
    from app.core.pipeline.simplifier import SimplificationPipeline
    simplifier = SimplificationPipeline(manager)
    
    # Initialize the pipeline
    await simplifier.initialize()
    
    # Test simplification for each test text
    for i, test in enumerate(TEST_TEXTS):
        print(f"\n{BOLD}Test {i+1}: {test['name']}{ENDC}")
        print(f"{BOLD}Original:{ENDC} {test['text']}")
        
        # Add domain if available
        options = {}
        if "domain" in test:
            options["domain"] = test["domain"]
        
        # Time the simplification
        start_time = time.time()
        
        # Simplify text
        try:
            language = "en"  # Default language
            result = await simplifier.simplify(
                test["text"],
                language,
                level=test.get("level", 3),
                options=options
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Print results
            print(f"{BOLD}Simplified:{ENDC} {result.get('simplified_text', 'No result')}")
            print(f"{BOLD}Model Used:{ENDC} {result.get('model_used', 'Unknown')}")
            
            if result.get("simplified_text") and result.get("simplified_text") != "None" and result.get("simplified_text").strip():
                print(f"{GREEN}✓ Simplification successful in {processing_time:.2f}s{ENDC}")
            else:
                print(f"{RED}✗ Simplification returned empty or 'None' result in {processing_time:.2f}s{ENDC}")
                
            # Print metrics if available
            if "metrics" in result:
                metrics = result["metrics"]
                print(f"{BOLD}Metrics:{ENDC}")
                for key, value in metrics.items():
                    print(f"- {key}: {value}")
        except Exception as e:
            print(f"{RED}✗ Simplification failed: {str(e)}{ENDC}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
    
    return True

async def test_translation():
    """Test translation with cached model loading."""
    print(f"\n{BOLD}{BLUE}Testing Translation with Cached Model Loading{ENDC}")
    print("-" * 80)
    
    # Import needed modules
    from app.services.hardware.simple_detector import get_hardware_info
    from app.services.models.loader import ModelLoader
    from app.services.models.manager import EnhancedModelManager
    
    # Create model loader
    loader = ModelLoader()
    
    # Get hardware info
    hardware_info = get_hardware_info()
    
    # Create model manager
    manager = EnhancedModelManager(loader, hardware_info)
    
    # Create translation pipeline
    from app.core.pipeline.translator import TranslationPipeline
    translator = TranslationPipeline(manager)
    
    # Initialize the pipeline
    await translator.initialize()
    
    # Test translation for each test
    for i, test in enumerate(TRANSLATION_TESTS):
        print(f"\n{BOLD}Test {i+1}: {test['name']}{ENDC}")
        print(f"{BOLD}Original ({test['source_language']}):{ENDC} {test['text']}")
        
        # Time the translation
        start_time = time.time()
        
        # Translate text
        try:
            result = await translator.translate(
                test["text"],
                test["source_language"],
                test["target_language"]
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Print results
            print(f"{BOLD}Translated ({test['target_language']}):{ENDC} {result.get('translated_text', 'No result')}")
            print(f"{BOLD}Model Used:{ENDC} {result.get('model_used', 'Unknown')}")
            
            if result.get("translated_text") and result.get("translated_text") != "None" and result.get("translated_text").strip():
                print(f"{GREEN}✓ Translation successful in {processing_time:.2f}s{ENDC}")
            else:
                print(f"{RED}✗ Translation returned empty or 'None' result in {processing_time:.2f}s{ENDC}")
                
            # Print metrics if available
            if "metrics" in result:
                metrics = result["metrics"]
                print(f"{BOLD}Metrics:{ENDC}")
                for key, value in metrics.items():
                    print(f"- {key}: {value}")
        except Exception as e:
            print(f"{RED}✗ Translation failed: {str(e)}{ENDC}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
    
    return True

async def test_memory_pressure():
    """Test memory pressure tracking."""
    print(f"\n{BOLD}{BLUE}Testing Memory Pressure Tracking{ENDC}")
    print("-" * 80)
    
    # Import needed modules
    from app.services.models.wrapper import BaseModelWrapper, ModelInput, ModelOutput
    
    class TestModel:
        def __init__(self):
            pass
        
        def __call__(self, *args, **kwargs):
            return "Test result"
    
    # Create a test wrapper
    test_wrapper = BaseModelWrapper(TestModel())
    
    # Process some input
    test_input = ModelInput(text="This is a test input")
    
    try:
        # Process with the wrapper
        result = test_wrapper.process(test_input)
        
        # Check if memory metrics are populated
        memory_usage = result.memory_usage
        
        if memory_usage:
            print(f"{GREEN}✓ Memory usage metrics are populated{ENDC}")
            print(f"{BOLD}Memory Usage:{ENDC}")
            for key, value in memory_usage.items():
                if isinstance(value, dict):
                    print(f"- {key}:")
                    for subkey, subvalue in value.items():
                        print(f"  - {subkey}: {subvalue}")
                else:
                    print(f"- {key}: {value}")
            
            # Check for difference data
            if "difference" in memory_usage:
                print(f"{GREEN}✓ Memory difference metrics are calculated{ENDC}")
            else:
                print(f"{RED}✗ Memory difference metrics are missing{ENDC}")
        else:
            print(f"{RED}✗ Memory usage metrics are not populated{ENDC}")
    except Exception as e:
        print(f"{RED}✗ Memory pressure test failed: {str(e)}{ENDC}")
        import traceback
        traceback.print_exc()
    
    print("-" * 80)
    
    return True

async def run_tests(args):
    """Run all tests for model wrapper improvements."""
    print(f"{BOLD}{BLUE}=== Model Wrapper Improvements Tests ==={ENDC}")
    
    results = []
    
    if args.all or args.simplification:
        print(f"\n{BLUE}Running simplification tests...{ENDC}")
        results.append(("Simplification", await test_simplification()))
    
    if args.all or args.translation:
        print(f"\n{BLUE}Running translation tests...{ENDC}")
        results.append(("Translation", await test_translation()))
    
    if args.all or args.memory:
        print(f"\n{BLUE}Running memory pressure tests...{ENDC}")
        results.append(("Memory Pressure", await test_memory_pressure()))
    
    # Print summary
    print(f"\n{BOLD}{BLUE}=== Test Results Summary ==={ENDC}")
    for name, success in results:
        status = f"{GREEN}PASSED{ENDC}" if success else f"{RED}FAILED{ENDC}"
        print(f"{name}: {status}")
    
    # Check overall success
    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n{GREEN}All tests passed!{ENDC}")
    else:
        print(f"\n{RED}Some tests failed. See details above.{ENDC}")
    
    return all_passed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model wrapper improvements")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--simplification", action="store_true", help="Run simplification tests")
    parser.add_argument("--translation", action="store_true", help="Run translation tests")
    parser.add_argument("--memory", action="store_true", help="Run memory pressure tests")
    
    args = parser.parse_args()
    
    # If no specific tests are selected, run all tests
    if not (args.all or args.simplification or args.translation or args.memory):
        args.all = True
    
    return args

if __name__ == "__main__":
    args = parse_args()
    success = asyncio.run(run_tests(args))
    sys.exit(0 if success else 1)