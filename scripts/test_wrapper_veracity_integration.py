"""
Test script for integration between ModelWrapper and VeracityAuditor.

This script simulates how the server would use the wrapper and veracity checking.
"""

import asyncio
import sys
import os
from typing import Dict, Any, Optional, Union

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

class MockModel:
    """Mock model for testing purposes."""
    
    def __init__(self, name="mock_model"):
        self.name = name
        self.device = "cpu"
    
    def to(self, device):
        self.device = device
        return self
    
    def generate(self, **kwargs):
        # For translation model mock
        # Return a tensor-like object with a shape
        class MockTensor:
            def __init__(self, values):
                self.values = values
                self.shape = (1, len(values))
            
            def __getitem__(self, idx):
                return self.values[idx]
        
        # Mock token IDs for "Hola mundo"
        return MockTensor([101, 2213, 2495, 102])

class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    
    def __call__(self, texts, **kwargs):
        return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
    
    def batch_decode(self, outputs, **kwargs):
        # For this test, just return fixed translations
        return ["Hola mundo"]

async def main():
    print("Initializing test environment...")
    
    # Import the necessary components
    from app.audit.veracity import VeracityAuditor
    from app.services.models.wrapper import get_wrapper_for_model
    from app.services.models.wrapper_base import ModelInput
    
    # Create a mock model and tokenizer
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Create a veracity auditor
    print("Creating VeracityAuditor...")
    auditor = VeracityAuditor()
    await auditor.initialize()
    
    # First, test the auditor directly
    print("\nTesting VeracityAuditor directly...")
    result = await auditor.verify_translation(
        "Hello world",
        "Hola mundo",
        "en",
        "es"
    )
    print(f"Direct verification result: {result.get('verified')} (score: {result.get('score'):.2f})")
    
    # Now create a model wrapper with the veracity auditor
    print("\nCreating model wrapper with veracity auditor...")
    
    # Create config for the wrapper
    config = {
        "device": "cpu",
        "max_length": 512,
        "generation_kwargs": {
            "max_length": 50,
            "min_length": 5,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95
        }
    }
    
    # Create the wrapper
    wrapper = get_wrapper_for_model(
        model_type="translation",
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Set the veracity checker manually
    print("Setting veracity checker on wrapper...")
    wrapper.veracity_checker = auditor
    
    # Verify the veracity checker is set
    print(f"Wrapper has veracity checker: {hasattr(wrapper, 'veracity_checker')}")
    
    # For debugging - monkey patch the _check_veracity_sync method to add debug printing
    original_check_veracity = wrapper._check_veracity_sync
    
    def debug_check_veracity(self, result, input_data):
        print(f"DEBUG: _check_veracity_sync called with result type: {type(result)}")
        print(f"DEBUG: result value: {result}")
        try:
            metrics = original_check_veracity(result, input_data)
            print(f"DEBUG: Veracity check successful: {metrics}")
            return metrics
        except Exception as e:
            print(f"DEBUG: Veracity check failed with error: {e}")
            raise
    
    # Apply the patched method
    from types import MethodType
    wrapper._check_veracity_sync = MethodType(debug_check_veracity, wrapper)
    
    # Create input for the wrapper
    input_data = ModelInput(
        text="Hello world",
        source_language="en",
        target_language="es"
    )
    
    # Process the input
    print("\nProcessing input through wrapper...")
    try:
        # Use sync version for TranslationModelWrapper
        result = wrapper.process(input_data)
        print("Used sync processing")
    except Exception as e:
        print(f"Processing failed: {e}")
        result = {"error": str(e)}
    
    # Print the result
    print("\nProcessing result:")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Result: {result.get('result')}")
        
        # Check for veracity data
        if "metadata" in result and "veracity" in result["metadata"]:
            veracity = result["metadata"]["veracity"]
            print(f"Veracity score: {veracity.get('score', 0):.2f}")
            print(f"Confidence: {veracity.get('confidence', 0):.2f}")
            if "checks_passed" in veracity:
                print(f"Checks passed: {', '.join(veracity['checks_passed'])}")
            if "checks_failed" in veracity:
                print(f"Checks failed: {', '.join(veracity['checks_failed'])}")
        else:
            print("No veracity data found in result")

if __name__ == "__main__":
    asyncio.run(main())