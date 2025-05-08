"""
Direct test of veracity checking with a custom wrapper that directly inherits from BaseModelWrapper.
"""

import asyncio
import sys
import os
from typing import Dict, Any, Optional, Union, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Create simple mock model and tokenizer
class MockModel:
    def __init__(self):
        self.device = "cpu"
    
    def generate(self, **kwargs):
        # Just return a simple token sequence
        class MockTensor:
            def __getitem__(self, idx):
                return [101, 2213, 2495, 102]
        return MockTensor()

class MockTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
    
    def batch_decode(self, token_ids, **kwargs):
        return ["Hola mundo"]

async def main():
    from app.audit.veracity import VeracityAuditor
    from app.services.models.wrapper_base import BaseModelWrapper, ModelInput, ModelOutput
    
    print("Creating a custom wrapper class that inherits from BaseModelWrapper...")
    
    # Implement a simple wrapper that directly inherits from BaseModelWrapper
    class MyCustomWrapper(BaseModelWrapper):
        def _preprocess(self, input_data):
            return {
                "texts": input_data.text if isinstance(input_data.text, list) else [input_data.text]
            }
        
        def _run_inference(self, preprocessed):
            return "Hola mundo"
        
        def _postprocess(self, model_output, input_data):
            return ModelOutput(
                result=model_output,
                metadata={
                    "source_language": input_data.source_language,
                    "target_language": input_data.target_language
                }
            )
    
    # Create model, tokenizer, and config
    model = MockModel()
    tokenizer = MockTokenizer()
    config = {"device": "cpu"}
    
    # Create the wrapper
    wrapper = MyCustomWrapper(model, tokenizer, config)
    
    print("Creating and initializing VeracityAuditor...")
    auditor = VeracityAuditor()
    await auditor.initialize()
    
    # Set the auditor on the wrapper
    wrapper.veracity_checker = auditor
    
    # Test direct veracity checking
    print("\nTesting direct veracity checking...")
    verification = await auditor.verify_translation(
        "Hello world",
        "Hola mundo",
        "en",
        "es"
    )
    print(f"Direct verification result: {verification.get('verified', False)}")
    
    # Test wrapper with veracity checking
    print("\nProcessing input with wrapper that has veracity checker...")
    input_data = ModelInput(
        text="Hello world",
        source_language="en",
        target_language="es"
    )
    
    # Process the input
    try:
        result = wrapper.process(input_data)
        print("Processing successful")
        
        print("\nResult:")
        print(f"Result value: {result.get('result')}")
        
        # Check for veracity data
        if "metadata" in result and "veracity" in result["metadata"]:
            veracity = result["metadata"]["veracity"]
            print(f"Veracity score: {veracity.get('score', 0):.2f}")
            print(f"Confidence: {veracity.get('confidence', 0):.2f}")
        else:
            print("No veracity data in result")
    except Exception as e:
        print(f"Error processing input: {e}")

if __name__ == "__main__":
    asyncio.run(main())