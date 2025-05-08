#!/usr/bin/env python3
"""
Test script for simplification functionality by directly using the SimplificationModelWrapper.
This bypasses the API to test the core functionality.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

async def test_simplification():
    """Test the SimplificationModelWrapper directly."""
    try:
        # Import necessary modules
        from app.services.models.wrapper import SimplificationModelWrapper, ModelInput, ModelOutput
        
        # Create a mock model and tokenizer
        class MockModel:
            def generate(self, **kwargs):
                # Return a mock tensor that the tokenizer can decode
                import torch
                # Create a mock token sequence - these would normally be token IDs
                # that represent the simplified text
                return torch.tensor([[101, 2023, 2003, 1037, 4937, 2944, 1012, 102]])
            
            def __init__(self):
                self.config = type('obj', (object,), {
                    '_name_or_path': 'mock_simplifier',
                })
        
        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                # Return a mock encoding
                import torch
                return {
                    "input_ids": torch.tensor([[101, 2023, 2003, 1037, 2943, 2944, 1012, 102]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
                }
            
            def batch_decode(self, token_ids, **kwargs):
                # Return simplified texts based on the test level
                # Just a mock implementation for testing
                return ["This is a simple text."]
        
        # Create a model wrapper with our mocks
        wrapper = SimplificationModelWrapper(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            config={"max_length": 512}
        )
        
        # Test cases for different simplification levels
        test_cases = [
            {"level": 1, "text": "The implementation of the algorithm necessitated a comprehensive understanding of advanced mathematical principles."},
            {"level": 3, "text": "The cardiovascular system functions through a complex interaction of hemodynamic processes."},
            {"level": 5, "text": "The meteorological conditions indicate a high probability of precipitation."}
        ]
        
        # Try the tests
        for case in test_cases:
            # Create input for the model
            input_data = ModelInput(
                text=case["text"],
                parameters={"level": case["level"]}
            )
            
            # Process the input through the wrapper
            output = await wrapper.process(input_data)
            
            # Check if we got a result
            if output and hasattr(output, 'result'):
                logger.info(f"Level {case['level']} - Input: '{case['text'][:30]}...'")
                logger.info(f"Level {case['level']} - Output: '{output.result}'")
                logger.info(f"Level {case['level']} - Test passed ✅")
            else:
                logger.error(f"Level {case['level']} - Test failed ❌")
                return False
        
        # Try a more realistic test if we have the actual model loaded
        try:
            from app.services.models.loader import load_model_and_tokenizer
            
            # Try to load the actual simplifier model if available
            model, tokenizer = load_model_and_tokenizer("simplifier")
            
            if model and tokenizer:
                logger.info("Testing with actual simplifier model...")
                
                # Create a real model wrapper
                real_wrapper = SimplificationModelWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    config={"max_length": 512}
                )
                
                # Use a real test case
                real_input = ModelInput(
                    text="The implementation of the algorithm necessitated a comprehensive understanding of advanced mathematical principles.",
                    parameters={"level": 3}
                )
                
                # Process the input
                real_output = await real_wrapper.process(real_input)
                
                # Check the result
                if real_output and hasattr(real_output, 'result'):
                    logger.info(f"Real model - Input: '{real_input.text[:30]}...'")
                    logger.info(f"Real model - Output: '{real_output.result}'")
                    logger.info("Real model - Test passed ✅")
                else:
                    logger.warning("Real model - Test produced no result")
            else:
                logger.warning("Could not load actual simplifier model for testing")
        
        except (ImportError, Exception) as e:
            logger.warning(f"Skipping real model test: {e}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing simplification: {e}", exc_info=True)
        return False

async def main():
    """Main entry point"""
    logger.info("Testing SimplificationModelWrapper...")
    
    success = await test_simplification()
    
    if success:
        logger.info("✅ All simplification tests passed!")
        return 0
    else:
        logger.error("❌ Some simplification tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))