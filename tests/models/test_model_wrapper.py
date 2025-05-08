"""
Test script to diagnose model wrapper issues
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import the ModelManager
    from app.services.models.manager import EnhancedModelManager
    from app.services.models.wrapper_base import BaseModelWrapper, ModelInput

    # Create a simplified test
    def test_model_wrapper():
        logger.info("Testing model wrapper classes...")

        # Create a dummy input
        model_input = ModelInput(
            text="Hello, how are you?",
            source_language="en",
            target_language="es",
            parameters={"test": True}
        )

        logger.info(f"Created model input: {model_input}")

        # Try to import explicitly to ensure it's loaded
        from app.services.models.wrapper import TranslationModelWrapper, get_wrapper_for_model, wrapper_map

        # Print out the wrapper map
        logger.info(f"Wrapper map: {wrapper_map}")

        # Create a dummy wrapper for testing
        class TestWrapper(BaseModelWrapper):
            def _preprocess(self, input_data):
                logger.info(f"Preprocessing data: {input_data}")
                return {"text": input_data.text}
                
            def _run_inference(self, preprocessed):
                logger.info(f"Running inference: {preprocessed}")
                return {"output": "This is a test"}
                
            def _postprocess(self, model_output, input_data):
                logger.info(f"Postprocessing: {model_output}")
                from app.services.models.wrapper_base import ModelOutput
                return ModelOutput(
                    result="Translated text",
                    metadata={"confidence": 0.9}
                )

        test_wrapper = TestWrapper(None, None, {"test": True})
        logger.info(f"Created test wrapper: {test_wrapper}")

        # Test the processing
        try:
            output = test_wrapper.process(model_input)
            logger.info(f"Output: {output}")
        except Exception as e:
            logger.error(f"Error in test wrapper: {e}", exc_info=True)

        # Test all wrapper implementations in wrapper_map
        for model_type, wrapper_class in wrapper_map.items():
            if wrapper_class is not None:
                logger.info(f"Testing wrapper for {model_type}...")
                try:
                    test_instance = wrapper_class(None, None, {"test": True})
                    logger.info(f"Created test instance for {model_type}")
                    
                    # Test the preprocessing
                    try:
                        logger.info(f"Testing _preprocess for {model_type}")
                        preprocessed = test_instance._preprocess(model_input)
                        logger.info(f"Preprocessed: {preprocessed}")
                    except Exception as e:
                        logger.error(f"Error in _preprocess for {model_type}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error creating test instance for {model_type}: {e}", exc_info=True)

    # Run the test
    test_model_wrapper()

except Exception as e:
    logger.error(f"Error during test: {e}", exc_info=True)