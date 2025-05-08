#!/usr/bin/env python3
"""
Test script for CasaLingua model manager fixes.
This script tests the model loading and pipeline functionality
to verify that the fixes have resolved the issues.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the application to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

async def test_model_manager_compatibility():
    """Test the compatibility between ModelManager and EnhancedModelManager"""
    logger.info("Testing ModelManager compatibility...")
    
    try:
        # Import the ModelManager
        from app.services.models.model_manager import ModelManager
        
        # Create a model manager instance
        manager = ModelManager()
        
        # Initialize the manager
        await manager.initialize()
        
        # Check that initialization succeeded
        assert manager.enhanced_manager is not None, "EnhancedModelManager was not initialized"
        
        # List loaded models (should be empty at this point)
        loaded_models = await manager.list_loaded_models()
        logger.info(f"Initially loaded models: {loaded_models}")
        
        logger.info("ModelManager compatibility test passed!")
        return True
    except Exception as e:
        logger.error(f"ModelManager compatibility test failed: {e}", exc_info=True)
        return False

async def test_model_loading():
    """Test loading models with the fixed ModelManager"""
    logger.info("Testing model loading...")
    
    try:
        # Import the ModelManager
        from app.services.models.model_manager import ModelManager
        
        # Create a model manager instance
        manager = ModelManager()
        
        # Initialize the manager
        await manager.initialize()
        
        # Load a simple model - language detection model is usually the smallest
        model_type = "language_detection"
        logger.info(f"Loading model: {model_type}")
        
        model_info = await manager.load_model(model_type)
        
        # Check if model was loaded
        assert model_info is not None, f"Failed to load {model_type} model"
        assert "model" in model_info, f"Model not found in model_info for {model_type}"
        
        # List loaded models (should include our model)
        loaded_models = await manager.list_loaded_models()
        logger.info(f"Loaded models after test: {loaded_models}")
        
        assert model_type in loaded_models, f"{model_type} not in loaded_models list"
        
        logger.info("Model loading test passed!")
        return True
    except Exception as e:
        logger.error(f"Model loading test failed: {e}", exc_info=True)
        return False

async def test_pipeline_operation():
    """Test the pipeline operation with language detection"""
    logger.info("Testing pipeline operation with language detection...")
    
    try:
        # Import the UnifiedProcessor
        from app.core.pipeline.processor import UnifiedProcessor
        
        # Import the ModelManager
        from app.services.models.model_manager import ModelManager
        
        # Create a model manager instance
        model_manager = ModelManager()
        
        # Initialize the model manager
        await model_manager.initialize()
        
        # Create a processor instance
        processor = UnifiedProcessor(model_manager=model_manager)
        
        # Initialize the processor
        await processor.initialize()
        
        # Test language detection
        test_text = "This is a test of the language detection functionality."
        
        detection_result = await processor.detect_language(test_text)
        
        # Check if language detection worked
        assert detection_result is not None, "Language detection returned None"
        assert "detected_language" in detection_result, "detected_language not in result"
        assert detection_result.get("detected_language") == "en", f"Expected 'en', got {detection_result.get('detected_language')}"
        
        logger.info(f"Detection result: {detection_result}")
        logger.info("Pipeline operation test passed!")
        return True
    except Exception as e:
        logger.error(f"Pipeline operation test failed: {e}", exc_info=True)
        return False

async def test_simplifier():
    """Test text simplification with the fixed code"""
    logger.info("Testing text simplification...")
    
    try:
        # Import SimplificationPipeline
        from app.core.pipeline.simplifier import SimplificationPipeline
        
        # Import the ModelManager
        from app.services.models.model_manager import ModelManager
        
        # Create a model manager instance
        model_manager = ModelManager()
        
        # Initialize the model manager
        await model_manager.initialize()
        
        # Create a simplifier instance
        simplifier = SimplificationPipeline(model_manager)
        
        # Initialize the simplifier
        await simplifier.initialize()
        
        # Test text simplification with a complex text
        test_text = "The applicant must furnish all necessary documentation in accordance with the aforementioned requirements prior to the application deadline."
        
        # Try different simplification levels
        for level in [1, 3, 5]:
            simplification_result = await simplifier.simplify(
                text=test_text,
                language="en",
                level=level
            )
            
            # Check if simplification worked
            assert simplification_result is not None, "Simplification returned None"
            assert "simplified_text" in simplification_result, "simplified_text not in result"
            
            simplified_text = simplification_result.get("simplified_text")
            assert simplified_text and len(simplified_text) > 0, "Simplified text is empty"
            
            logger.info(f"Level {level} simplification: {simplified_text}")
        
        logger.info("Simplifier test passed!")
        return True
    except Exception as e:
        logger.error(f"Simplifier test failed: {e}", exc_info=True)
        return False

async def run_all_tests():
    """Run all tests and report results"""
    results = {}
    
    # Test 1: Model Manager Compatibility
    results["model_manager_compatibility"] = await test_model_manager_compatibility()
    
    # Test 2: Model Loading
    results["model_loading"] = await test_model_loading()
    
    # Test 3: Pipeline Operation
    results["pipeline_operation"] = await test_pipeline_operation()
    
    # Test 4: Simplifier
    results["simplifier"] = await test_simplifier()
    
    # Print summary
    print("\n--- TEST RESULTS SUMMARY ---")
    for test, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test}: {status}")
    
    # Overall result
    overall = all(results.values())
    print(f"\nOVERALL RESULT: {'PASSED' if overall else 'FAILED'}")
    return overall

if __name__ == "__main__":
    # Run the tests
    try:
        overall_result = asyncio.run(run_all_tests())
        sys.exit(0 if overall_result else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnhandled exception: {e}")
        sys.exit(1)