#!/usr/bin/env python3
"""
Direct test of the TranslationModelWrapper class
"""

import os
import sys
import logging
import asyncio
from app.services.models.wrapper import TranslationModelWrapper, ModelInput
from app.services.models.loader import ModelLoader
from app.services.hardware.detector import HardwareDetector
from app.services.models.manager import EnhancedModelManager

async def test_translation_wrapper():
    """Test the TranslationModelWrapper directly"""
    print("Starting direct test of TranslationModelWrapper...")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize hardware detector
    hardware_detector = HardwareDetector()
    hardware_info = hardware_detector.detect_all()
    print(f"Hardware info detected")
    
    # Create model loader
    model_loader = ModelLoader()
    
    # Create model manager
    model_manager = EnhancedModelManager(model_loader, hardware_info)
    
    # Load translation model
    print("Loading translation model...")
    try:
        model_info = await model_manager.load_model("mbart_translation")
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Create wrapper
        print("Creating TranslationModelWrapper...")
        wrapper = TranslationModelWrapper(
            model=model,
            tokenizer=tokenizer,
            config={"task": "translation", "device": "cpu"}
        )
        
        # Test text
        text = "Hello world, this is a test."
        source_language = "en"
        target_language = "es"
        
        # Create input
        model_input = ModelInput(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=[]
        )
        
        # Process input (directly calling each step for debugging)
        print(f"Testing text: '{text}' from {source_language} to {target_language}")
        
        print("1. Preprocessing input...")
        preprocessed = wrapper._preprocess(model_input)
        print(f"Preprocessed: {preprocessed.keys()}")
        
        print("2. Running inference...")
        raw_output = wrapper._run_inference(preprocessed)
        print(f"Raw output type: {type(raw_output)}")
        
        print("3. Postprocessing output...")
        result = wrapper._postprocess(raw_output, model_input)
        print(f"Result: {result.result}")
        print(f"Metadata: {result.metadata}")
        
        # Check if translation was successful
        if text == result.result:
            print("⚠️ WARNING: Input and translated text are identical - no translation occurred")
        else:
            print(f"✅ Successful translation: '{result.result}'")
        
        return True
    except Exception as e:
        print(f"Error testing translation wrapper: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_translation_wrapper())