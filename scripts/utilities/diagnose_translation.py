#!/usr/bin/env python3
"""
Diagnose translation issues by examining exactly what the model is doing
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def run_diagnosis():
    """
    Run a diagnosis of the translation pipeline to identify the issue
    """
    try:
        # Import necessary components
        print("Importing components...")
        from app.services.hardware.detector import HardwareDetector
        from app.services.models.loader import ModelLoader
        from app.services.models.manager import EnhancedModelManager
        from app.services.models.wrapper import TranslationModelWrapper, ModelInput
        
        # Initialize detector
        print("Initializing hardware detector...")
        hardware_detector = HardwareDetector()
        hardware_info = hardware_detector.detect_all()
        
        # Create model loader and manager
        print("Creating model manager...")
        model_loader = ModelLoader()
        model_manager = EnhancedModelManager(model_loader, hardware_info)
        
        # Load MBART model directly
        print("Loading MBART translation model...")
        model_info = await model_manager.load_model("mbart_translation")
        
        # Test text
        text = "Hello world, this is a test."
        source_language = "en"
        target_language = "es"
        
        print(f"\nTest translation: '{text}' from {source_language} to {target_language}")
        
        # 1. Test direct ModelInput to wrapper
        print("\n1. Testing direct wrapper implementation...")
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        wrapper = TranslationModelWrapper(
            model=model,
            tokenizer=tokenizer,
            config={"task": "mbart_translation", "device": "cpu"}
        )
        
        model_input = ModelInput(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=[]
        )
        
        # Process using direct wrapper methods for detailed debugging
        print("1.1 Preprocessing...")
        preprocessed = wrapper._preprocess(model_input)
        print(f"Preprocessed keys: {preprocessed.keys()}")
        
        print("1.2 Running inference...")
        raw_output = wrapper._run_inference(preprocessed)
        print(f"Raw output type: {type(raw_output)}")
        
        print("1.3 Postprocessing...")
        result = wrapper._postprocess(raw_output, model_input)
        print(f"Result: {result.result}")
        print(f"Metadata: {result.metadata}")
        
        # 2. Test through model_manager run_model
        print("\n2. Testing through model_manager.run_model...")
        input_data = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
            "parameters": {
                "mbart_source_lang": "en_XX",
                "mbart_target_lang": "es_XX"
            }
        }
        
        # Run model through manager
        print("Calling model_manager.run_model...")
        result = await model_manager.run_model(
            "mbart_translation",
            "process",
            input_data
        )
        
        print(f"Result from model_manager: {result}")
        
        # 3. Check if the model appears to be correctly initialized
        print("\n3. Checking model configuration...")
        if hasattr(model, "config"):
            print(f"Model config: {model.config}")
        
        if hasattr(tokenizer, "src_lang") and hasattr(tokenizer, "tgt_lang"):
            print(f"Tokenizer default src_lang: {tokenizer.src_lang}")
            print(f"Tokenizer default tgt_lang: {tokenizer.tgt_lang}")
        
        # 4. Test MBART specific translation
        print("\n4. Testing with MBART-specific parameters...")
        # Set tokenizer languages for MBART
        if hasattr(tokenizer, "src_lang") and hasattr(tokenizer, "tgt_lang"):
            # Save original values
            original_src = tokenizer.src_lang
            original_tgt = tokenizer.tgt_lang
            
            # Set to test values
            tokenizer.src_lang = "en_XX"
            tokenizer.tgt_lang = "es_XX"
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            print(f"Tokenized input: {inputs.keys()}")
            
            if hasattr(model, "generate") and callable(model.generate):
                print("Generating with MBART model...")
                # Direct generate call
                with open("generate_output.txt", "w") as f:
                    f.write(f"Input: {text}\n")
                    f.write(f"Tokenized: {inputs}\n")
                    
                    try:
                        forced_bos_token_id = None
                        if hasattr(tokenizer, "lang_code_to_id"):
                            forced_bos_token_id = tokenizer.lang_code_to_id.get("es_XX")
                            f.write(f"Forced BOS token ID: {forced_bos_token_id}\n")
                        
                        generation_kwargs = {
                            "max_length": 512,
                            "num_beams": 4,
                            "temperature": 1.0,
                            "no_repeat_ngram_size": 3,
                            "early_stopping": True,
                        }
                        
                        if forced_bos_token_id is not None:
                            generation_kwargs["forced_bos_token_id"] = forced_bos_token_id
                        
                        f.write(f"Generation kwargs: {generation_kwargs}\n")
                        
                        outputs = model.generate(**inputs, **generation_kwargs)
                        f.write(f"Generated outputs shape: {outputs.shape}\n")
                        
                        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        f.write(f"Decoded output: {decoded}\n")
                        
                        print(f"Generated translation: {decoded}")
                    except Exception as e:
                        f.write(f"Error during generation: {str(e)}\n")
                        print(f"Error generating: {str(e)}")
            
            # Restore original values
            tokenizer.src_lang = original_src
            tokenizer.tgt_lang = original_tgt
        
        print("\nDiagnosis complete. Check generate_output.txt for detailed output.")
        return True
    except Exception as e:
        logger.error(f"Error during diagnosis: {str(e)}", exc_info=True)
        print(f"Diagnosis failed: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(run_diagnosis())