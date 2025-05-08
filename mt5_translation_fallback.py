#!/usr/bin/env python3
"""
MT5 fallback for Spanish to English translation.

This script implements a fix for the Spanish to English translation issue
by automatically using MT5 instead of MBART for Spanish to English translations.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_model_registry_config():
    """Update the model registry to use MT5 for translation."""
    registry_path = "config/model_registry.json"
    if not os.path.exists(registry_path):
        logger.error(f"Registry file {registry_path} not found")
        return False
    
    # Read the existing registry
    import json
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Create backup
    backup_path = f"{registry_path}.bak"
    with open(backup_path, 'w') as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Created backup of registry at {backup_path}")
    
    # Update the translation model to use MT5 instead of MBART
    # Keep mbart_translation for other language pairs that might work
    if 'translation' in registry:
        original_model = registry['translation'].get('model_name', '')
        if 'mbart' in original_model.lower():
            logger.info(f"Replacing MBART model {original_model} with MT5 for translation")
            registry['translation']['model_name'] = "google/mt5-small"
            registry['translation']['tokenizer_name'] = "google/mt5-small"
            
            # Write the updated registry
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            logger.info("✅ Updated model registry to use MT5 for translation")
            return True
        else:
            logger.info(f"Translation model is already not using MBART: {original_model}")
            return True
    else:
        logger.error("Translation model not found in registry")
        return False

def add_spanish_to_english_detection_in_preprocess():
    """
    Modify the TranslationModelWrapper._preprocess method to detect 
    Spanish to English translations and always use MT5 mode.
    """
    file_path = 'app/services/models/wrapper.py'
    logger.info(f"Adding Spanish to English detection in {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = f"{file_path}.bak_es2en_fix"
    with open(backup_path, 'w') as f:
        f.write(content)
    logger.info(f"Created backup at {backup_path}")
    
    # Find the pattern - early part of _preprocess where we set is_spanish_to_english
    pattern = """        # Special handling for Spanish to English translations
        is_spanish_to_english = source_lang == "es" and target_lang == "en"
        if is_spanish_to_english:
            logger.info("⚠️ Special handling for Spanish->English translation")"""
    
    # Replacement with MT5 style forcing regardless of model
    replacement = """        # Special handling for Spanish to English translations
        is_spanish_to_english = source_lang == "es" and target_lang == "en"
        if is_spanish_to_english:
            logger.info("⚠️ Special handling for Spanish->English translation")
            
            # Force MT5 style prefixing for Spanish to English
            logger.info("⚠️ Forcing MT5-style processing for Spanish to English")
            # Force variable to help influence branching logic later
            force_mt5_style = True"""
    
    # Second pattern - MBART vs MT5 branching
    pattern2 = """        # Handle MBART vs MT5 models
        model_name = getattr(getattr(self.model, "config", None), "_name_or_path", "") if hasattr(self.model, "config") else ""
        if "mbart" in model_name.lower():"""
    
    # Replacement with Spanish to English detection
    replacement2 = """        # Handle MBART vs MT5 models
        model_name = getattr(getattr(self.model, "config", None), "_name_or_path", "") if hasattr(self.model, "config") else ""
        
        # Special case for Spanish to English - always use MT5 style regardless of model
        if is_spanish_to_english and 'force_mt5_style' in locals() and force_mt5_style:
            logger.info("⚠️ Using MT5 style prompting for Spanish->English translation")
            # Override MBART logic and use MT5 style for es->en
            # MT5 and other models (use text prefix format)
            try:
                # Attempt to use the enhanced prompt generator if available
                from app.services.models.translation_prompt_enhancer import TranslationPromptEnhancer
                parameters = {}
                if hasattr(input_data, 'parameters') and input_data.parameters:
                    parameters = input_data.parameters
                elif isinstance(input_data, dict) and 'parameters' in input_data:
                    parameters = input_data.get('parameters', {})
                
                domain = parameters.get("domain", "")
                formality = parameters.get("formality", "")
                context = input_data.context if hasattr(input_data, 'context') else None
                
                if parameters.get("enhance_prompts", True):
                    prompt_enhancer = TranslationPromptEnhancer()
                    
                    enhanced_texts = []
                    for text in texts:
                        enhanced_prompt = prompt_enhancer.enhance_mt5_prompt(
                            text, source_lang, target_lang, domain, formality, context, parameters
                        )
                        enhanced_texts.append(enhanced_prompt)
                    
                    prefixed_texts = enhanced_texts
                    logger.info(f"Enhanced MT5 prompts for Spanish->English: {prefixed_texts[0][:50]}...")
                else:
                    # Use standard prompt format if enhancement is disabled
                    prefixed_texts = [f"translate {source_lang} to {target_lang}: {text}" for text in texts]
            except ImportError:
                # Fall back to standard prompt format if enhancer is not available
                logger.warning("TranslationPromptEnhancer not available, using standard prompts")
                prefixed_texts = [f"translate {source_lang} to {target_lang}: {text}" for text in texts]
            
            # Tokenize inputs for non-MBART models
            if self.tokenizer:
                inputs = self.tokenizer(
                    prefixed_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=(self.config.get("max_length", 1024) if isinstance(self.config, dict) else getattr(self.config, "max_length", 1024))
                )
            else:
                inputs = {"texts": prefixed_texts}
        elif "mbart" in model_name.lower():"""
    
    # Replace the patterns
    if pattern in content and pattern2 in content:
        new_content = content.replace(pattern, replacement).replace(pattern2, replacement2)
        
        with open(file_path, 'w') as f:
            f.write(new_content)
            
        logger.info("✅ Added Spanish to English MT5 forcing in wrapper.py")
        return True
    else:
        logger.error("Could not find patterns to replace in wrapper.py")
        return False

def create_test_script():
    """Create a test script to verify the MT5 fallback for Spanish to English."""
    file_path = 'test_mt5_spanish_english_fallback.py'
    logger.info(f"Creating test script at {file_path}...")
    
    with open(file_path, 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Test for MT5-based Spanish to English translation fallback.

This script tests Spanish to English translation using the MT5 fallback
approach instead of MBART, which has issues on Apple Silicon hardware.
\"\"\"

import os
import sys
import logging
import torch
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mt5_spanish_to_english():
    \"\"\"Test Spanish to English translation using MT5 model.\"\"\"
    from app.services.models.wrapper import TranslationModelWrapper, ModelInput
    from app.services.models.loader import get_model_loader
    
    # Test text
    test_text = "Estoy muy feliz de conocerte hoy."
    logger.info(f"\\n=== Testing Spanish->English with MT5 fallback: '{test_text}' ===")
    
    # Get the model loader
    loader = get_model_loader()
    
    # Load translation model
    logger.info("Loading translation model...")
    model_info = loader.load_model("translation")
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    config = model_info["config"]
    
    # Check which model was loaded
    if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
        model_name = model.config._name_or_path
        logger.info(f"Loaded model: {model_name}")
        if "mbart" in model_name.lower():
            logger.warning("⚠️ MBART model loaded - should be using MT5 for Spanish->English!")
        elif "mt5" in model_name.lower():
            logger.info("✅ MT5 model loaded correctly")
        else:
            logger.info(f"Other model type loaded: {model_name}")
    
    # Create wrapper
    logger.info("Creating TranslationModelWrapper...")
    wrapper = TranslationModelWrapper(model, tokenizer, config)
    
    # Create model input for Spanish to English
    input_data = ModelInput(
        text=test_text,
        source_language="es",
        target_language="en"
    )
    
    # Process translation
    logger.info("Running translation...")
    result = asyncio.run(wrapper.process(input_data))
    
    # Check result
    logger.info(f"Translation result: '{result.result}'")
    
    # Check if translation is available and not an error
    success = (
        result.result and
        result.result != "Translation unavailable" and
        "unavailable" not in result.result.lower() and
        "error" not in result.result.lower() and
        len(result.result) > 5
    )
    
    if success:
        logger.info("✅ Translation successful!")
        return True, result.result
    else:
        logger.error("❌ Translation failed or returned error message")
        return False, result.result

def main():
    \"\"\"Run the MT5 Spanish to English translation test.\"\"\"
    logger.info("=== Testing MT5 Fallback for Spanish to English Translation ===")
    
    # Test with MT5 fallback
    success, result = test_mt5_spanish_to_english()
    
    if success:
        logger.info("\\n✅✅✅ MT5 FALLBACK FOR SPANISH TO ENGLISH WORKING SUCCESSFULLY ✅✅✅")
        logger.info(f"Translation result: '{result}'")
        return 0
    else:
        logger.error("\\n❌❌❌ MT5 FALLBACK FOR SPANISH TO ENGLISH FAILED ❌❌❌")
        logger.error(f"Error result: '{result}'")
        return 1

if __name__ == "__main__":
    sys.exit(main())
""")
    
    logger.info("✅ Created test script successfully")
    return True

def main():
    """Apply MT5 fallback for Spanish to English translation."""
    logger.info("Starting MT5 fallback implementation for Spanish to English translation...")
    
    # Update the model registry to use MT5 instead of MBART for translation
    registry_updated = update_model_registry_config()
    if not registry_updated:
        logger.error("Failed to update model registry")
        return 1
    
    # Update the TranslationModelWrapper to use MT5 style for Spanish to English
    wrapper_updated = add_spanish_to_english_detection_in_preprocess()
    if not wrapper_updated:
        logger.error("Failed to update wrapper")
        return 1
    
    # Create test script
    test_script_created = create_test_script()
    if not test_script_created:
        logger.error("Failed to create test script")
        return 1
    
    # Success message
    logger.info("\n✅✅✅ MT5 FALLBACK FOR SPANISH TO ENGLISH IMPLEMENTED SUCCESSFULLY ✅✅✅")
    logger.info("Run the test with: python test_mt5_spanish_english_fallback.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())