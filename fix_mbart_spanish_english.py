#!/usr/bin/env python3
"""
Fix for MBART Spanish to English translation issues on Apple MPS devices.

This script fixes the issue with Spanish to English translations using MBART models on
Apple Silicon devices. The main problems addressed are:

1. Ensuring MBART models are always forced to CPU on MPS devices
2. Properly handling the forced_bos_token_id parameter for Spanish to English translations
3. Ensuring the tokenizer and model are on the same device
"""

import os
import sys
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_file_paths():
    """Find the necessary file paths for the fix."""
    base_dir = Path(".")
    
    # Find loader.py
    loader_paths = list(base_dir.glob("**/loader.py"))
    loader_paths = [p for p in loader_paths if "app/services/models" in str(p)]
    
    # Find wrapper.py
    wrapper_paths = list(base_dir.glob("**/wrapper.py"))
    wrapper_paths = [p for p in wrapper_paths if "app/services/models" in str(p)]
    
    if not loader_paths:
        logger.error("Could not find loader.py in app/services/models/")
        return None, None
        
    if not wrapper_paths:
        logger.error("Could not find wrapper.py in app/services/models/")
        return None, None
    
    return loader_paths[0], wrapper_paths[0]

def create_backup(file_path):
    """Create a backup of the file before modifying it."""
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak_mbart_fix")
    
    # Check if backup already exists
    if backup_path.exists():
        logger.info(f"Backup already exists at {backup_path}")
        return
    
    # Create backup
    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    
    logger.info(f"Created backup at {backup_path}")

def fix_loader_determine_device(loader_path):
    """
    Enhance the _determine_device method in ModelLoader to improve MBART detection
    and force CPU usage on MPS devices.
    """
    if not loader_path.exists():
        logger.error(f"Loader file not found: {loader_path}")
        return False
    
    with open(loader_path, 'r') as f:
        content = f.read()
    
    # Find the _determine_device method
    method_pattern = r'def _determine_device\(self, model_type: str = None, model_size: str = None\) -> str:.*?(?=\n    def|\n    async|\Z)'
    method_match = re.search(method_pattern, content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find _determine_device method in loader.py")
        return False
    
    method_code = method_match.group(0)
    original_method = method_code
    
    # Check if the method already has the enhanced MBART check
    if "# Enhanced MBART detection" in method_code:
        logger.info("MBART detection already enhanced in _determine_device method")
        return True
    
    # Find the MBART-specific logic section
    mbart_section_pattern = r'# Special case for MBART models.*?if "mps" in self.available_devices:.*?return "cpu"'
    mbart_section_match = re.search(mbart_section_pattern, method_code, re.DOTALL)
    
    if not mbart_section_match:
        logger.error("Could not find MBART special case section in _determine_device method")
        return False
    
    mbart_section = mbart_section_match.group(0)
    
    # Enhanced MBART detection logic with better model detection
    enhanced_mbart_section = """        # Enhanced MBART detection - force CPU for all MBART variants when using MPS
        if "mps" in self.available_devices:  # Only apply on MPS devices
            # First check model_type-based detection
            is_mbart_model = False
            
            # Check the model type name for MBART indicators
            if model_type and any(mbart_marker in model_type.lower() for mbart_marker in [
                "mbart", "translation", "multilingual"
            ]):
                # Look up the actual model name if possible
                if model_type in self.registry:
                    model_name = self.registry[model_type].model_name.lower()
                    # More thorough check of model name
                    if any(mbart_id in model_name for mbart_id in [
                        "mbart", "facebook/mbart", "multilingual-translation", "nllb"
                    ]):
                        is_mbart_model = True
                        logger.warning(f"⚠️ Forcing CPU device for MBART model {model_type} due to known MPS compatibility issues")
                        return "cpu"
                # Special case for translation model type with no registry entry
                elif model_type == "translation" or model_type == "mbart_translation":
                    logger.warning(f"⚠️ Forcing CPU device for translation model {model_type} due to potential MBART compatibility issues with MPS")
                    return "cpu"
            
            # Special case for Spanish to English translation models
            # These are known to have issues with MPS even if not explicitly MBART
            if model_type and model_type.lower() in ["translation", "mbart_translation", "mt5_translation"]:
                logger.warning(f"⚠️ Forcing CPU device for {model_type} model due to potential Spanish-English translation compatibility issues with MPS")
                return "cpu"
"""
    
    # Replace the original MBART section with the enhanced one
    updated_method = method_code.replace(mbart_section, enhanced_mbart_section)
    
    # Replace the original method in the file content
    updated_content = content.replace(original_method, updated_method)
    
    # Write the updated content back to the file
    with open(loader_path, 'w') as f:
        f.write(updated_content)
    
    logger.info("Successfully enhanced MBART detection in ModelLoader._determine_device")
    return True

def fix_wrapper_init(wrapper_path):
    """
    Enhance the BaseModelWrapper.__init__ method to improve MBART detection
    and force CPU usage on MPS devices.
    """
    if not wrapper_path.exists():
        logger.error(f"Wrapper file not found: {wrapper_path}")
        return False
    
    with open(wrapper_path, 'r') as f:
        content = f.read()
    
    # Find the __init__ method in BaseModelWrapper
    init_pattern = r'def __init__\(self, model=None, tokenizer=None, config=None\):.*?(?=\n    async def|\n    @abstractmethod|\n    def _|\Z)'
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if not init_match:
        logger.error("Could not find __init__ method in BaseModelWrapper")
        return False
    
    init_code = init_match.group(0)
    original_init = init_code
    
    # Check if the method already has the enhanced MBART check
    if "# Enhanced MBART detection" in init_code:
        logger.info("MBART detection already enhanced in BaseModelWrapper.__init__ method")
        return True
    
    # Find the MPS-specific logic section
    mps_section_pattern = r'# Special handling for MPS device.*?self\.config\["device"\] = "cpu"'
    mps_section_match = re.search(mps_section_pattern, init_code, re.DOTALL)
    
    if not mps_section_match:
        logger.error("Could not find MPS special handling section in BaseModelWrapper.__init__ method")
        return False
    
    mps_section = mps_section_match.group(0)
    
    # Enhanced MPS handling with better MBART detection
    enhanced_mps_section = """        # Enhanced MBART detection - Special handling for MPS device due to stability issues
        if device == "mps":
            # 1. Check if this is an MBART model using the model configuration
            is_mbart_model = False
            model_name = ""
            
            # Check model config attribute for MBART identifiers
            if hasattr(model, "config"):
                if hasattr(model.config, "_name_or_path"):
                    model_name = model.config._name_or_path.lower()
                elif hasattr(model.config, "name_or_path"):
                    model_name = model.config.name_or_path.lower()
                elif hasattr(model.config, "model_type"):
                    model_name = model.config.model_type.lower()
                    
                # Check for MBART indicators in the name
                if any(mbart_id in model_name for mbart_id in ["mbart", "facebook/mbart", "nllb"]):
                    is_mbart_model = True
            
            # 2. Check the config dictionary for MBART indicators
            if not is_mbart_model and config:
                model_type = config.get("model_type", "").lower()
                task = config.get("task", "").lower()
                
                if "mbart" in model_type or "translation" in task or "translation" in model_type:
                    is_mbart_model = True
            
            # 3. Force CPU for any identified MBART models
            if is_mbart_model:
                logger.warning(f"⚠️ Forcing CPU device for MBART model due to known MPS compatibility issues")
                device = "cpu"
                # Update config to reflect the device change
                self.config["device"] = "cpu"
            
            # 4. Special handling for any translation model, even if not explicitly MBART
            elif config and "task" in config and config["task"] == "translation":
                logger.warning(f"⚠️ Forcing CPU device for translation model due to potential MBART compatibility issues with MPS")
                device = "cpu"
                self.config["device"] = "cpu"
"""
    
    # Replace the original MPS section with the enhanced one
    updated_init = init_code.replace(mps_section, enhanced_mps_section)
    
    # Replace the original method in the file content
    updated_content = content.replace(original_init, updated_init)
    
    # Write the updated content back to the file
    with open(wrapper_path, 'w') as f:
        f.write(updated_content)
    
    logger.info("Successfully enhanced MBART detection in BaseModelWrapper.__init__")
    return True

def fix_wrapper_run_inference(wrapper_path):
    """
    Enhance the TranslationModelWrapper._run_inference method to properly handle
    forced_bos_token_id for Spanish to English translations.
    """
    if not wrapper_path.exists():
        logger.error(f"Wrapper file not found: {wrapper_path}")
        return False
    
    with open(wrapper_path, 'r') as f:
        content = f.read()
    
    # Find the _run_inference method in TranslationModelWrapper
    method_pattern = r'def _run_inference\(self, preprocessed: Dict\[str, Any\]\) -> Any:.*?(?=\n    def _postprocess|\n    async def|\Z)'
    method_match = re.search(method_pattern, content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find _run_inference method in TranslationModelWrapper")
        return False
    
    method_code = method_match.group(0)
    original_method = method_code
    
    # Check if the method already has the enhanced Spanish-English handling
    if "# Enhanced Spanish to English handling" in method_code:
        logger.info("Spanish to English handling already enhanced in _run_inference method")
        return True
    
    # Find the section where MBART forced_bos_token_id is handled
    bos_token_section_pattern = r'# MBART model forced_bos_token_id handling.*?if "forced_bos_token_id" in gen_kwargs:.*?logger\.info\(f"Using forced_bos_token_id=.*?\)'
    bos_token_section_match = re.search(bos_token_section_pattern, method_code, re.DOTALL)
    
    if not bos_token_section_match:
        logger.error("Could not find forced_bos_token_id handling section in _run_inference method")
        return False
    
    bos_token_section = bos_token_section_match.group(0)
    
    # Enhanced Spanish to English handling with better token ID management
    enhanced_bos_token_section = """        # Enhanced Spanish to English handling - Critical for MBART model forced_bos_token_id handling
        if is_mbart:
            # Always ensure forced_bos_token_id is explicitly set for MBART
            if "forced_bos_token_id" not in gen_kwargs and "forced_bos_token_id" not in inputs:
                # Special handling for Spanish->English to improve quality
                if is_spanish_to_english:
                    # Make sure forced_bos_token_id is set to English (2) for MBART
                    gen_kwargs["forced_bos_token_id"] = 2  # Hardcoded English token ID for MBART
                    logger.info(f"Set forced_bos_token_id=2 (English) for Spanish->English translation in MBART model")
                elif hasattr(self.tokenizer, "lang_code_to_id"):
                    # For other language pairs, look up the token ID from the tokenizer
                    try:
                        gen_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[target_lang_code]
                        logger.info(f"Set forced_bos_token_id={gen_kwargs['forced_bos_token_id']} for {target_lang_code}")
                    except (KeyError, AttributeError) as e:
                        # Fallback to hardcoded values with expanded language mapping
                        lang_code_mapping = {
                            "en_XX": 2,   # English token ID
                            "es_XX": 8,   # Spanish token ID
                            "fr_XX": 6,   # French token ID
                            "de_XX": 4,   # German token ID
                            "it_XX": 7,   # Italian token ID
                            "pt_XX": 15,  # Portuguese token ID
                            "zh_CN": 10,  # Chinese (Simplified) token ID
                            "zh_TW": 11,  # Chinese (Traditional) token ID
                            # Add more common languages if needed
                        }
                        gen_kwargs["forced_bos_token_id"] = lang_code_mapping.get(target_lang_code, 2)  # Default to English if unknown
                        logger.info(f"Set forced_bos_token_id={gen_kwargs['forced_bos_token_id']} for {target_lang_code} using mapping")
            
            # Handle cases where forced_bos_token_id is in inputs
            elif "forced_bos_token_id" in inputs:
                # Make sure to use the right value
                if is_spanish_to_english:
                    # For Spanish->English, always override with explicit English token ID
                    gen_kwargs["forced_bos_token_id"] = 2
                    logger.info(f"Overrode forced_bos_token_id from inputs with 2 (English) for Spanish->English")
                else:
                    # Otherwise, use the value from inputs
                    gen_kwargs["forced_bos_token_id"] = inputs["forced_bos_token_id"]
                    logger.info(f"Using forced_bos_token_id={gen_kwargs['forced_bos_token_id']} from inputs")
        
        # Log the forced_bos_token_id that will be used
        if "forced_bos_token_id" in gen_kwargs:
            logger.info(f"Using forced_bos_token_id={gen_kwargs['forced_bos_token_id']} for generation")
"""
    
    # Replace the original section with the enhanced one
    updated_method = method_code.replace(bos_token_section, enhanced_bos_token_section)
    
    # Replace the original method in the file content
    updated_content = content.replace(original_method, updated_method)
    
    # Write the updated content back to the file
    with open(wrapper_path, 'w') as f:
        f.write(updated_content)
    
    logger.info("Successfully enhanced Spanish to English handling in TranslationModelWrapper._run_inference")
    return True

def main():
    logger.info("Starting MBART Spanish to English translation fix...")
    
    # Find necessary file paths
    loader_path, wrapper_path = find_file_paths()
    if not loader_path or not wrapper_path:
        logger.error("Could not find necessary files for the fix.")
        return 1
    
    logger.info(f"Found loader.py at: {loader_path}")
    logger.info(f"Found wrapper.py at: {wrapper_path}")
    
    # Create backups
    create_backup(loader_path)
    create_backup(wrapper_path)
    
    # Apply fixes
    loader_fixed = fix_loader_determine_device(loader_path)
    wrapper_init_fixed = fix_wrapper_init(wrapper_path)
    wrapper_inference_fixed = fix_wrapper_run_inference(wrapper_path)
    
    if loader_fixed and wrapper_init_fixed and wrapper_inference_fixed:
        logger.info("✅ All fixes have been successfully applied.")
        logger.info("To test, restart the server and try a Spanish to English translation.")
        
        # Print example curl command for testing
        logger.info("\nTest with this curl command:")
        logger.info('curl -X POST http://localhost:5000/pipeline/translate -H "Content-Type: application/json" -d \'{"text": "Estoy muy feliz de conocerte hoy", "source_language": "es", "target_language": "en"}\'')
        
        return 0
    else:
        logger.error("❌ Some fixes could not be applied. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())