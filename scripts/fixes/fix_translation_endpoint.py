#!/usr/bin/env python3
"""
Fix for the translation endpoint
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patch the wrapper_map in wrapper.py
def fix_wrapper_map():
    """
    Fix the wrapper_map initialization in wrapper.py
    """
    # Get project root (3 levels up from script in scripts/fixes/)
    project_root = Path(__file__).parent.parent.parent
    wrapper_file = project_root / "app" / "services" / "models" / "wrapper.py"
    
    print(f"Fixing wrapper_map in {wrapper_file}")
    
    # Back up the file
    backup_file = project_root / "app" / "services" / "models" / "wrapper.py.bak"
    
    # Only create backup if it doesn't exist
    if not backup_file.exists():
        print(f"Creating backup at {backup_file}")
        with open(wrapper_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
    
    # Read the file content
    with open(wrapper_file, 'r') as f:
        content = f.read()
    
    # Check if the global wrapper_map is initialized before use
    if "global wrapper_map" in content and "if not wrapper_map:" in content:
        print("File already has global wrapper_map initialization, no need to fix")
        return True
    
    # Fix: Ensure global wrapper_map is used in get_wrapper_for_model function
    replacement_code = """def get_wrapper_for_model(model_type: str, model, tokenizer, config: Dict[str, Any] = None, **kwargs):
    \"\"\"
    Factory function to get the appropriate wrapper for a model type
    
    Args:
        model_type: Type of model to wrap
        model: The model to wrap
        tokenizer: The tokenizer to use
        config: Configuration parameters
        
    Returns:
        BaseModelWrapper: Appropriate model wrapper
    \"\"\"
    # Update the global wrapper_map with the actual wrapper classes
    global wrapper_map
    
    # Debug logging for model type
    logger.debug(f"Creating wrapper for model type: {model_type}")
    
    # Handle model-specific cases for better performance and reliability
    model_name_lower = ""
    if hasattr(model, 'config') and hasattr(model.config, 'model_name_or_path'):
        model_name_lower = model.config.model_name_or_path.lower()
    
    # MBART special case
    if "mbart" in model_type.lower() or "mbart" in model_name_lower:
        logger.info(f"Using TranslationModelWrapper for MBART model")
        return TranslationModelWrapper(model, tokenizer, config, **kwargs)
    
    # MT5 special case
    if "mt5" in model_type.lower() or "mt5" in model_name_lower:
        logger.info(f"Using TranslationModelWrapper for MT5 model")
        return TranslationModelWrapper(model, tokenizer, config, **kwargs)
    
    # Initialize wrapper_map if empty
    if not wrapper_map:
        logger.debug("Initializing wrapper_map")
        wrapper_map.update({
            "translation": TranslationModelWrapper,
            "mbart_translation": TranslationModelWrapper,  # Use TranslationModelWrapper for MBART
            "language_detection": LanguageDetectionWrapper,
            "ner_detection": NERDetectionWrapper,
            "rag_generator": RAGGeneratorWrapper,
            "rag_retriever": RAGRetrieverWrapper,
            "simplifier": SimplificationModelWrapper,  # Use correct wrapper class
            "anonymizer": AnonymizerWrapper,
            # "embedding_model" will be patched at import time
        })
    
    if model_type in wrapper_map:
        wrapper_class = wrapper_map[model_type]
        if wrapper_class:
            logger.debug(f"Creating wrapper {wrapper_class.__name__} for model type {model_type}")
            return wrapper_class(model, tokenizer, config, **kwargs)
        else:
            if model_type == "embedding_model":
                # Try to import EmbeddingModelWrapper here to avoid circular imports
                try:
                    from app.services.models.embedding_wrapper import EmbeddingModelWrapper
                    return EmbeddingModelWrapper(model, tokenizer, config, **kwargs)
                except ImportError:
                    logger.warning(f"No specific wrapper for model type: {model_type}, using base wrapper")
                    return BaseModelWrapper(model, tokenizer, config, **kwargs)
    else:
        logger.warning(f"No specific wrapper for model type: {model_type}, using base wrapper")
        return BaseModelWrapper(model, tokenizer, config, **kwargs)"""
    
    # Find the original get_wrapper_for_model function
    import re
    match = re.search(r'def get_wrapper_for_model.*?create_model_wrapper = get_wrapper_for_model', content, re.DOTALL)
    
    if not match:
        print("Error: Could not find get_wrapper_for_model function")
        return False
    
    # Replace the function
    new_content = content.replace(
        match.group(0),
        replacement_code + "\n\n# Alias for backward compatibility\ncreate_model_wrapper = get_wrapper_for_model"
    )
    
    # Write the updated content back to the file
    with open(wrapper_file, 'w') as f:
        f.write(new_content)
    
    print("Successfully updated wrapper.py with improved model wrapper selection")
    return True

if __name__ == "__main__":
    # Fix the wrapper map to always return the correct wrapper for translation
    success = fix_wrapper_map()
    print(f"Fix wrapper_map: {'Success' if success else 'Failed'}")
    
    # Print instruction to restart the server
    print("\nTo complete the fix, restart the server with:")
    print("pkill -f 'python.*main.py' && cd $(dirname $(dirname $(dirname $0))) && python app/main.py")