#!/usr/bin/env python3
"""
Direct patch for the wrapper selection issue
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_wrapper_module():
    """Patch the wrapper.py module to fix wrapper selection"""
    # Get project root (3 levels up from script in scripts/fixes/)
    project_root = Path(__file__).parent.parent.parent
    # Path to the wrapper.py file
    wrapper_path = project_root / "app" / "services" / "models" / "wrapper.py"
    
    if not wrapper_path.exists():
        logger.error(f"Wrapper file not found at {wrapper_path}")
        return False
    
    logger.info(f"Patching wrapper module at {wrapper_path}")
    
    # Create backup
    backup_path = wrapper_path.with_suffix(".py.bak2")
    if not backup_path.exists():
        logger.info(f"Creating backup at {backup_path}")
        with open(wrapper_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    # Read the current content
    with open(wrapper_path, 'r') as f:
        content = f.read()
    
    # Direct implementation of create_model_wrapper function that always returns TranslationModelWrapper for mbart_translation
    new_function = """def create_model_wrapper(model_type: str, model, tokenizer, config: Dict[str, Any] = None, **kwargs):
    \"\"\"
    Factory function to create the appropriate wrapper for a model type
    
    Args:
        model_type: Type of model to wrap
        model: The model to wrap
        tokenizer: The tokenizer to use
        config: Configuration parameters
        
    Returns:
        BaseModelWrapper: Appropriate model wrapper
    \"\"\"
    # Special case for mbart_translation - ALWAYS use TranslationModelWrapper
    if model_type == 'mbart_translation' or 'mbart' in model_type.lower():
        logger.info(f"Using TranslationModelWrapper for MBART model type: {model_type}")
        return TranslationModelWrapper(model, tokenizer, config, **kwargs)
    
    # Special case for translation - ALWAYS use TranslationModelWrapper 
    if model_type == 'translation' or 'translation' in model_type.lower():
        logger.info(f"Using TranslationModelWrapper for translation model type: {model_type}")
        return TranslationModelWrapper(model, tokenizer, config, **kwargs)
        
    # Use wrapper_map for other model types
    if model_type in wrapper_map and wrapper_map[model_type]:
        logger.info(f"Using wrapper from map for model type: {model_type}")
        wrapper_class = wrapper_map[model_type]
        return wrapper_class(model, tokenizer, config, **kwargs)
    
    # Special handling for embedding models
    if model_type == 'embedding_model':
        try:
            from app.services.models.embedding_wrapper import EmbeddingModelWrapper
            return EmbeddingModelWrapper(model, tokenizer, config, **kwargs)
        except ImportError:
            logger.warning(f"EmbeddingModelWrapper not available, using base wrapper for {model_type}")
    
    # Fallback to base wrapper with warning
    logger.warning(f"No specific wrapper for model type: {model_type}, using base wrapper")
    return BaseModelWrapper(model, tokenizer, config, **kwargs)"""
    
    # Find the existing create_model_wrapper/get_wrapper_for_model function
    import re
    pattern = r'def get_wrapper_for_model\(.*?\).*?create_model_wrapper = get_wrapper_for_model'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        logger.error("Could not find get_wrapper_for_model function in wrapper.py")
        return False
    
    # Replace with our new implementation
    new_content = content.replace(
        match.group(0),
        new_function + "\n\n# Alias for backward compatibility\nget_wrapper_for_model = create_model_wrapper"
    )
    
    # Write the updated content
    with open(wrapper_path, 'w') as f:
        f.write(new_content)
    
    logger.info("Successfully patched wrapper module with fixed wrapper selection")
    return True

def main():
    """Apply the patch and provide instructions"""
    success = patch_wrapper_module()
    
    if success:
        print("\n✅ Successfully applied patch to fix wrapper selection.")
        print("\nTo apply the changes, restart the server with:")
        print("\npkill -f 'python.*main.py' && cd $(dirname $(dirname $(dirname $0))) && python app/main.py")
        return 0
    else:
        print("\n❌ Failed to apply patch.")
        return 1

if __name__ == "__main__":
    sys.exit(main())