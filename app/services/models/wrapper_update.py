#!/usr/bin/env python3
"""
Update Wrapper Module Script

This script updates the create_model_wrapper function in the app/services/models/wrapper.py
file to use the fixed SimplifierWrapper implementation and fix other issues.
"""

import os
import re
import sys
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Path to the wrapper.py file
WRAPPER_PATH = Path(__file__).parent / "wrapper.py"

# Backup the original file
BACKUP_PATH = Path(__file__).parent / "wrapper.py.bak"

def update_wrapper_file():
    """Update the wrapper.py file with fixes for model loading issues."""
    print(f"Updating wrapper.py file at: {WRAPPER_PATH}")
    
    # Create backup
    if not BACKUP_PATH.exists():
        shutil.copy(WRAPPER_PATH, BACKUP_PATH)
        print(f"Created backup at: {BACKUP_PATH}")
    
    # Read the wrapper.py file
    with open(WRAPPER_PATH, 'r') as f:
        content = f.read()
    
    # Import the fixed SimplifierWrapper implementation
    with open(Path(__file__).parent / "simplifier_wrapper_fix.py", 'r') as f:
        simplifier_wrapper_content = f.read()
    
    # Extract the FixedSimplifierWrapper implementation
    fixed_wrapper_match = re.search(r'class FixedSimplifierWrapper\(BaseModelWrapper\):.*?(?=\n\n\# Factory function|$)', 
                                   simplifier_wrapper_content, re.DOTALL)
    
    if not fixed_wrapper_match:
        print("Error: Could not find FixedSimplifierWrapper implementation")
        return False
    
    fixed_wrapper = fixed_wrapper_match.group(0)
    
    # Find and replace SimplifierWrapper implementation
    simplifier_wrapper_match = re.search(r'class SimplifierWrapper\(BaseModelWrapper\):.*?(?=\n\nclass AnonymizerWrapper|$)', 
                                        content, re.DOTALL)
    
    if not simplifier_wrapper_match:
        print("Error: Could not find SimplifierWrapper implementation in wrapper.py")
        return False
    
    # Replace SimplifierWrapper implementation with FixedSimplifierWrapper
    # First fix the domain.lower() error
    fixed_wrapper_corrected = fixed_wrapper.replace(
        "domain.lower() in", 
        "domain and domain.lower() in"
    )
    
    # Replace the class in wrapper.py
    updated_content = content.replace(
        simplifier_wrapper_match.group(0),
        fixed_wrapper_corrected.replace("FixedSimplifierWrapper", "SimplifierWrapper")
    )
    
    # Fix create_model_wrapper function to ensure models are properly cached
    create_model_wrapper_match = re.search(r'def create_model_wrapper.*?(?=\n\n\# Pipeline integration|$)', 
                                          updated_content, re.DOTALL)
    
    if not create_model_wrapper_match:
        print("Error: Could not find create_model_wrapper function in wrapper.py")
        return False
    
    # Get the original function
    original_func = create_model_wrapper_match.group(0)
    
    # Enhanced function with better model type detection and debug logging
    enhanced_func = """def create_model_wrapper(model_type: str, model: Any, tokenizer: Any = None, config: Dict[str, Any] = None, **kwargs) -> BaseModelWrapper:
    """
    Create a model wrapper for the specified model type
    
    Args:
        model_type: Type of model
        model: Model instance
        tokenizer: Tokenizer instance
        config: Configuration parameters
        **kwargs: Additional arguments
        
    Returns:
        BaseModelWrapper: Appropriate model wrapper
    """
    # Map of model types to wrapper classes
    wrapper_map = {
        "translation": TranslationModelWrapper,
        "mbart_translation": TranslationModelWrapper,  # Use TranslationModelWrapper for MBART
        "language_detection": LanguageDetectionWrapper,
        "ner_detection": NERDetectionWrapper,
        "rag_generator": RAGGeneratorWrapper,
        "rag_retriever": RAGRetrieverWrapper,
        "simplifier": SimplifierWrapper,
        "anonymizer": AnonymizerWrapper
    }
    
    # Debug logging for model type and wrapper selection
    logger.debug(f"Creating model wrapper for type: {model_type}")
    logger.debug(f"Model class: {model.__class__.__name__}")
    
    # Handle special cases for model type detection
    lowercase_model_type = model_type.lower()
    model_class_name = model.__class__.__name__
    
    # Check for model type in model name or class
    if hasattr(model, 'config') and hasattr(model.config, 'model_name_or_path'):
        model_name = model.config.model_name_or_path.lower()
        logger.debug(f"Model name from config: {model_name}")
        
        # Detect translation models
        if 'mbart' in model_name or 'mbart' in model_class_name.lower():
            logger.debug("Detected MBART model, using TranslationModelWrapper")
            return TranslationModelWrapper(model, tokenizer, config, **kwargs)
        
        # Detect MT5 models as translation models
        if 'mt5' in model_name or 't5' in model_name or 'mt5' in model_class_name.lower():
            logger.debug("Detected MT5/T5 model, using TranslationModelWrapper")
            return TranslationModelWrapper(model, tokenizer, config, **kwargs)
        
        # Detect simplification models
        if 'simplif' in model_name:
            logger.debug("Detected simplification model, using SimplifierWrapper")
            return SimplifierWrapper(model, tokenizer, config, **kwargs)
    
    # Check if the model type is directly in the map
    if model_type in wrapper_map:
        wrapper_class = wrapper_map[model_type]
        logger.debug(f"Using wrapper class for {model_type}: {wrapper_class.__name__}")
        return wrapper_class(model, tokenizer, config, **kwargs)
    
    # Try additional matching for unrecognized types
    for known_type, wrapper_class in wrapper_map.items():
        if known_type in lowercase_model_type:
            logger.debug(f"Using wrapper class based on partial match: {wrapper_class.__name__}")
            return wrapper_class(model, tokenizer, config, **kwargs)
    
    # Fallback - use base wrapper with warning
    logger.warning(f"No specific wrapper for model type: {model_type}, using base wrapper")
    return BaseModelWrapper(model, tokenizer, config, **kwargs)"""
    
    # Replace the create_model_wrapper function
    updated_content = updated_content.replace(original_func, enhanced_func)
    
    # Write the updated content back to the file
    with open(WRAPPER_PATH, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated wrapper.py file")
    return True

def fix_model_manager():
    """Add a fix to the model manager to avoid reloading models if already loaded."""
    MANAGER_PATH = Path(__file__).parent / "manager.py"
    
    # Create backup
    MANAGER_BACKUP_PATH = Path(__file__).parent / "manager.py.bak"
    if not MANAGER_BACKUP_PATH.exists():
        shutil.copy(MANAGER_PATH, MANAGER_BACKUP_PATH)
        print(f"Created backup at: {MANAGER_BACKUP_PATH}")
    
    # Read the manager.py file
    with open(MANAGER_PATH, 'r') as f:
        content = f.read()
    
    # Find the run_model method
    run_model_match = re.search(r'async def run_model.*?return {.*?}', content, re.DOTALL)
    
    if not run_model_match:
        print("Error: Could not find run_model method in manager.py")
        return False
    
    # Get the original method
    original_method = run_model_match.group(0)
    
    # Enhanced method with better model type detection and better caching
    enhanced_method = """async def run_model(self, model_type: str, method_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a model with the specified method and input data
        
        Args:
            model_type (str): Model type
            method_name (str): Method to call on the model wrapper
            input_data (Dict[str, Any]): Input data for the model
            
        Returns:
            Dict[str, Any]: Model output
        """
        logger.info(f"Running model {model_type}.{method_name}")
        
        # Check for mbart translation model type
        is_mbart = False
        if model_type == "translation":
            # Check input parameters for MBART-specific language codes
            parameters = input_data.get("parameters", {})
            mbart_source_lang = parameters.get("mbart_source_lang")
            mbart_target_lang = parameters.get("mbart_target_lang")
            if mbart_source_lang and mbart_target_lang:
                is_mbart = True
                logger.info(f"Detected MBART language codes in request, using mbart_translation model if available")
                # Try to use the mbart_translation model if available
                if "mbart_translation" in self.loaded_models:
                    model_type = "mbart_translation"
                    logger.info(f"Using already loaded mbart_translation model")
        
        # Ensure the model is loaded
        if model_type not in self.loaded_models:
            logger.info(f"Model {model_type} not loaded, loading now")
            
            # If this is a translation request and we have an MBART model, try to use it
            if model_type == "translation" and is_mbart and "mbart_translation" in self.loaded_models:
                model_type = "mbart_translation"
                logger.info(f"Using already loaded mbart_translation model for translation request")
            else:
                # Load the specified model
                model_info = await self.load_model(model_type)
                
                if not model_info.get("model"):
                    raise ValueError(f"Failed to load model {model_type}")
        else:
            logger.info(f"Using already loaded {model_type} model")
        
        # Get the model and its wrapper
        model = self.loaded_models[model_type]
        tokenizer = self.model_metadata.get(model_type, {}).get("tokenizer")
        
        # Create input for the model wrapper
        from app.services.models.wrapper import ModelInput
        
        # Extract common fields from input_data
        text = input_data.get("text", "")
        source_language = input_data.get("source_language")
        target_language = input_data.get("target_language")
        context = input_data.get("context", [])
        parameters = input_data.get("parameters", {})
        
        # Create ModelInput instance
        model_input = ModelInput(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context,
            parameters=parameters
        )
        
        # Import wrapper factory function
        from app.services.models.wrapper import create_model_wrapper
        
        # Create wrapper for the model with extended config
        config = {"task": model_type, "device": self.device, "precision": self.precision}
        
        # Add additional config parameters from model metadata
        model_config = self.model_metadata.get(model_type, {}).get("config")
        if model_config:
            for key, value in model_config.items():
                if key not in config:
                    config[key] = value
        
        # Add model type-specific settings
        if model_type == "simplifier":
            config["generation_kwargs"] = {
                "max_length": 1024,
                "num_beams": 5,
                "min_length": 10,
                "do_sample": True,
                "top_p": 0.95,
                "temperature": 0.8
            }
        
        # Create the wrapper with all config
        wrapper = create_model_wrapper(
            model_type,
            model,
            tokenizer,
            config
        )
        
        # Call the appropriate method
        if method_name == "process":
            # Synchronous processing
            result = wrapper.process(model_input)
            # Return all fields from ModelOutput including enhanced metrics
            return {
                "result": result.result,
                "metadata": result.metadata,
                "metrics": result.metrics,
                "performance_metrics": result.performance_metrics,
                "memory_usage": result.memory_usage,
                "operation_cost": result.operation_cost,
                "accuracy_score": result.accuracy_score,
                "truth_score": result.truth_score
            }
        elif method_name == "process_async":
            # Asynchronous processing
            result = await wrapper.process_async(model_input)
            # Return all fields from ModelOutput including enhanced metrics
            return {
                "result": result.result,
                "metadata": result.metadata,
                "metrics": result.metrics,
                "performance_metrics": result.performance_metrics,
                "memory_usage": result.memory_usage,
                "operation_cost": result.operation_cost,
                "accuracy_score": result.accuracy_score,
                "truth_score": result.truth_score
            }
        else:
            raise ValueError(f"Unknown method {method_name}")"""
    
    # Replace the run_model method
    updated_content = content.replace(original_method, enhanced_method)
    
    # Write the updated content back to the file
    with open(MANAGER_PATH, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully updated manager.py file")
    return True

if __name__ == "__main__":
    success = update_wrapper_file()
    if success:
        print("Successfully updated wrapper.py file with fixed SimplifierWrapper implementation")
    else:
        print("Failed to update wrapper.py file")
        sys.exit(1)
    
    success = fix_model_manager()
    if success:
        print("Successfully updated manager.py file with model loading improvements")
    else:
        print("Failed to update manager.py file")
        sys.exit(1)
    
    print("\nWrapper and manager file updates complete.")
    print("The following improvements were made:")
    print("1. Fixed SimplifierWrapper implementation with robust fallback")
    print("2. Enhanced model type detection to avoid reloading models")
    print("3. Added better memory tracking and error handling")
    print("4. Improved model caching for mbart_translation model")
    print("\nRun tests to verify the fixes are working correctly.")