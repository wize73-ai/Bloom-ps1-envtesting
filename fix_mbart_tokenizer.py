#!/usr/bin/env python3
"""
Fix for MBART tokenizer warnings in the TranslationModelWrapper class.

This script provides a fix for the issue where MBART tokenizers that don't support 
the 'src_lang' parameter generate warnings. The solution checks the tokenizer's 
capabilities before attempting to use the parameter, rather than relying on 
try/except blocks that generate warnings.
"""

import os
import sys
import inspect
import logging
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_wrapper_module_path():
    """Find the path to the module containing the TranslationModelWrapper class."""
    # Common locations to check
    potential_paths = [
        Path("app/services/models/wrapper.py"),
        Path("app/models/wrapper.py"),
        Path("models/wrapper.py"),
    ]
    
    # Check each potential path
    for path in potential_paths:
        if path.exists():
            return path
    
    # If not found in the common locations, search the entire project
    logger.info("Searching for wrapper.py file in the project...")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == "wrapper.py":
                path = Path(os.path.join(root, file))
                if "TranslationModelWrapper" in path.read_text():
                    return path
    
    return None

def load_module(module_path):
    """Load a Python module from a file path."""
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def patch_translation_model_wrapper(module):
    """
    Patch the TranslationModelWrapper._preprocess method to handle MBART tokenizer gracefully.
    
    This function modifies the _preprocess method to check if the tokenizer supports 
    the src_lang parameter before attempting to use it, avoiding unnecessary warnings.
    """
    if not hasattr(module, "TranslationModelWrapper"):
        logger.error("TranslationModelWrapper class not found in the loaded module.")
        return False
    
    # Get the original _preprocess method
    original_preprocess = module.TranslationModelWrapper._preprocess
    
    # Define the new _preprocess method
    def patched_preprocess(self, texts, source_lang=None, *args, **kwargs):
        """
        Improved preprocessing method that checks tokenizer capabilities.
        
        This patched method checks if the tokenizer supports src_lang before
        attempting to use it, avoiding unnecessary warnings.
        """
        # Original method may handle various preprocessing tasks, so we keep that code
        if not isinstance(texts, list):
            texts = [texts]
        
        # Get source language code (original code preserved)
        source_lang_code = source_lang
        if hasattr(self, "source_lang_map") and source_lang in self.source_lang_map:
            source_lang_code = self.source_lang_map.get(source_lang, source_lang)
        
        # Check if tokenizer supports src_lang before trying to use it
        supports_src_lang = False
        try:
            # Check if the tokenizer's forward signature has src_lang
            sig = inspect.signature(self.tokenizer.__call__)
            supports_src_lang = 'src_lang' in sig.parameters
            
            # Alternative check - see if tokenizer has src_lang attribute or method
            if not supports_src_lang:
                supports_src_lang = (hasattr(self.tokenizer, 'src_lang') or 
                                   hasattr(self.tokenizer, 'set_src_lang_special_tokens'))
                                   
            logger.debug(f"Tokenizer supports src_lang: {supports_src_lang}")
        except Exception as e:
            logger.debug(f"Error checking tokenizer signature: {e}")
            supports_src_lang = False
        
        # Now branch based on src_lang support
        if supports_src_lang and source_lang_code:
            logger.debug(f"Using MBART tokenizer with src_lang={source_lang_code}")
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.get("max_length", 1024),
                src_lang=source_lang_code
            )
        else:
            # Use standard tokenization without warning
            logger.debug("Using standard tokenization without src_lang")
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.get("max_length", 1024)
            )
        
        return inputs
    
    # Apply the patch
    module.TranslationModelWrapper._preprocess = patched_preprocess
    logger.info("Successfully patched TranslationModelWrapper._preprocess method")
    return True

def create_backup(module_path):
    """Create a backup of the original module file."""
    backup_path = module_path.with_suffix('.py.bak')
    
    # Check if backup already exists
    if backup_path.exists():
        logger.info(f"Backup already exists at {backup_path}")
        return
    
    # Create backup
    with open(module_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    
    logger.info(f"Created backup at {backup_path}")

def write_patched_module(module_path, module):
    """Write the patched module back to disk."""
    # Get all classes from the module
    class_dict = {name: obj for name, obj in module.__dict__.items() 
                  if isinstance(obj, type)}
    
    # Read the original file
    with open(module_path, 'r') as f:
        lines = f.readlines()
    
    # Find the TranslationModelWrapper class and its _preprocess method
    in_class = False
    in_preprocess = False
    class_indent = ""
    method_indent = ""
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if "class TranslationModelWrapper" in line:
            in_class = True
            class_indent = line[:line.find("class")]
        elif in_class and "def _preprocess" in line:
            in_preprocess = True
            method_indent = line[:line.find("def")]
            start_line = i
        elif in_preprocess and line.strip() and not line.startswith(method_indent + " "):
            in_preprocess = False
            end_line = i
            break
    
    if start_line is None or end_line is None:
        logger.error("Could not find the _preprocess method in the source file.")
        return False
    
    # Generate the new method source code
    new_method_source = [
        f"{method_indent}def _preprocess(self, texts, source_lang=None, *args, **kwargs):\n",
        f"{method_indent}    \"\"\"\n",
        f"{method_indent}    Preprocess input texts for the model.\n",
        f"{method_indent}    \n",
        f"{method_indent}    This method handles tokenization with proper src_lang support checking\n",
        f"{method_indent}    to avoid unnecessary warnings.\n",
        f"{method_indent}    \"\"\"\n",
        f"{method_indent}    # Convert single text to list for batch processing\n",
        f"{method_indent}    if not isinstance(texts, list):\n",
        f"{method_indent}        texts = [texts]\n",
        f"{method_indent}    \n",
        f"{method_indent}    # Get source language code\n",
        f"{method_indent}    source_lang_code = source_lang\n",
        f"{method_indent}    if hasattr(self, \"source_lang_map\") and source_lang in self.source_lang_map:\n",
        f"{method_indent}        source_lang_code = self.source_lang_map.get(source_lang, source_lang)\n",
        f"{method_indent}    \n",
        f"{method_indent}    # Check if tokenizer supports src_lang before trying to use it\n",
        f"{method_indent}    supports_src_lang = False\n",
        f"{method_indent}    try:\n",
        f"{method_indent}        # Check if the tokenizer's forward signature has src_lang\n",
        f"{method_indent}        import inspect\n",
        f"{method_indent}        sig = inspect.signature(self.tokenizer.__call__)\n",
        f"{method_indent}        supports_src_lang = 'src_lang' in sig.parameters\n",
        f"{method_indent}        \n",
        f"{method_indent}        # Alternative check - see if tokenizer has src_lang attribute or method\n",
        f"{method_indent}        if not supports_src_lang:\n",
        f"{method_indent}            supports_src_lang = (hasattr(self.tokenizer, 'src_lang') or \n",
        f"{method_indent}                               hasattr(self.tokenizer, 'set_src_lang_special_tokens'))\n",
        f"{method_indent}            \n",
        f"{method_indent}        # For debugging only\n",
        f"{method_indent}        import logging\n",
        f"{method_indent}        logging.getLogger(__name__).debug(f\"Tokenizer supports src_lang: {{supports_src_lang}}\")\n",
        f"{method_indent}    except Exception as e:\n",
        f"{method_indent}        import logging\n",
        f"{method_indent}        logging.getLogger(__name__).debug(f\"Error checking tokenizer signature: {{e}}\")\n",
        f"{method_indent}        supports_src_lang = False\n",
        f"{method_indent}    \n",
        f"{method_indent}    # Now branch based on src_lang support\n",
        f"{method_indent}    if supports_src_lang and source_lang_code:\n",
        f"{method_indent}        # Use src_lang parameter\n",
        f"{method_indent}        inputs = self.tokenizer(\n",
        f"{method_indent}            texts, \n",
        f"{method_indent}            return_tensors=\"pt\", \n",
        f"{method_indent}            padding=True, \n",
        f"{method_indent}            truncation=True,\n",
        f"{method_indent}            max_length=self.config.get(\"max_length\", 1024),\n",
        f"{method_indent}            src_lang=source_lang_code\n",
        f"{method_indent}        )\n",
        f"{method_indent}    else:\n",
        f"{method_indent}        # Use standard tokenization without src_lang and without warning\n",
        f"{method_indent}        inputs = self.tokenizer(\n",
        f"{method_indent}            texts, \n",
        f"{method_indent}            return_tensors=\"pt\", \n",
        f"{method_indent}            padding=True, \n",
        f"{method_indent}            truncation=True,\n",
        f"{method_indent}            max_length=self.config.get(\"max_length\", 1024)\n",
        f"{method_indent}        )\n",
        f"{method_indent}    \n",
        f"{method_indent}    return inputs\n",
    ]
    
    # Replace the old method with the new one
    new_lines = lines[:start_line] + new_method_source + lines[end_line:]
    
    # Write the modified file
    with open(module_path, 'w') as f:
        f.writelines(new_lines)
    
    logger.info(f"Successfully patched {module_path}")
    return True

def verify_patch(module_path):
    """Verify that the patch was applied successfully."""
    with open(module_path, 'r') as f:
        content = f.read()
    
    # Check if the patch was applied correctly
    markers = ["supports_src_lang = False", "inspect.signature", "supports_src_lang and source_lang_code"]
    return all(marker in content for marker in markers)

def main():
    logger.info("Starting MBART tokenizer fix script...")
    
    # Find the wrapper module path
    wrapper_path = find_wrapper_module_path()
    if not wrapper_path:
        logger.error("Could not find the wrapper module containing TranslationModelWrapper.")
        return 1
    
    logger.info(f"Found wrapper module at: {wrapper_path}")
    
    # Create a backup of the original file
    create_backup(wrapper_path)
    
    try:
        # Load the module
        module = load_module(wrapper_path)
        
        # Directly modify the file (more reliable than monkey patching)
        if not write_patched_module(wrapper_path, module):
            logger.error("Failed to write patched module.")
            return 1
        
        # Verify the patch
        if verify_patch(wrapper_path):
            logger.info("✅ Patch successfully applied and verified.")
        else:
            logger.error("❌ Patch verification failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Error occurred during patching: {e}")
        return 1
    
    logger.info("MBART tokenizer fix completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())