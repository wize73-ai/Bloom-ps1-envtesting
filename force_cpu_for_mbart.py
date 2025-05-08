#!/usr/bin/env python3
"""
Force CPU device for MBART models when running on Apple Silicon.

This script directly modifies the ModelLoader._load_transformers_model method to always
force CPU usage for MBART models on Apple MPS devices, regardless of what's specified
in the config or elsewhere.
"""

import os
import sys
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_loader_file():
    """Find the ModelLoader file."""
    base_dir = Path(".")
    loader_paths = list(base_dir.glob("**/loader.py"))
    loader_paths = [p for p in loader_paths if "app/services/models" in str(p)]
    
    if not loader_paths:
        logger.error("Could not find loader.py in app/services/models/")
        return None
        
    return loader_paths[0]

def create_backup(file_path):
    """Create a backup of the file."""
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak_force_cpu")
    
    # Check if backup already exists
    if backup_path.exists():
        logger.info(f"Backup already exists at {backup_path}")
        return
    
    # Create backup
    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
        dst.write(src.read())
    
    logger.info(f"Created backup at {backup_path}")

def modify_load_transformers_method(file_path):
    """Add code to force CPU usage for MBART models on MPS devices."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the _load_transformers_model method
    method_pattern = r'def _load_transformers_model\(self, model_config: ModelConfig, device: str\) -> Any:.*?(?=\n    def |\n    async def |\Z)'
    method_match = re.search(method_pattern, content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find _load_transformers_model method")
        return False
    
    method_code = method_match.group(0)
    
    # Check if method already has our force CPU code for MBART on MPS
    if "# Force CPU for MBART on MPS regardless of what was specified" in method_code:
        logger.info("Method already has the forced CPU fix for MBART")
        return True
    
    # Find the appropriate insertion point after device extraction but before model loading
    insertion_point = method_code.find("logger.info(f\"Loading Transformers model: {model_config.model_name} on {device}\")")
    
    if insertion_point == -1:
        logger.error("Could not find appropriate insertion point in _load_transformers_model method")
        return False
    
    # Find the correct line in the code
    lines = method_code.split('\n')
    insertion_line = 0
    
    for i, line in enumerate(lines):
        if "logger.info(f\"Loading Transformers model:" in line:
            insertion_line = i
            break
    
    # Code to insert to force CPU for MBART
    force_cpu_code = """
        # Force CPU for MBART on MPS regardless of what was specified
        if "mps" in device and model_config:
            model_name = ""
            task = ""
            
            # Get model name and task
            if hasattr(model_config, "model_name"):
                model_name = model_config.model_name.lower()
            
            if hasattr(model_config, "task"):
                task = model_config.task.lower()
            
            # Check if this is an MBART model or translation task
            is_mbart = "mbart" in model_name or "mbart" in task
            is_translation = task == "translation" or task == "mbart_translation"
            
            if is_mbart or is_translation:
                logger.warning(f"⚠️ Forcing CPU device for {model_config.model_name} due to MPS compatibility issues")
                device = "cpu"  # Force CPU device
"""
    
    # Insert the code at the appropriate position
    updated_lines = lines[:insertion_line] + [force_cpu_code] + lines[insertion_line:]
    updated_method = '\n'.join(updated_lines)
    
    # Replace the old method with the updated one
    updated_content = content.replace(method_code, updated_method)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    logger.info(f"Successfully updated {file_path} to force CPU for MBART models on MPS")
    return True

def main():
    """Main entry point."""
    logger.info("Starting fix to force CPU for MBART models on MPS devices...")
    
    # Find loader file
    loader_file = find_loader_file()
    if not loader_file:
        logger.error("Could not find loader file")
        return 1
    
    logger.info(f"Found loader file at: {loader_file}")
    
    # Create backup
    create_backup(loader_file)
    
    # Apply fix
    if modify_load_transformers_method(loader_file):
        logger.info("✅ Successfully modified loader to force CPU for MBART models on MPS")
        logger.info("Restart the server for the changes to take effect.")
        return 0
    else:
        logger.error("❌ Failed to apply force CPU fix")
        return 1

if __name__ == "__main__":
    sys.exit(main())