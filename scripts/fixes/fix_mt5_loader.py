#!/usr/bin/env python3
"""
Fix for the MT5 model loading issue.
This script patches the loader.py file to use MT5ForConditionalGeneration for MT5 models.
"""

import os
import sys
import re
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

# Path to the loader.py file
loader_path = project_root / "app" / "services" / "models" / "loader.py"

def fix_mt5_loader():
    """Fix the MT5 model loading issue by updating the code to use MT5ForConditionalGeneration."""
    print(f"🔧 Patching {loader_path}...")
    
    # Read the current content
    with open(loader_path, "r") as f:
        content = f.read()
    
    # Add check for MT5 model in the _load_transformers_model method
    # Find the task-based selection section
    section_pattern = r"elif model_config\.task == \"translation\" or model_config\.task == \"rag_generation\" or model_config\.task == \"mbart_translation\":"
    
    # Check if MT5 is already handled
    if "mt5" in content and "MT5ForConditionalGeneration" in content:
        # Check if there's already a section that handles MT5 specifically
        if "mt5" in content.lower() and "mt5forconditionalgeneration" in content.lower():
            # Already has MT5 handling, but we'll check if it's correct
            print("✅ MT5 handling already exists in loader.py")
            
            # Make sure the check is using the right model class
            mt5_fixed = False
            
            # Fix any incorrect MT5 model loading
            if "model = AutoModel.from_pretrained" in content and "mt5" in content.lower():
                # Replace incorrect AutoModel with MT5ForConditionalGeneration for MT5
                content = content.replace(
                    "model = AutoModel.from_pretrained(",
                    "model = MT5ForConditionalGeneration.from_pretrained("
                )
                mt5_fixed = True
                
            if not mt5_fixed:
                print("📝 No incorrect MT5 loading found, skipping patch")
                return
        else:
            # No MT5 handling, add it
            # First, check if MBART section exists to insert after
            if section_pattern in content:
                # Insert MT5 handling after MBART section
                mbart_section = re.search(r"(# Check if this is an MBART model based on model name or task.*?)(\n\s+else:)", content, re.DOTALL)
                
                if mbart_section:
                    mt5_section = """
                # Check if this is an MT5 model based on model name
                elif "mt5" in model_config.model_name.lower() or model_config.task == "mt5_translation":
                    # Use specific MT5ForConditionalGeneration for MT5 models
                    logger.info(f"Loading MT5 model: {model_config.model_name}")
                    
                    # Try with MT5ForConditionalGeneration
                    try:
                        logger.info(f"Loading MT5 with MT5ForConditionalGeneration: {model_config.model_name}")
                        model = MT5ForConditionalGeneration.from_pretrained(
                            model_config.model_name,
                            **model_kwargs
                        )
                        logger.info(f"Successfully loaded MT5 model using MT5ForConditionalGeneration")
                        return model
                    except Exception as e:
                        logger.warning(f"Failed to load MT5 with MT5ForConditionalGeneration: {e}, trying AutoModelForSeq2SeqLM")
                        # Fall back to AutoModelForSeq2SeqLM
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_config.model_name,
                            **model_kwargs
                        )
                        logger.info(f"Successfully loaded MT5 model using AutoModelForSeq2SeqLM")
                        return model
                """
                    
                    # Insert the MT5 section
                    new_content = content.replace(mbart_section.group(2), mt5_section + mbart_section.group(2))
                    content = new_content
    else:
        # Add imports if missing
        if "MT5ForConditionalGeneration" not in content:
            # Add MT5ForConditionalGeneration to imports
            import_pattern = r"from transformers import \((.*?)\)"
            if re.search(import_pattern, content, re.DOTALL):
                imports = re.search(import_pattern, content, re.DOTALL).group(1)
                if "MT5ForConditionalGeneration" not in imports:
                    new_imports = imports.rstrip() + ",\n            MT5ForConditionalGeneration"
                    content = content.replace(imports, new_imports)
        
        # Add MT5 handling section
        mbart_section = re.search(r"(# Check if this is an MBART model based on model name or task.*?)(\n\s+else:)", content, re.DOTALL)
        
        if mbart_section:
            mt5_section = """
                # Check if this is an MT5 model based on model name
                elif "mt5" in model_config.model_name.lower() or model_config.task == "mt5_translation":
                    # Use specific MT5ForConditionalGeneration for MT5 models
                    logger.info(f"Loading MT5 model: {model_config.model_name}")
                    
                    # Try with MT5ForConditionalGeneration
                    try:
                        logger.info(f"Loading MT5 with MT5ForConditionalGeneration: {model_config.model_name}")
                        model = MT5ForConditionalGeneration.from_pretrained(
                            model_config.model_name,
                            **model_kwargs
                        )
                        logger.info(f"Successfully loaded MT5 model using MT5ForConditionalGeneration")
                        return model
                    except Exception as e:
                        logger.warning(f"Failed to load MT5 with MT5ForConditionalGeneration: {e}, trying AutoModelForSeq2SeqLM")
                        # Fall back to AutoModelForSeq2SeqLM
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_config.model_name,
                            **model_kwargs
                        )
                        logger.info(f"Successfully loaded MT5 model using AutoModelForSeq2SeqLM")
                        return model
                """
                
            # Insert the MT5 section
            new_content = content.replace(mbart_section.group(2), mt5_section + mbart_section.group(2))
            content = new_content
    
    # Write the updated content
    with open(loader_path, "w") as f:
        f.write(content)
    
    print("✅ Successfully patched loader.py to fix MT5 model loading")

if __name__ == "__main__":
    fix_mt5_loader()