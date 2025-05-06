#!/usr/bin/env python3
"""
Fix for the translation translator.py module
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_translator_module():
    """
    Fix the translator.py module to ensure fallback to MBART works correctly
    """
    # Get project root (3 levels up from script in scripts/fixes/)
    project_root = Path(__file__).parent.parent.parent
    translator_file = project_root / "app" / "core" / "pipeline" / "translator.py"
    
    print(f"Fixing translator.py at {translator_file}")
    
    # Back up the file
    backup_file = project_root / "app" / "core" / "pipeline" / "translator.py.bak"
    
    # Only create backup if it doesn't exist
    if not backup_file.exists():
        print(f"Creating backup at {backup_file}")
        with open(translator_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
    
    # Read the file content
    with open(translator_file, 'r') as f:
        content = f.read()
    
    # Fix 1: Always try to use MBART for translation when using default endpoint
    # Modify the translate_text method to always try MBART first
    translate_text_start = """async def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        model_id: Optional[str] = None,
        glossary_id: Optional[str] = None,
        preserve_formatting: bool = True,
        formality: Optional[str] = None,
        verify: bool = False,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        use_fallback: bool = True
    ) -> Dict[str, Any]:"""
    
    if translate_text_start not in content:
        print("Error: Could not find translate_text method")
        return False
    
    # Find the section where model_id is checked
    model_id_check = """        # Default to "translation" (now points to MBART in config) if no model specified
        if model_id is None or model_id == "mt5_translation":
            # Get MBART language codes
            mbart_source_lang = self._get_mbart_language_code(source_language)
            mbart_target_lang = self._get_mbart_language_code(target_language)
            
            # Use "translation" as primary model (now configured as MBART in model_registry.json)
            logger.info(f"Using MBART as primary translation model for {source_language} to {target_language}")
            model_id = "translation"
            
            # Create translation request with MBART specific parameters
            request = TranslationRequest(
                text=text,
                source_language=source_language,
                target_language=target_language,
                model_name=model_id,
                glossary_id=glossary_id,
                preserve_formatting=preserve_formatting,
                formality=formality,
                context=[],  # No context in simplified interface
                domain=None,  # No domain in simplified interface
                parameters={
                    "mbart_source_lang": mbart_source_lang,
                    "mbart_target_lang": mbart_target_lang,
                    "primary": True
                }
            )
        else:
            # Use the specified model
            logger.info(f"Using specified model {model_id} for translation from {source_language} to {target_language}")
            request = TranslationRequest(
                text=text,
                source_language=source_language,
                target_language=target_language,
                model_name=model_id,
                glossary_id=glossary_id,
                preserve_formatting=preserve_formatting,
                formality=formality,
                context=[],  # No context in simplified interface
                domain=None  # No domain in simplified interface
            )"""
    
    # Replace with more robust implementation
    improved_model_id_check = """        # Always use MBART as primary model unless explicitly specified otherwise
        if model_id is None or model_id == "translation" or model_id == "mt5_translation" or model_id == "":
            # Get MBART language codes
            mbart_source_lang = self._get_mbart_language_code(source_language)
            mbart_target_lang = self._get_mbart_language_code(target_language)
            
            # Use "mbart_translation" explicitly to ensure we get the right model
            logger.info(f"Using MBART as primary translation model for {source_language} to {target_language}")
            model_id = "mbart_translation"  # Use MBART translation model directly
            
            # Create translation request with MBART specific parameters
            request = TranslationRequest(
                text=text,
                source_language=source_language,
                target_language=target_language,
                model_name=model_id,
                glossary_id=glossary_id,
                preserve_formatting=preserve_formatting,
                formality=formality,
                context=[],  # No context in simplified interface
                domain=None,  # No domain in simplified interface
                parameters={
                    "mbart_source_lang": mbart_source_lang,
                    "mbart_target_lang": mbart_target_lang,
                    "primary": True
                }
            )
        else:
            # Use the specified model
            logger.info(f"Using specified model {model_id} for translation from {source_language} to {target_language}")
            request = TranslationRequest(
                text=text,
                source_language=source_language,
                target_language=target_language,
                model_name=model_id,
                glossary_id=glossary_id,
                preserve_formatting=preserve_formatting,
                formality=formality,
                context=[],  # No context in simplified interface
                domain=None  # No domain in simplified interface
            )"""
    
    if model_id_check not in content:
        print("Warning: Could not find model_id check section. This may mean it was already modified.")
    else:
        # Replace the model_id check with improved version
        new_content = content.replace(model_id_check, improved_model_id_check)
        
        # Write the updated content back to the file
        with open(translator_file, 'w') as f:
            f.write(new_content)
        
        print("Successfully updated translator.py with improved model selection")
        return True
    
    return False

if __name__ == "__main__":
    # Fix the translation pipeline to use mbart_translation model directly 
    success = fix_translator_module()
    print(f"Fix translator module: {'Success' if success else 'Failed'}")
    
    # Print instruction to restart the server
    print("\nTo complete the fix, restart the server with:")
    print("pkill -f 'python.*main.py' && cd $(dirname $(dirname $(dirname $0))) && python app/main.py")