#!/usr/bin/env python3
# NLLB wrapper patch for TranslationModelWrapper
# Adds support for NLLB's specific language codes

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

from app.services.models.wrapper import TranslationModelWrapper
from app.core.pipeline.nllb_language_mapping import get_nllb_code, get_iso_code

# Patch the TranslationModelWrapper._prepare_input method to handle NLLB language codes
original_prepare_input = TranslationModelWrapper._prepare_input

def patched_prepare_input(self, input_data):
    # Check if this is an NLLB model
    is_nllb = "nllb" in self.model_name.lower()
    
    # Use the original method first
    result = original_prepare_input(self, input_data)
    
    # If this is an NLLB model, adjust the language codes
    if is_nllb:
        # Get source and target languages
        source_lang = input_data.get("source_language", "en")
        target_lang = input_data.get("target_language", "en")
        
        # Convert to NLLB format if needed
        if "_" not in source_lang:
            source_lang = get_nllb_code(source_lang)
        if "_" not in target_lang:
            target_lang = get_nllb_code(target_lang)
        
        # Update the tokenizer kwargs
        if self.tokenizer is not None:
            self.tokenizer.src_lang = source_lang
            self.tokenizer.tgt_lang = target_lang
        
        # Update the result
        result["src_lang"] = source_lang
        result["tgt_lang"] = target_lang
    
    return result

# Apply the patch
TranslationModelWrapper._prepare_input = patched_prepare_input

print("✅ Successfully patched TranslationModelWrapper for NLLB support")
