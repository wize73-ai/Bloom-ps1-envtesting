#!/usr/bin/env python3
"""
Fix translation model loading issues by ensuring proper model class specification
and updating language code mappings for NLLB.
"""

import os
import sys
import json
import time
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import necessary modules
from app.utils.config import load_config

def fix_model_registry():
    """
    Update the model registry with proper configurations for translation models.
    Ensures NLLB, MBART, and MT5 configurations are correct.
    """
    print("🔧 Updating model registry configuration...")
    
    registry_path = project_root / "config" / "model_registry.json"
    
    # Load existing registry
    with open(registry_path, "r") as f:
        registry = json.load(f)
    
    # Set up NLLB as primary translation model
    registry["translation"] = {
        "model_name": "facebook/nllb-200-1.3B",
        "tokenizer_name": "facebook/nllb-200-1.3B",
        "task": "translation",
        "type": "transformers",
        "model_class": "AutoModelForSeq2SeqLM",
        "framework": "transformers",
        "is_primary": True,
        "allow_mps": True,
        "tokenizer_kwargs": {
            "src_lang": "eng_Latn",
            "tgt_lang": "spa_Latn"
        }
    }
    
    # Update MBART configuration
    if "mbart_translation" in registry:
        registry["mbart_translation"].update({
            "is_primary": False,
            "force_cpu": True,
            "model_class": "AutoModelForSeq2SeqLM"
        })
    
    # Ensure MT5 models use MT5ForConditionalGeneration class
    if "mt5_translation" in registry:
        registry["mt5_translation"]["model_class"] = "MT5ForConditionalGeneration"
    
    # Add text-to-speech model if missing
    if "tts" not in registry:
        registry["tts"] = {
            "model_name": "suno/bark-small",
            "task": "text-to-speech",
            "type": "transformers",
            "model_class": "AutoModelForTextToSpeech",
            "framework": "transformers",
            "tokenizer_kwargs": {
                "max_length": 256
            }
        }
    
    # Ensure language detection model uses correct class
    if "language_detection" in registry:
        registry["language_detection"]["model_class"] = "AutoModelForSequenceClassification"
    
    # Update other model classes to ensure proper typing
    for model_key in registry:
        if "model_class" not in registry[model_key] and registry[model_key].get("type") == "transformers":
            # Set default model class based on task
            task = registry[model_key].get("task", "")
            
            if "translation" in task:
                registry[model_key]["model_class"] = "AutoModelForSeq2SeqLM"
            elif task == "rag_generation":
                if "mt5" in registry[model_key].get("model_name", "").lower():
                    registry[model_key]["model_class"] = "MT5ForConditionalGeneration"
                else:
                    registry[model_key]["model_class"] = "AutoModelForSeq2SeqLM"
            elif task == "language_detection":
                registry[model_key]["model_class"] = "AutoModelForSequenceClassification"
            elif task == "ner_detection" or task == "anonymization":
                registry[model_key]["model_class"] = "AutoModelForTokenClassification"
    
    # Save updated registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Successfully updated model registry at {registry_path}")
    return registry

def create_nllb_language_mapping():
    """
    Create a helper module for mapping ISO language codes to NLLB-specific format.
    """
    print("📝 Creating NLLB language mapping module...")
    
    nllb_mapping_path = project_root / "app" / "core" / "pipeline" / "nllb_language_mapping.py"
    
    mapping_code = '''# NLLB language mapping module
# Maps ISO language codes to NLLB-specific format

# NLLB uses specific language codes in the format {lang_code}_{script}
# For example: eng_Latn, spa_Latn, fra_Latn, etc.

# This is a mapping of common ISO language codes to NLLB language codes
ISO_TO_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "ara_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "sw": "swh_Latn",
    "he": "heb_Hebr",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "da": "dan_Latn",
    "sv": "swe_Latn",
    "fi": "fin_Latn",
    "no": "nno_Latn",
    "hu": "hun_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "el": "ell_Grek",
    "bg": "bul_Cyrl",
    "uk": "ukr_Cyrl",
    "fa": "pes_Arab",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ur": "urd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ha": "hau_Latn",
    "yo": "yor_Latn",
    "zu": "zul_Latn",
    "ny": "nya_Latn",
    "so": "som_Latn",
    "am": "amh_Ethi",
    "ti": "tir_Ethi",
    "km": "khm_Khmr",
    "lo": "lao_Laoo"
}

# Reverse mapping from NLLB to ISO
NLLB_TO_ISO = {v: k for k, v in ISO_TO_NLLB.items()}

def get_nllb_code(iso_code):
    """Convert an ISO language code to NLLB format.
    
    Args:
        iso_code: ISO language code (e.g., 'en', 'es')
        
    Returns:
        NLLB language code (e.g., 'eng_Latn', 'spa_Latn')
    """
    # First, normalize the ISO code
    iso_normalized = iso_code.lower().split('-')[0].split('_')[0]
    
    # Return the NLLB code or default to English if not found
    return ISO_TO_NLLB.get(iso_normalized, "eng_Latn")

def get_iso_code(nllb_code):
    """Convert an NLLB language code to ISO format.
    
    Args:
        nllb_code: NLLB language code (e.g., 'eng_Latn', 'spa_Latn')
        
    Returns:
        ISO language code (e.g., 'en', 'es')
    """
    return NLLB_TO_ISO.get(nllb_code, "en")
'''
    
    # Write the mapping module
    with open(nllb_mapping_path, "w") as f:
        f.write(mapping_code)
    
    print(f"✅ Successfully created NLLB language mapping at {nllb_mapping_path}")

def update_translation_wrapper():
    """
    Update the translation wrapper to support NLLB's specific language codes.
    """
    print("🔄 Updating translation wrapper for NLLB support...")
    
    # Create a patch file for the translation wrapper
    patch_path = project_root / "scripts" / "fixes" / "nllb_wrapper_patch.py"
    
    patch_code = '''#!/usr/bin/env python3
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
'''
    
    # Create the fixes directory if it doesn't exist
    fixes_dir = project_root / "scripts" / "fixes"
    fixes_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the patch file
    with open(patch_path, "w") as f:
        f.write(patch_code)
    
    print(f"✅ Successfully created translation wrapper patch at {patch_path}")

def make_script_executable():
    """Make the NLLB wrapper patch script executable."""
    patch_path = project_root / "scripts" / "fixes" / "nllb_wrapper_patch.py"
    os.chmod(patch_path, 0o755)
    print(f"✅ Made script executable: {patch_path}")

def create_nllb_test_script():
    """Create a test script to verify NLLB translation works correctly."""
    print("📝 Creating NLLB test script...")
    
    test_script_path = project_root / "scripts" / "test_nllb_translation.py"
    
    test_script = '''#!/usr/bin/env python3
# Test NLLB translation functionality
# This script tests the NLLB model with different language pairs

import os
import sys
import time
import json
import asyncio
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import necessary modules
from app.utils.config import load_config
from app.services.models.loader import ModelLoader

# Explicit import of the NLLB language mapping
sys.path.append(str(project_root / "app" / "core" / "pipeline"))
from nllb_language_mapping import get_nllb_code

async def test_translation():
    """Test NLLB translation functionality."""
    print("🔍 Testing NLLB translation...")
    
    # Load configuration
    config = load_config()
    
    # Create model loader
    loader = ModelLoader(config=config)
    
    # Load registry configuration
    registry_path = project_root / "config" / "model_registry.json"
    with open(registry_path, "r") as f:
        registry = json.load(f)
    
    # Load NLLB model
    print("⏳ Loading NLLB translation model...")
    loader.model_config = registry
    translation_model = await loader.load_model("translation")
    
    if translation_model is None:
        print("❌ Failed to load NLLB translation model")
        return
    
    # Extract model and tokenizer
    model = translation_model.get("model")
    tokenizer = translation_model.get("tokenizer")
    
    if model is None or tokenizer is None:
        print("❌ Model or tokenizer is None")
        return
    
    print(f"✅ Successfully loaded NLLB model: {model.__class__.__name__}")
    print(f"✅ Device: {next(model.parameters()).device}")
    
    # Test translations
    test_translations = [
        {"text": "Hello, how are you?", "source": "en", "target": "es"},
        {"text": "This is a test of the NLLB translation model.", "source": "en", "target": "fr"},
        {"text": "I hope this works correctly.", "source": "en", "target": "de"},
        {"text": "The quick brown fox jumps over the lazy dog.", "source": "en", "target": "es"}
    ]
    
    for i, test in enumerate(test_translations):
        text = test["text"]
        source = test["source"]
        target = test["target"]
        
        # Convert to NLLB format
        source_nllb = get_nllb_code(source)
        target_nllb = get_nllb_code(target)
        
        print(f"\\n🔄 Test {i+1}: Translating from {source} ({source_nllb}) to {target} ({target_nllb})\\n")
        print(f"Input: {text}")
        
        # Set source language
        tokenizer.src_lang = source_nllb
        
        # Prepare the input
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        
        # Force target language
        forced_bos_token_id = tokenizer.lang_code_to_id[target_nllb]
        
        # Generate translation
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )
        translation_time = time.time() - start_time
        
        # Decode translation
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(f"Output: {translation}")
        print(f"Translation completed in {translation_time:.3f} seconds")

if __name__ == "__main__":
    asyncio.run(test_translation())
'''
    
    # Write the test script
    with open(test_script_path, "w") as f:
        f.write(test_script)
    
    # Make the script executable
    os.chmod(test_script_path, 0o755)
    
    print(f"✅ Successfully created NLLB test script at {test_script_path}")

def main():
    """Main function to fix model loading issues."""
    print("🔧 Fixing model loading issues for NLLB translation...")
    
    # Update model registry
    registry = fix_model_registry()
    
    # Create NLLB language mapping
    create_nllb_language_mapping()
    
    # Update translation wrapper
    update_translation_wrapper()
    
    # Make script executable
    make_script_executable()
    
    # Create test script
    create_nllb_test_script()
    
    print("✅ All fixes have been applied. Please restart the server to apply changes.")
    print()
    print("To test NLLB translation, run:")
    print("python scripts/test_nllb_translation.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())