#!/usr/bin/env python3
"""
Script to update the model registry to use NLLB-200-1.3B as the primary translation model
and fix the device selection logic to allow NLLB models to run on MPS.

Usage:
    python update_to_nllb_13b.py
"""

import os
import json
import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nllb_update")

# Paths
CONFIG_DIR = Path("config")
MODEL_REGISTRY_PATH = CONFIG_DIR / "model_registry.json"
BACKUP_REGISTRY_PATH = CONFIG_DIR / "model_registry.json.bak"

def update_model_registry():
    """Update the model registry to use NLLB-200-1.3B for translation."""
    logger.info("Starting update of model registry to use NLLB-200-1.3B")

    # Create backup of current registry
    if MODEL_REGISTRY_PATH.exists():
        shutil.copyfile(MODEL_REGISTRY_PATH, BACKUP_REGISTRY_PATH)
        logger.info(f"Created backup at {BACKUP_REGISTRY_PATH}")

    # Load current registry
    if MODEL_REGISTRY_PATH.exists():
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
        logger.info(f"Loaded existing model registry with {len(registry)} entries")
    else:
        registry = {}
        logger.warning("No existing model registry found. Creating new registry.")

    # Update translation entry to use NLLB-200-1.3B
    registry["translation"] = {
        "model_name": "facebook/nllb-200-1.3B",
        "tokenizer_name": "facebook/nllb-200-1.3B",
        "task": "translation",
        "type": "transformers",
        "model_class": "AutoModelForSeq2SeqLM",
        "framework": "transformers",
        "is_primary": True,
        "memory_required": 8,
        "gpu_memory_required": 4,
        "requires_gpu": False,
        "language_codes": {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ru": "rus_Cyrl",
            "ar": "arb_Arab",
            "pt": "por_Latn"
        }
    }

    # Keep MT5 model as a fallback
    registry["mt5_translation"] = {
        "model_name": "google/mt5-small",
        "tokenizer_name": "google/mt5-small",
        "task": "mt5_translation",
        "type": "transformers",
        "model_class": "AutoModelForSeq2SeqLM",
        "framework": "transformers",
        "is_fallback": True,
        "tokenizer_kwargs": {
            "model_max_length": 512
        }
    }

    # Save the updated registry
    with open(MODEL_REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Updated model registry saved to {MODEL_REGISTRY_PATH}")
    logger.info("Primary translation model is now NLLB-200-1.3B")

def create_nllb_fix_script():
    """Create a script to fix the NLLB MPS compatibility issue in loader.py and wrapper.py"""
    logger.info("Creating NLLB fix script for device compatibility")
    
    fix_script_path = Path("fix_nllb_mps_compatibility.py")
    script_content = """#!/usr/bin/env python3
\"\"\"
Script to fix the NLLB model compatibility with MPS devices.
This script modifies the device selection logic to allow NLLB models
to run on MPS when available.

Usage:
    python fix_nllb_mps_compatibility.py
\"\"\"

import re
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nllb_mps_fix")

# Paths
LOADER_PATH = Path("app/services/models/loader.py")
WRAPPER_PATH = Path("app/services/models/wrapper.py")
BACKUP_SUFFIX = ".bak_before_nllb_fix"

def backup_file(file_path, backup_suffix=BACKUP_SUFFIX):
    """Create a backup of the file"""
    import shutil
    backup_path = file_path.with_suffix(backup_suffix)
    shutil.copyfile(file_path, backup_path)
    logger.info(f"Created backup of {file_path} at {backup_path}")
    return backup_path

def fix_loader_device_selection():
    """Fix the _determine_device method in ModelLoader to allow NLLB models on MPS"""
    if not LOADER_PATH.exists():
        logger.error(f"{LOADER_PATH} not found. Skipping loader fix.")
        return False
    
    backup_file(LOADER_PATH)
    
    with open(LOADER_PATH, 'r') as f:
        content = f.read()
    
    # Find the MBART detection pattern in _determine_device method
    pattern = r"(if \"mps\" in self\.available_devices:.*?# First check model_type-based detection.*?if model_type.*?\\[\\s\\n\\r]*\"mbart\", \"translation\", \"multilingual\"\\s*\\].*?model_name = self\.registry\\[model_type\\]\.model_name\.lower\\(\\).*?if any\\(mbart_id in model_name for mbart_id in \\[.*?\"mbart\", \"facebook/mbart\", \"multilingual-translation\", \"nllb\"\\s*\\]\\).*?is_mbart_model = True)"
    
    new_code = (
        'if "mps" in self.available_devices:  # Only apply on MPS devices\n'
        '            # First check model_type-based detection\n'
        '            is_mbart_model = False\n'
        '            \n'
        '            # Check the model type name for MBART indicators\n'
        '            if model_type and any(mbart_marker in model_type.lower() for mbart_marker in [\n'
        '                "mbart", "translation", "multilingual"\n'
        '            ]):\n'
        '                # Check if NLLB in model name\n'
        '                if model_type in self.registry:\n'
        '                    model_name = self.registry[model_type].model_name.lower()\n'
        '                    # Special handling for NLLB models vs. MBART models\n'
        '                    if any(mbart_id in model_name for mbart_id in [\n'
        '                        "mbart", "facebook/mbart", "multilingual-translation"\n'
        '                    ]):\n'
        '                        # Force CPU only for MBART models\n'
        '                        logger.warning(f"⚠️ Forcing CPU device for MBART model {model_type} due to known MPS compatibility issues")\n'
        '                        is_mbart_model = True\n'
        '                        return "cpu"\n'
        '                    elif "nllb" in model_name:\n'
        '                        # Allow NLLB to use MPS\n'
        '                        logger.info(f"✓ Allowing NLLB model {model_type} to use MPS device")\n'
        '                        # Continue with normal device selection\n'
    )
    
    if re.search(pattern, content, re.DOTALL):
        # Replace the pattern with our new code
        modified_content = re.sub(pattern, new_code, content, flags=re.DOTALL)
        with open(LOADER_PATH, 'w') as f:
            f.write(modified_content)
        logger.info(f"Updated device selection logic in {LOADER_PATH}")
        return True
    else:
        logger.error(f"Could not find the pattern to replace in {LOADER_PATH}")
        return False

def fix_wrapper_device_selection():
    """Fix the device selection in BaseModelWrapper to allow NLLB models on MPS"""
    if not WRAPPER_PATH.exists():
        logger.error(f"{WRAPPER_PATH} not found. Skipping wrapper fix.")
        return False
    
    backup_file(WRAPPER_PATH)
    
    with open(WRAPPER_PATH, 'r') as f:
        content = f.read()
    
    # Find the MBART detection pattern in BaseModelWrapper.__init__
    pattern = r"(# Enhanced MBART detection.*?if device == \"mps\":.*?is_mbart_model = False.*?# Check model config attribute for MBART identifiers.*?if hasattr\\(model, \"config\"\\):.*?if any\\(mbart_id in model_name for mbart_id in \\[\"mbart\", \"facebook/mbart\", \"nllb\"\\]\\):.*?is_mbart_model = True)"
    
    new_code = (
        '        # Enhanced device selection for MPS compatibility\n'
        '        if device == "mps":\n'
        '            # 1. Check if this is an MBART model using the model configuration\n'
        '            is_mbart_model = False\n'
        '            model_name = ""\n'
        '            \n'
        '            # Check model config attribute for model identifiers\n'
        '            if hasattr(model, "config"):\n'
        '                if hasattr(model.config, "_name_or_path"):\n'
        '                    model_name = model.config._name_or_path.lower()\n'
        '                elif hasattr(model.config, "name_or_path"):\n'
        '                    model_name = model.config.name_or_path.lower()\n'
        '                elif hasattr(model.config, "model_type"):\n'
        '                    model_name = model.config.model_type.lower()\n'
        '                    \n'
        '                # Check for MBART indicators in the name\n'
        '                if any(mbart_id in model_name for mbart_id in ["mbart", "facebook/mbart"]):\n'
        '                    is_mbart_model = True\n'
        '                    \n'
        '                # Special handling for NLLB models - allow them to use MPS\n'
        '                if "nllb" in model_name:\n'
        '                    # This is an NLLB model, which can run on MPS\n'
        '                    is_mbart_model = False'
    )
    
    if re.search(pattern, content, re.DOTALL):
        # Replace the pattern with our new code
        modified_content = re.sub(pattern, new_code, content, flags=re.DOTALL)
        with open(WRAPPER_PATH, 'w') as f:
            f.write(modified_content)
        logger.info(f"Updated device selection logic in {WRAPPER_PATH}")
        return True
    else:
        logger.error(f"Could not find the pattern to replace in {WRAPPER_PATH}")
        return False

if __name__ == "__main__":
    logger.info("Starting NLLB-MPS compatibility fix...")
    fix_loader_device_selection()
    fix_wrapper_device_selection()
    logger.info("NLLB-MPS compatibility fix completed")
"""
    
    with open(fix_script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(fix_script_path, 0o755)
    
    logger.info(f"Created NLLB MPS compatibility fix script at {fix_script_path}")

def create_test_script():
    """Create a test script to verify NLLB model works with Spanish-English translations"""
    logger.info("Creating NLLB test script")
    
    test_script_path = Path("test_nllb_translation.py")
    script_content = """#!/usr/bin/env python3
\"\"\"
Test script to verify NLLB-200-1.3B model works with translations,
especially Spanish to English on MPS-enabled devices.

Usage:
    python test_nllb_translation.py
\"\"\"

import argparse
import json
import logging
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nllb_test")

# Test texts for different language pairs
TEST_TEXTS = {
    "es-en": [
        "Estoy muy feliz de conocerte hoy.",
        "El cielo es azul y las nubes son blancas.",
        "La tecnología moderna ha cambiado nuestra forma de vivir.",
    ],
    "en-es": [
        "I am very happy to meet you today.",
        "The sky is blue and the clouds are white.",
        "Modern technology has changed our way of life.",
    ],
    "en-fr": [
        "I am very happy to meet you today.",
        "The sky is blue and the clouds are white.",
    ],
    "fr-en": [
        "Je suis très heureux de vous rencontrer aujourd'hui.",
        "Le ciel est bleu et les nuages sont blancs.",
    ]
}

def test_nllb_direct():
    """Test NLLB model directly using transformers library"""
    logger.info("Testing NLLB model directly with transformers")
    
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        # Load NLLB model and tokenizer
        model_name = "facebook/nllb-200-1.3B"
        logger.info(f"Loading NLLB model: {model_name}")
        
        # Determine available device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)
        
        # Language codes for NLLB
        lang_code_mapping = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ru": "rus_Cyrl",
            "ar": "arb_Arab",
            "pt": "por_Latn"
        }
        
        # Test Spanish to English translation (the problematic case)
        test_es_to_en(tokenizer, model, device, lang_code_mapping)
        
        # Test English to Spanish translation
        test_en_to_es(tokenizer, model, device, lang_code_mapping)
        
        # Optional: Test other language pairs
        test_other_pairs(tokenizer, model, device, lang_code_mapping)
        
        logger.info("NLLB model tests completed")
        
    except Exception as e:
        logger.error(f"Error in direct NLLB test: {e}", exc_info=True)
        return False
        
    return True

def test_es_to_en(tokenizer, model, device, lang_code_mapping):
    """Test Spanish to English translation with NLLB"""
    logger.info("Testing Spanish to English translation")
    
    source_lang = "es"
    target_lang = "en"
    source_lang_code = lang_code_mapping[source_lang]
    target_lang_code = lang_code_mapping[target_lang]
    
    # Process each test text
    for text in TEST_TEXTS["es-en"]:
        logger.info(f"Translating ES->EN: {text}")
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forced BOS token ID for target language
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(f"Translation result: {translation}")
        
    logger.info("Spanish to English translation test completed")

def test_en_to_es(tokenizer, model, device, lang_code_mapping):
    """Test English to Spanish translation with NLLB"""
    logger.info("Testing English to Spanish translation")
    
    source_lang = "en"
    target_lang = "es"
    source_lang_code = lang_code_mapping[source_lang]
    target_lang_code = lang_code_mapping[target_lang]
    
    # Process each test text
    for text in TEST_TEXTS["en-es"]:
        logger.info(f"Translating EN->ES: {text}")
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forced BOS token ID for target language
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(f"Translation result: {translation}")
        
    logger.info("English to Spanish translation test completed")

def test_other_pairs(tokenizer, model, device, lang_code_mapping):
    """Test other language pairs with NLLB"""
    logger.info("Testing other language pairs")
    
    for lang_pair in ["en-fr", "fr-en"]:
        source_lang, target_lang = lang_pair.split("-")
        source_lang_code = lang_code_mapping[source_lang]
        target_lang_code = lang_code_mapping[target_lang]
        
        # Process each test text
        for text in TEST_TEXTS[lang_pair]:
            logger.info(f"Translating {source_lang.upper()}->{target_lang.upper()}: {text}")
            
            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forced BOS token ID for target language
            forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode the generated tokens
            translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            logger.info(f"Translation result: {translation}")
    
    logger.info("Other language pairs test completed")

def test_api_translation():
    """Test NLLB model through the API endpoint"""
    logger.info("Testing NLLB model through API endpoint")
    
    # Check if the server is running
    try:
        import requests
        
        # Server URL (adjust if needed)
        base_url = "http://localhost:8000"
        
        # Test Spanish to English translation
        es_to_en_payload = {
            "text": "Estoy muy feliz de conocerte hoy.",
            "source_language": "es",
            "target_language": "en"
        }
        
        # Send request to translation endpoint
        response = requests.post(f"{base_url}/api/translate", json=es_to_en_payload)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"API Translation result: {result}")
            logger.info("API translation test passed")
        else:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            logger.info("Note: You may need to restart the server to load the new model")
    
    except Exception as e:
        logger.error(f"Error in API test: {e}")
        logger.info("API test skipped. Make sure the server is running.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NLLB-200-1.3B model for translations")
    parser.add_argument("--mode", choices=["direct", "api", "all"], default="all",
                      help="Test mode: 'direct' for direct model testing, 'api' for API testing, 'all' for both")
    
    args = parser.parse_args()
    
    logger.info("Starting NLLB translation tests")
    
    if args.mode in ["direct", "all"]:
        test_nllb_direct()
    
    if args.mode in ["api", "all"]:
        test_api_translation()
    
    logger.info("All tests completed")
"""
    
    with open(test_script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(test_script_path, 0o755)
    
    logger.info(f"Created NLLB test script at {test_script_path}")

def update_language_code_handling():
    """Create a script to update language code handling for NLLB models"""
    logger.info("Creating language code handling script for NLLB models")
    
    script_path = Path("update_nllb_language_codes.py")
    script_content = """#!/usr/bin/env python3
\"\"\"
Script to update language code handling for NLLB models.
This ensures that all language codes used with NLLB are in the correct format.

Usage:
    python update_nllb_language_codes.py
\"\"\"

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nllb_lang_codes")

# Paths
TOKENIZER_PATH = Path("app/core/pipeline/tokenizer.py")
BACKUP_SUFFIX = ".bak_before_nllb_update"

def backup_file(file_path, backup_suffix=BACKUP_SUFFIX):
    """Create a backup of the file"""
    import shutil
    backup_path = file_path.with_suffix(backup_suffix)
    shutil.copyfile(file_path, backup_path)
    logger.info(f"Created backup of {file_path} at {backup_path}")
    return backup_path

def update_language_codes():
    """Update the language code mapping in tokenizer.py to include more languages for NLLB"""
    if not TOKENIZER_PATH.exists():
        logger.error(f"{TOKENIZER_PATH} not found. Skipping language code update.")
        return False
    
    backup_file(TOKENIZER_PATH)
    
    with open(TOKENIZER_PATH, 'r') as f:
        content = f.read()
    
    # Find the existing language code mapping
    import re
    pattern = r"LANG_CODE_MAPPING = \\{[^\\}]*\\}"
    
    # Define expanded language code mapping
    new_mapping = '''LANG_CODE_MAPPING = {
    # Essential languages with their NLLB language codes
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    
    # Additional languages
    "nl": "nld_Latn",
    "ko": "kor_Hang",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "cs": "ces_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ro": "ron_Latn",
    "bn": "ben_Beng",
    
    # Additional Chinese variants
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh-hk": "yue_Hant",  # Cantonese
}'''
    
    if re.search(pattern, content):
        # Replace the pattern with our new mapping
        modified_content = re.sub(pattern, new_mapping, content)
        
        # Also update the prepare_translation_inputs method to better handle NLLB
        prepare_pattern = r"def prepare_translation_inputs.*?if \"nllb\" in self\.model_name:.*?return \\{.*?\\}"
        
        nllb_update = '''def prepare_translation_inputs(self, text: str, source_lang: str, target_lang: str) -> dict:
        """Prepare inputs for translation with proper language codes."""
        source_code = LANG_CODE_MAPPING.get(source_lang, source_lang)
        target_code = LANG_CODE_MAPPING.get(target_lang, target_lang)

        if "nllb" in self.model_name:
            # NLLB-specific tokenization
            if hasattr(self.tokenizer, "src_lang"):
                self.tokenizer.src_lang = source_code
            model_inputs = self.tokenizer(text, return_tensors="pt")
            
            # Get forced_bos_token_id for target language
            try:
                forced_bos_id = self.tokenizer.lang_code_to_id[target_code]
            except (KeyError, AttributeError) as e:
                logger.warning(f"Could not find language code '{target_code}' in tokenizer: {e}")
                # Default to English if target code not found
                target_code = "eng_Latn"
                forced_bos_id = self.tokenizer.lang_code_to_id.get(target_code, None)
            
            return {
                "inputs": model_inputs,
                "forced_bos_token_id": forced_bos_id,
                "source_lang": source_code,
                "target_lang": target_code
            }'''
        
        if re.search(prepare_pattern, modified_content, re.DOTALL):
            modified_content = re.sub(prepare_pattern, nllb_update, modified_content, flags=re.DOTALL)
        
        with open(TOKENIZER_PATH, 'w') as f:
            f.write(modified_content)
        
        logger.info(f"Updated language code mapping in {TOKENIZER_PATH}")
        return True
    else:
        logger.error(f"Could not find the language code mapping pattern in {TOKENIZER_PATH}")
        return False

if __name__ == "__main__":
    logger.info("Starting NLLB language code update...")
    update_language_codes()
    logger.info("NLLB language code update completed")
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"Created NLLB language code update script at {script_path}")

if __name__ == "__main__":
    try:
        # Update model registry
        update_model_registry()
        
        # Create fix scripts
        create_nllb_fix_script()
        update_language_code_handling()
        create_test_script()
        
        print("\n✅ NLLB-200-1.3B update completed successfully.")
        print("\nNext steps:")
        print("1. Run 'python fix_nllb_mps_compatibility.py' to fix device selection")
        print("2. Run 'python update_nllb_language_codes.py' to update language code handling")
        print("3. Restart the server to use the new model")
        print("4. Run 'python test_nllb_translation.py' to test the translation capabilities")
        
    except Exception as e:
        logger.error(f"Error updating to NLLB-1.3B: {e}", exc_info=True)
        print(f"\n❌ Error updating to NLLB-1.3B: {e}")
        raise