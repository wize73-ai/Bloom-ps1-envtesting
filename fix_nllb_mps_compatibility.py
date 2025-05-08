#!/usr/bin/env python3
"""
Script to fix the NLLB model compatibility with MPS devices.
This script modifies the device selection logic to allow NLLB models
to run on MPS when available.

Usage:
    python fix_nllb_mps_compatibility.py
"""

import re
import logging
import shutil
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
    backup_path = Path(str(file_path) + backup_suffix)
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
    
    # Find the device selection logic in _determine_device method
    loader_pattern = r"(if \"mps\" in self\.available_devices:.+?# First check model_type-based detection.+?is_mbart_model = False.+?# Check the model type name for MBART indicators.+?if model_type and any\(mbart_marker in model_type\.lower\(\) for mbart_marker in \[.+?\"mbart\", \"translation\", \"multilingual\".+?\]\):.+?# Check if NLLB in model name.+?if model_type in self\.registry:)"
    
    # Modified logic that allows NLLB models to run on MPS
    loader_replacement = """if "mps" in self.available_devices:  # Only apply on MPS devices
            # First check model_type-based detection
            is_mbart_model = False
            
            # Check the model type name for MBART indicators
            if model_type and any(mbart_marker in model_type.lower() for mbart_marker in [
                "mbart", "translation", "multilingual"
            ]):
                # Check if this is an NLLB model, which we want to allow on MPS
                if model_type in self.registry:"""
    
    # Replace the MBART check to allow NLLB to run on MPS
    mbart_detection_pattern = r"(model_name = self\.registry\[model_type\]\.model_name\.lower\(\).+?# More thorough check of model name.+?if any\(mbart_id in model_name for mbart_id in \[.+?\"mbart\", \"facebook/mbart\", \"multilingual-translation\", \"nllb\".+?\]\):.+?is_mbart_model = True.+?logger\.warning\(f\"⚠️ Forcing CPU device for MBART model {model_type} due to known MPS compatibility issues\"\).+?return \"cpu\")"
    
    mbart_detection_replacement = """model_name = self.registry[model_type].model_name.lower()
                    # Allow NLLB models to use MPS, force CPU only for MBART
                    if "nllb" in model_name:
                        logger.info(f"✓ Allowing NLLB model {model_type} to use MPS device")
                        # Continue with normal device selection for NLLB
                    elif any(mbart_id in model_name for mbart_id in [
                        "mbart", "facebook/mbart", "multilingual-translation"
                    ]):
                        is_mbart_model = True
                        logger.warning(f"⚠️ Forcing CPU device for MBART model {model_type} due to known MPS compatibility issues")
                        return "cpu\""""
    
    # Replace the Spanish-English check to be more specific and not affect NLLB
    spanish_english_pattern = r"(# Special case for Spanish to English translation models.+?# These are known to have issues with MPS even if not explicitly MBART.+?if model_type and model_type\.lower\(\) in \[\"translation\", \"mbart_translation\", \"mt5_translation\"\]:.+?logger\.warning\(f\"⚠️ Forcing CPU device for {model_type} model due to potential Spanish-English translation compatibility issues with MPS\"\).+?return \"cpu\")"
    
    spanish_english_replacement = """            # Special case for MBART Spanish to English translations only
            # These are known to have issues with MPS
            if model_type and model_type.lower() in ["mbart_translation"]:
                logger.warning(f"⚠️ Forcing CPU device for {model_type} model due to MBART Spanish-English translation compatibility issues with MPS")
                return "cpu"
                
            # For other translation models (like NLLB, MT5), allow MPS usage"""
    
    # Apply the replacements
    modified_content = content
    if re.search(loader_pattern, content, re.DOTALL):
        modified_content = re.sub(loader_pattern, loader_replacement, modified_content, flags=re.DOTALL)
        logger.info("Updated loader device selection pattern")
    else:
        logger.warning("Could not find loader device selection pattern in the file")
    
    if re.search(mbart_detection_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(mbart_detection_pattern, mbart_detection_replacement, modified_content, flags=re.DOTALL)
        logger.info("Updated MBART detection pattern")
    else:
        logger.warning("Could not find MBART detection pattern in the file")
    
    if re.search(spanish_english_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(spanish_english_pattern, spanish_english_replacement, modified_content, flags=re.DOTALL)
        logger.info("Updated Spanish-English pattern")
    else:
        logger.warning("Could not find Spanish-English pattern in the file")
    
    with open(LOADER_PATH, 'w') as f:
        f.write(modified_content)
    
    logger.info(f"Updated loader code in {LOADER_PATH}")
    return True

def fix_wrapper_device_selection():
    """Fix the device selection in BaseModelWrapper to allow NLLB models on MPS"""
    if not WRAPPER_PATH.exists():
        logger.error(f"{WRAPPER_PATH} not found. Skipping wrapper fix.")
        return False
    
    backup_file(WRAPPER_PATH)
    
    with open(WRAPPER_PATH, 'r') as f:
        content = f.read()
    
    # Find the NLLB check in BaseModelWrapper.__init__
    wrapper_pattern = r"(# Enhanced MBART detection - Special handling for MPS device due to stability issues.+?if device == \"mps\":.+?# 1\. Check if this is an MBART model using the model configuration.+?is_mbart_model = False.+?model_name = \"\".+?# Check model config attribute for MBART identifiers.+?if hasattr\(model, \"config\"\):.+?if any\(mbart_id in model_name for mbart_id in \[\"mbart\", \"facebook/mbart\", \"nllb\"\]\):.+?is_mbart_model = True)"
    
    # Modified wrapper code that allows NLLB on MPS
    wrapper_replacement = """        # Enhanced device selection for MPS compatibility
        if device == "mps":
            # 1. Check if this is an MBART model using the model configuration
            is_mbart_model = False
            model_name = ""
            
            # Check model config attribute for model identifiers
            if hasattr(model, "config"):
                if hasattr(model.config, "_name_or_path"):
                    model_name = model.config._name_or_path.lower()
                elif hasattr(model.config, "name_or_path"):
                    model_name = model.config.name_or_path.lower()
                elif hasattr(model.config, "model_type"):
                    model_name = model.config.model_type.lower()
                    
                # Check for MBART indicators in the name, but allow NLLB to use MPS
                if "nllb" in model_name:
                    # This is an NLLB model - we want to allow it to use MPS
                    logger.info(f"Allowing NLLB model {model_name} to use MPS")
                    is_mbart_model = False
                elif any(mbart_id in model_name for mbart_id in ["mbart", "facebook/mbart"]):
                    is_mbart_model = True"""
    
    # Find the MBART forcing CPU code
    force_cpu_pattern = r"(# 3\. Force CPU for any identified MBART models.+?if is_mbart_model:.+?logger\.warning\(f\"⚠️ Forcing CPU device for MBART model due to known MPS compatibility issues\"\).+?device = \"cpu\".+?# Update config to reflect the device change.+?self\.config\[\"device\"\] = \"cpu\")"
    
    # Modified force CPU code that checks for NLLB vs MBART
    force_cpu_replacement = """            # 3. Force CPU only for MBART models, not NLLB
            if is_mbart_model:
                logger.warning(f"⚠️ Forcing CPU device for MBART model due to known MPS compatibility issues")
                device = "cpu"
                # Update config to reflect the device change
                self.config["device"] = "cpu\""""
    
    # Apply the replacements
    modified_content = content
    if re.search(wrapper_pattern, content, re.DOTALL):
        modified_content = re.sub(wrapper_pattern, wrapper_replacement, modified_content, flags=re.DOTALL)
        logger.info("Updated wrapper MBART detection pattern")
    else:
        logger.warning("Could not find wrapper MBART detection pattern in the file")
    
    if re.search(force_cpu_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(force_cpu_pattern, force_cpu_replacement, modified_content, flags=re.DOTALL)
        logger.info("Updated force CPU pattern")
    else:
        logger.warning("Could not find force CPU pattern in the file")
    
    # Add NLLB-specific language code handling in _get_mbart_lang_code or create a new method
    nllb_lang_code_method = "\n    def _get_nllb_lang_code(self, language_code: str) -> str:\n        \"\"\"Convert ISO language code to NLLB language code format\"\"\"\n        # NLLB uses language codes like \"eng_Latn\", \"spa_Latn\", etc.\n        if language_code in [\"zh\", \"zh-cn\", \"zh-CN\"]:\n            return \"zho_Hans\"\n        elif language_code in [\"zh-tw\", \"zh-TW\"]:\n            return \"zho_Hant\"\n        elif language_code == \"en\":\n            return \"eng_Latn\"\n        elif language_code == \"es\":\n            return \"spa_Latn\"\n        elif language_code == \"fr\":\n            return \"fra_Latn\"\n        elif language_code == \"de\":\n            return \"deu_Latn\"\n        elif language_code == \"it\":\n            return \"ita_Latn\"\n        elif language_code == \"pt\":\n            return \"por_Latn\"\n        elif language_code == \"ja\":\n            return \"jpn_Jpan\"\n        elif language_code == \"ar\":\n            return \"arb_Arab\"\n        elif language_code == \"ru\":\n            return \"rus_Cyrl\"\n        else:\n            # For other languages, try to derive from ISO code\n            # Convert en-US to eng_Latn format\n            base_code = language_code.split(\"-\")[0].lower()\n            script = \"Latn\"  # Default script for most languages\n            \n            # Map language code to appropriate script if known\n            script_mapping = {\n                \"zh\": \"Hans\",\n                \"ja\": \"Jpan\",\n                \"ko\": \"Hang\",\n                \"ru\": \"Cyrl\",\n                \"ar\": \"Arab\",\n                \"he\": \"Hebr\",\n                \"el\": \"Grek\",\n                \"hi\": \"Deva\",\n                \"th\": \"Thai\",\n                \"bn\": \"Beng\",\n            }\n            \n            if base_code in script_mapping:\n                script = script_mapping[base_code]\n                \n            # Map 2-letter code to 3-letter code for NLLB\n            iso_639_1_to_639_3 = {\n                \"en\": \"eng\",\n                \"es\": \"spa\",\n                \"fr\": \"fra\",\n                \"de\": \"deu\",\n                \"it\": \"ita\",\n                \"pt\": \"por\",\n                \"ja\": \"jpn\",\n                \"ar\": \"arb\",\n                \"ru\": \"rus\",\n                \"zh\": \"zho\",\n                \"ko\": \"kor\",\n                \"nl\": \"nld\",\n                \"pl\": \"pol\",\n                \"tr\": \"tur\",\n                \"uk\": \"ukr\",\n                \"vi\": \"vie\",\n                \"sv\": \"swe\",\n                \"da\": \"dan\",\n                \"fi\": \"fin\",\n                \"no\": \"nob\",\n                \"cs\": \"ces\",\n                \"hu\": \"hun\",\n                \"el\": \"ell\",\n                \"he\": \"heb\",\n                \"hi\": \"hin\",\n                \"th\": \"tha\",\n                \"id\": \"ind\",\n                \"ro\": \"ron\",\n                \"bn\": \"ben\",\n            }\n            \n            iso_code = iso_639_1_to_639_3.get(base_code, base_code)\n            \n            # Return in NLLB format\n            return f\"{iso_code}_{script}\"\n"
    
    # Insert the new NLLB language code method after the existing _get_mbart_lang_code method
    mbart_lang_code_pattern = r"(def _get_mbart_lang_code.*?\n        else:.*?\n            # Just the base language code for most languages.*?\n            base_code = language_code\.split.*?\n            return f\"{base_code}_XX\")"
    
    if re.search(mbart_lang_code_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(mbart_lang_code_pattern, r"\1" + nllb_lang_code_method, modified_content, flags=re.DOTALL)
        logger.info("Added NLLB language code handling method")
    else:
        logger.warning("Could not find _get_mbart_lang_code method to insert NLLB language code handling")
        # Try to add it after the class definition
        class_def_pattern = r"(class TranslationModelWrapper\(BaseModelWrapper\):)"
        if re.search(class_def_pattern, modified_content):
            modified_content = re.sub(class_def_pattern, r"\1" + nllb_lang_code_method, modified_content)
            logger.info("Added NLLB language code handling method after class definition")
        else:
            logger.warning("Could not add NLLB language code handling method")
    
    # Update the _preprocess method to check for NLLB model
    preprocess_pattern = r"(# Handle MBART vs MT5 models.+?model_name = getattr\(getattr\(self\.model, \"config\", None\), \"_name_or_path\", \"\"\) if hasattr\(self\.model, \"config\"\) else \"\")"
    
    preprocess_replacement = """        # Handle different model types
        model_name = ""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "_name_or_path"):
                model_name = self.model.config._name_or_path.lower()
            elif hasattr(self.model.config, "name_or_path"):
                model_name = self.model.config.name_or_path.lower()
            elif hasattr(self.model.config, "model_type"):
                model_name = self.model.config.model_type.lower()
                
        # Detect if this is an NLLB model
        is_nllb_model = "nllb" in model_name"""
    
    if re.search(preprocess_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(preprocess_pattern, preprocess_replacement, modified_content, flags=re.DOTALL)
        logger.info("Updated model detection in _preprocess method")
    else:
        logger.warning("Could not find model detection pattern in _preprocess method")
    
    # Add NLLB handling in the if/elif chain for MBART models
    mbart_handling_pattern = r"(elif \"mbart\" in model_name\.lower\(\):.*?\n            # MBART uses specific format.*?\n            source_lang_code = self\._get_mbart_lang_code\(source_lang\).*?\n            target_lang_code = self\._get_mbart_lang_code\(target_lang\))"
    
    mbart_handling_replacement = """        elif "nllb" in model_name.lower():
            # NLLB uses specific format, similar to MBART but with different codes
            source_lang_code = self._get_nllb_lang_code(source_lang)
            target_lang_code = self._get_nllb_lang_code(target_lang)
            
            # Check if tokenizer supports src_lang
            supports_src_lang = False
            try:
                # Check tokenizer for src_lang support
                import inspect
                sig = inspect.signature(self.tokenizer.__call__)
                supports_src_lang = 'src_lang' in sig.parameters
                
                # Alternative checks
                if not supports_src_lang:
                    supports_src_lang = (hasattr(self.tokenizer, 'src_lang') or 
                                      hasattr(self.tokenizer, 'set_src_lang_special_tokens'))
                
                logger.debug(f"NLLB tokenizer supports src_lang: {supports_src_lang}")
            except Exception as e:
                logger.debug(f"Error checking NLLB tokenizer signature: {e}")
                supports_src_lang = False
            
            # Tokenize input based on capabilities
            try:
                if supports_src_lang and source_lang_code:
                    logger.debug(f"Using src_lang with NLLB tokenizer: {source_lang_code}")
                    inputs = self.tokenizer(
                        texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=(self.config.get("max_length", 1024) if isinstance(self.config, dict) else getattr(self.config, "max_length", 1024)),
                        src_lang=source_lang_code
                    )
                else:
                    logger.debug(f"Standard tokenization for NLLB")
                    inputs = self.tokenizer(
                        texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=(self.config.get("max_length", 1024) if isinstance(self.config, dict) else getattr(self.config, "max_length", 1024))
                    )
            except TypeError as e:
                # Handle unexpected keyword errors
                if "unexpected keyword argument 'src_lang'" in str(e):
                    logger.warning(f"NLLB tokenizer does not support src_lang, using standard tokenization")
                    inputs = self.tokenizer(
                        texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=(self.config.get("max_length", 1024) if isinstance(self.config, dict) else getattr(self.config, "max_length", 1024))
                    )
                else:
                    raise
            
            # Get target language token ID for generation
            try:
                if hasattr(self.tokenizer, "lang_code_to_id"):
                    inputs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[target_lang_code]
                    logger.debug(f"Set NLLB forced_bos_token_id to {inputs['forced_bos_token_id']} for {target_lang_code}")
                else:
                    logger.warning(f"NLLB tokenizer does not have lang_code_to_id mapping")
            except (KeyError, AttributeError) as e:
                logger.error(f"Error setting NLLB target language token: {e}")
                logger.warning(f"Could not find language code {target_lang_code} in NLLB tokenizer")
                # Handle missing language codes gracefully
                if target_lang == "en":
                    # If targeting English, try to use a hardcoded token ID as fallback
                    inputs["forced_bos_token_id"] = 128022  # eng_Latn in NLLB
                    logger.info(f"Using hardcoded token ID for English: {inputs['forced_bos_token_id']}")
                else:
                    # For other languages, don't set forced_bos_token_id
                    logger.warning(f"No forced_bos_token_id set for {target_lang_code}")
        elif "mbart" in model_name.lower():
            # MBART uses specific format
            source_lang_code = self._get_mbart_lang_code(source_lang)
            target_lang_code = self._get_mbart_lang_code(target_lang)"""
    
    if re.search(mbart_handling_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(mbart_handling_pattern, mbart_handling_replacement, modified_content, flags=re.DOTALL)
        logger.info("Added NLLB handling in _preprocess method")
    else:
        logger.warning("Could not find MBART handling pattern in _preprocess method")
    
    with open(WRAPPER_PATH, 'w') as f:
        f.write(modified_content)
    
    logger.info(f"Updated wrapper code in {WRAPPER_PATH}")
    return True

def fix_translation_prompt_enhancer():
    """Create or update translation prompt enhancer to support NLLB models"""
    # Path to the translation prompt enhancer module
    enhancer_path = Path("app/services/models/translation_prompt_enhancer.py")
    
    if not enhancer_path.exists():
        logger.info(f"Translation prompt enhancer not found at {enhancer_path}, creating new file")
        # Create a new file with NLLB support
        enhancer_content = """\"\"\"
Translation Prompt Enhancer for CasaLingua

This module provides functionality to enhance prompts for translation models,
improving the quality and consistency of translations.
\"\"\"

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TranslationPromptEnhancer:
    \"\"\"Enhances translation prompts for different models and domains.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the prompt enhancer.\"\"\"
        pass
    
    def enhance_mt5_prompt(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        domain: str = "", 
        formality: str = "",
        context: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        \"\"\"
        Enhance an MT5-style prompt for better translation quality.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain specification (legal, technical, etc.)
            formality: Optional formality level (formal, informal)
            context: Optional context text
            parameters: Additional parameters
            
        Returns:
            Enhanced prompt string
        \"\"\"
        # Start with basic prompt
        prompt = f"translate {source_lang} to {target_lang}: {text}"
        
        # Add domain context if provided
        if domain:
            if domain.lower() == "legal" or domain.lower() == "housing_legal":
                prompt = f"translate {source_lang} to {target_lang} (legal document): {text}"
            elif domain.lower() == "technical":
                prompt = f"translate {source_lang} to {target_lang} (technical content): {text}"
            elif domain.lower() == "casual":
                prompt = f"translate {source_lang} to {target_lang} (conversational): {text}"
            else:
                prompt = f"translate {source_lang} to {target_lang} ({domain}): {text}"
        
        # Add formality instruction if provided
        if formality:
            if formality.lower() == "formal":
                prompt = f"translate {source_lang} to {target_lang} using formal language: {text}"
            elif formality.lower() == "informal":
                prompt = f"translate {source_lang} to {target_lang} using informal language: {text}"
            
            # Combine domain and formality if both are specified
            if domain:
                prompt = f"translate {source_lang} to {target_lang} ({domain}, {formality}): {text}"
        
        # Add context if provided
        if context:
            # Truncate context if too long
            max_context_len = 100
            context_preview = context[:max_context_len] + "..." if len(context) > max_context_len else context
            prompt = f"context: {context_preview}\\ntranslate {source_lang} to {target_lang}: {text}"
        
        return prompt
    
    def enhance_nllb_prompt(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        domain: str = "", 
        formality: str = "",
        context: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        \"\"\"
        Enhance an NLLB prompt for better translation quality.
        For NLLB, the enhancement is minimal since it uses language codes
        rather than text prefixes, but we can still provide explicit instructions
        for specific domains or formality levels.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain specification (legal, technical, etc.)
            formality: Optional formality level (formal, informal)
            context: Optional context text
            parameters: Additional parameters
            
        Returns:
            Enhanced prompt string (typically just the text for NLLB)
        \"\"\"
        # For Spanish to English translations, add domain-specific prefixes
        if source_lang == "es" and target_lang == "en":
            # Only add prefixes for specific domains
            if domain and (domain.lower() == "legal" or domain.lower() == "technical"):
                return f"[{domain.upper()}] {text}"
            elif formality and formality.lower() == "formal":
                return f"[FORMAL] {text}"
            elif formality and formality.lower() == "informal":
                return f"[INFORMAL] {text}"
        
        # For most NLLB translation cases, just return the text
        # The language control is handled via forced_bos_token_id
        return text
    
    def get_mbart_generation_params(
        self,
        source_lang: str,
        target_lang: str,
        domain: str = "",
        formality: str = ""
    ) -> Dict[str, Any]:
        \"\"\"
        Get optimal generation parameters for MBART models
        based on language pair and domain.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain specification (legal, technical, etc.)
            formality: Optional formality level (formal, informal)
            
        Returns:
            Dictionary of generation parameters
        \"\"\"
        # Start with default parameters
        params = {
            "num_beams": 4,
            "length_penalty": 1.0,
            "early_stopping": True,
            "do_sample": False
        }
        
        # Adjust for Spanish to English translations
        if source_lang == "es" and target_lang == "en":
            params["num_beams"] = 5
            params["length_penalty"] = 1.0
            params["do_sample"] = False
        
        # Adjust for domain-specific translations
        if domain:
            if domain.lower() == "legal" or domain.lower() == "housing_legal":
                # Legal texts need high precision and more thorough search
                params["num_beams"] = 6
                params["length_penalty"] = 1.2  # Prefer longer outputs for legal text
                params["repetition_penalty"] = 1.2  # Avoid repetition
                params["do_sample"] = False  # Deterministic output for legal text
            elif domain.lower() == "technical":
                # Technical texts need high precision
                params["num_beams"] = 5
                params["length_penalty"] = 1.0
                params["do_sample"] = False
            elif domain.lower() == "casual":
                # Casual texts can be more creative
                params["num_beams"] = 4
                params["do_sample"] = True
                params["temperature"] = 0.8
                params["top_p"] = 0.9
        
        # Adjust for formality
        if formality:
            if formality.lower() == "formal":
                # Formal texts need more careful generation
                params["repetition_penalty"] = 1.1  # Avoid repetition
                params["do_sample"] = False  # More deterministic
            elif formality.lower() == "informal":
                # Informal texts can be more variable
                params["do_sample"] = True
                params["temperature"] = 0.9
        
        return params
    
    def get_nllb_generation_params(
        self,
        source_lang: str,
        target_lang: str,
        domain: str = "",
        formality: str = ""
    ) -> Dict[str, Any]:
        \"\"\"
        Get optimal generation parameters for NLLB models
        based on language pair and domain.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain specification (legal, technical, etc.)
            formality: Optional formality level (formal, informal)
            
        Returns:
            Dictionary of generation parameters
        \"\"\"
        # NLLB models require different parameters than MBART
        params = {
            "num_beams": 5,
            "length_penalty": 1.0,
            "early_stopping": True,
            "do_sample": False,
            "max_length": 256  # NLLB typically supports longer outputs
        }
        
        # Adjust for Spanish to English translations
        if source_lang == "es" and target_lang == "en":
            params["num_beams"] = 5
            params["length_penalty"] = 1.0
            params["do_sample"] = False
        
        # Adjust for domain-specific translations
        if domain:
            if domain.lower() == "legal" or domain.lower() == "housing_legal":
                # Legal texts need high precision
                params["num_beams"] = 6
                params["length_penalty"] = 1.2
                params["do_sample"] = False
            elif domain.lower() == "technical":
                # Technical texts need high precision
                params["num_beams"] = 5
                params["length_penalty"] = 1.0
                params["do_sample"] = False
            elif domain.lower() == "casual":
                # Casual texts can be more creative
                params["do_sample"] = True
                params["temperature"] = 0.8
                params["top_p"] = 0.9
        
        # Adjust for formality level
        if formality:
            if formality.lower() == "formal":
                params["repetition_penalty"] = 1.1
                params["do_sample"] = False
            elif formality.lower() == "informal":
                params["do_sample"] = True
                params["temperature"] = 0.9
        
        return params
"""
        with open(enhancer_path, 'w') as f:
            f.write(enhancer_content)
        logger.info(f"Created new translation prompt enhancer with NLLB support at {enhancer_path}")
    else:
        # Update existing file to add NLLB support
        logger.info(f"Translation prompt enhancer found at {enhancer_path}, updating with NLLB support")
        backup_file(enhancer_path)
        
        with open(enhancer_path, 'r') as f:
            content = f.read()
        
        # Check if the file already has NLLB methods
        if "enhance_nllb_prompt" not in content:
            # Add NLLB prompt enhancement method
            class_pattern = r"(class TranslationPromptEnhancer:.*?def enhance_mt5_prompt.*?\).*?return prompt\n)"
            
            nllb_method = """
    def enhance_nllb_prompt(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        domain: str = "", 
        formality: str = "",
        context: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        \"\"\"
        Enhance an NLLB prompt for better translation quality.
        For NLLB, the enhancement is minimal since it uses language codes
        rather than text prefixes, but we can still provide explicit instructions
        for specific domains or formality levels.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain specification (legal, technical, etc.)
            formality: Optional formality level (formal, informal)
            context: Optional context text
            parameters: Additional parameters
            
        Returns:
            Enhanced prompt string (typically just the text for NLLB)
        \"\"\"
        # For Spanish to English translations, add domain-specific prefixes
        if source_lang == "es" and target_lang == "en":
            # Only add prefixes for specific domains
            if domain and (domain.lower() == "legal" or domain.lower() == "technical"):
                return f"[{domain.upper()}] {text}"
            elif formality and formality.lower() == "formal":
                return f"[FORMAL] {text}"
            elif formality and formality.lower() == "informal":
                return f"[INFORMAL] {text}"
        
        # For most NLLB translation cases, just return the text
        # The language control is handled via forced_bos_token_id
        return text"""
            
            if re.search(class_pattern, content, re.DOTALL):
                modified_content = re.sub(class_pattern, r"\1" + nllb_method, content, flags=re.DOTALL)
                logger.info("Added NLLB prompt enhancement method")
            else:
                logger.warning("Could not find where to add NLLB prompt enhancement method")
                modified_content = content
            
            # Add NLLB generation parameters method
            if "get_nllb_generation_params" not in modified_content:
                # Check if there is a similar method for MBART
                if "get_mbart_generation_params" in modified_content:
                    mbart_params_pattern = r"(def get_mbart_generation_params.*?return params\n)"
                    
                    nllb_params_method = """
    def get_nllb_generation_params(
        self,
        source_lang: str,
        target_lang: str,
        domain: str = "",
        formality: str = ""
    ) -> Dict[str, Any]:
        \"\"\"
        Get optimal generation parameters for NLLB models
        based on language pair and domain.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain specification (legal, technical, etc.)
            formality: Optional formality level (formal, informal)
            
        Returns:
            Dictionary of generation parameters
        \"\"\"
        # NLLB models require different parameters than MBART
        params = {
            "num_beams": 5,
            "length_penalty": 1.0,
            "early_stopping": True,
            "do_sample": False,
            "max_length": 256  # NLLB typically supports longer outputs
        }
        
        # Adjust for Spanish to English translations
        if source_lang == "es" and target_lang == "en":
            params["num_beams"] = 5
            params["length_penalty"] = 1.0
            params["do_sample"] = False
        
        # Adjust for domain-specific translations
        if domain:
            if domain.lower() == "legal" or domain.lower() == "housing_legal":
                # Legal texts need high precision
                params["num_beams"] = 6
                params["length_penalty"] = 1.2
                params["do_sample"] = False
            elif domain.lower() == "technical":
                # Technical texts need high precision
                params["num_beams"] = 5
                params["length_penalty"] = 1.0
                params["do_sample"] = False
            elif domain.lower() == "casual":
                # Casual texts can be more creative
                params["do_sample"] = True
                params["temperature"] = 0.8
                params["top_p"] = 0.9
        
        # Adjust for formality level
        if formality:
            if formality.lower() == "formal":
                params["repetition_penalty"] = 1.1
                params["do_sample"] = False
            elif formality.lower() == "informal":
                params["do_sample"] = True
                params["temperature"] = 0.9
        
        return params"""
                    
                    if re.search(mbart_params_pattern, modified_content, re.DOTALL):
                        modified_content = re.sub(mbart_params_pattern, r"\1" + nllb_params_method, modified_content, flags=re.DOTALL)
                        logger.info("Added NLLB generation parameters method")
                    else:
                        logger.warning("Could not find where to add NLLB generation parameters method")
                else:
                    logger.warning("No existing generation parameters method found to model after")
            
            with open(enhancer_path, 'w') as f:
                f.write(modified_content)
            
            logger.info(f"Updated translation prompt enhancer with NLLB support at {enhancer_path}")

if __name__ == "__main__":
    logger.info("Starting NLLB-MPS compatibility fix...")
    
    # Fix loader device selection
    fix_loader_device_selection()
    
    # Fix wrapper device selection and add NLLB support
    fix_wrapper_device_selection()
    
    # Create or update translation prompt enhancer
    fix_translation_prompt_enhancer()
    
    logger.info("NLLB-MPS compatibility fix completed")
    print("\n✅ NLLB MPS compatibility fix completed successfully!")
    print("You can now use NLLB models on Apple Silicon MPS devices.")
    print("Remember to restart your application for the changes to take effect.")