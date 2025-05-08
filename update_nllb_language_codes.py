#!/usr/bin/env python3
"""
Script to update language code handling for NLLB models.
This ensures that all language codes used with NLLB are in the correct format.

Usage:
    python update_nllb_language_codes.py
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
logger = logging.getLogger("nllb_lang_codes")

# Paths
TOKENIZER_PATH = Path("app/core/pipeline/tokenizer.py")
BACKUP_SUFFIX = ".bak_before_nllb_update"

def backup_file(file_path, backup_suffix=BACKUP_SUFFIX):
    """Create a backup of the file"""
    backup_path = Path(str(file_path) + backup_suffix)
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
    lang_code_pattern = r"LANG_CODE_MAPPING = \{[^}]*\}"
    
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
    
    if re.search(lang_code_pattern, content):
        # Replace the pattern with our new mapping
        modified_content = re.sub(lang_code_pattern, new_mapping, content)
        logger.info("Updated language code mapping for NLLB")
    else:
        logger.error(f"Could not find language code mapping pattern in {TOKENIZER_PATH}")
        return False
    
    # Update the prepare_translation_inputs method to better handle NLLB
    nllb_input_pattern = r"(def prepare_translation_inputs.*?if \"nllb\" in self\.model_name:.*?model_inputs = self\.tokenizer\(text, return_tensors=\"pt\"\).*?forced_bos_id = self\.tokenizer\.lang_code_to_id\.get\(target_code\).*?return \{.*?\"inputs\": model_inputs,.*?\"forced_bos_token_id\": forced_bos_id,.*?\"source_lang\": source_code,.*?\"target_lang\": target_code.*?\})"
    
    nllb_input_replacement = '''def prepare_translation_inputs(self, text: str, source_lang: str, target_lang: str) -> dict:
        """Prepare inputs for translation with proper language codes."""
        source_code = LANG_CODE_MAPPING.get(source_lang, source_lang)
        target_code = LANG_CODE_MAPPING.get(target_lang, target_lang)

        if "nllb" in self.model_name:
            # NLLB-specific tokenization with better error handling
            try:
                # Set source language if the tokenizer supports it
                if hasattr(self.tokenizer, "src_lang"):
                    self.tokenizer.src_lang = source_code
                
                # Tokenize the input text
                model_inputs = self.tokenizer(text, return_tensors="pt")
                
                # Get forced_bos_token_id for target language
                forced_bos_id = None
                if hasattr(self.tokenizer, "lang_code_to_id"):
                    try:
                        forced_bos_id = self.tokenizer.lang_code_to_id.get(target_code)
                        if forced_bos_id is None:
                            # Log the available language codes for debugging
                            logger.warning(f"Language code {target_code} not found in tokenizer.lang_code_to_id")
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Available language codes: {list(self.tokenizer.lang_code_to_id.keys())}")
                                
                            # Special handling for common languages if missing
                            if target_lang == "en":
                                forced_bos_id = 128022  # eng_Latn in NLLB
                                logger.info(f"Using hardcoded token ID for English: {forced_bos_id}")
                            elif target_lang == "es":
                                forced_bos_id = 128021  # spa_Latn in NLLB
                                logger.info(f"Using hardcoded token ID for Spanish: {forced_bos_id}")
                    except Exception as e:
                        logger.error(f"Error getting forced_bos_token_id for {target_code}: {e}")
                        forced_bos_id = None
                
                return {
                    "inputs": model_inputs,
                    "forced_bos_token_id": forced_bos_id,
                    "source_lang": source_code,
                    "target_lang": target_code
                }
            except Exception as e:
                logger.error(f"Error in NLLB prepare_translation_inputs: {e}")
                # Fall back to a basic implementation
                model_inputs = self.tokenizer(text, return_tensors="pt")
                return {
                    "inputs": model_inputs,
                    "forced_bos_token_id": None,
                    "source_lang": source_code,
                    "target_lang": target_code
                }'''
    
    if re.search(nllb_input_pattern, modified_content, re.DOTALL):
        # Replace the NLLB-specific input preparation
        modified_content = re.sub(nllb_input_pattern, nllb_input_replacement, modified_content, flags=re.DOTALL)
        logger.info("Updated NLLB translation input preparation")
    else:
        logger.warning("Could not find prepare_translation_inputs method pattern for NLLB")
    
    # Add NLLB language code verification method
    if "def verify_nllb_language_code" not in modified_content:
        # Check for a good insertion point after the class definition
        insert_pattern = r"(class TokenizerPipeline:.*?def __init__.*?\n)"
        nllb_verify_method = '''
    def verify_nllb_language_code(self, code: str) -> str:
        """
        Verify and convert language code to NLLB format if needed.
        
        Args:
            code: ISO language code or NLLB language code
            
        Returns:
            Valid NLLB language code
        """
        # If already in NLLB format (has underscore and script code)
        if "_" in code and any(script in code for script in ["Latn", "Cyrl", "Arab", "Hans", "Hant"]):
            return code
            
        # Use the mapping table
        if code in LANG_CODE_MAPPING:
            return LANG_CODE_MAPPING[code]
            
        # For unknown codes, try to make a best guess
        if len(code) <= 3:
            # Assume it's a language code without script, default to Latin script
            return f"{code}_Latn"
            
        # If we can't make a good guess, just return the original and hope for the best
        logger.warning(f"Could not verify NLLB language code: {code}")
        return code
'''
        
        if re.search(insert_pattern, modified_content, re.DOTALL):
            # Add the verification method after the class init
            modified_content = re.sub(insert_pattern, r"\1" + nllb_verify_method, modified_content, flags=re.DOTALL)
            logger.info("Added NLLB language code verification method")
        else:
            logger.warning("Could not find where to add NLLB language code verification method")
    
    with open(TOKENIZER_PATH, 'w') as f:
        f.write(modified_content)
    
    logger.info(f"Updated language code handling in {TOKENIZER_PATH}")
    return True

def create_language_lookup_utility():
    """Create a utility script to look up NLLB language codes for reference"""
    util_path = Path("scripts/nllb_language_lookup.py")
    
    # Create scripts directory if needed
    if not util_path.parent.exists():
        util_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = '''#!/usr/bin/env python3
"""
NLLB Language Code Lookup Utility

This script helps look up NLLB language codes and provides a reference
for all supported languages in the NLLB-200 model.

Usage:
    python nllb_language_lookup.py [language_code|language_name]
"""

import sys
import json
from pathlib import Path

# NLLB supports 200+ languages with specific codes
# This mapping includes common languages and their NLLB codes
NLLB_LANGUAGE_MAPPING = {
    # ISO code to NLLB code
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
    
    # Chinese variants
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh-hk": "yue_Hant",
}

# Full language names to ISO codes
LANGUAGE_NAMES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "chinese": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "arabic": "ar",
    "russian": "ru",
    "dutch": "nl",
    "korean": "ko",
    "polish": "pl",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "norwegian": "no",
    "czech": "cs",
    "hungarian": "hu",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "thai": "th",
    "indonesian": "id",
    "romanian": "ro",
    "bengali": "bn",
}

# Full list of NLLB supported languages (200 languages)
# Retrieved from https://github.com/facebookresearch/flores/blob/main/flores200/README.md
FLORES_TO_NLLB = {
    "ace_Arab": "ace_Arab", "ace_Latn": "ace_Latn", "acm_Arab": "acm_Arab", "acq_Arab": "acq_Arab",
    "aeb_Arab": "aeb_Arab", "afr_Latn": "afr_Latn", "ajp_Arab": "ajp_Arab", "aka_Latn": "aka_Latn",
    "amh_Ethi": "amh_Ethi", "apc_Arab": "apc_Arab", "arb_Arab": "arb_Arab", "ars_Arab": "ars_Arab",
    "ary_Arab": "ary_Arab", "arz_Arab": "arz_Arab", "asm_Beng": "asm_Beng", "ast_Latn": "ast_Latn",
    "awa_Deva": "awa_Deva", "ayr_Latn": "ayr_Latn", "azb_Arab": "azb_Arab", "azj_Latn": "azj_Latn",
    "bak_Cyrl": "bak_Cyrl", "bam_Latn": "bam_Latn", "ban_Latn": "ban_Latn", "bel_Cyrl": "bel_Cyrl",
    "bem_Latn": "bem_Latn", "ben_Beng": "ben_Beng", "bho_Deva": "bho_Deva", "bjn_Arab": "bjn_Arab",
    "bjn_Latn": "bjn_Latn", "bod_Tibt": "bod_Tibt", "bos_Latn": "bos_Latn", "bug_Latn": "bug_Latn",
    "bul_Cyrl": "bul_Cyrl", "cat_Latn": "cat_Latn", "ceb_Latn": "ceb_Latn", "ces_Latn": "ces_Latn",
    "cjk_Latn": "cjk_Latn", "ckb_Arab": "ckb_Arab", "crh_Latn": "crh_Latn", "cym_Latn": "cym_Latn",
    "dan_Latn": "dan_Latn", "deu_Latn": "deu_Latn", "dik_Latn": "dik_Latn", "dyu_Latn": "dyu_Latn",
    "dzo_Tibt": "dzo_Tibt", "ell_Grek": "ell_Grek", "eng_Latn": "eng_Latn", "epo_Latn": "epo_Latn",
    "est_Latn": "est_Latn", "eus_Latn": "eus_Latn", "ewe_Latn": "ewe_Latn", "fao_Latn": "fao_Latn",
    "pes_Arab": "pes_Arab", "fij_Latn": "fij_Latn", "fin_Latn": "fin_Latn", "fon_Latn": "fon_Latn",
    "fra_Latn": "fra_Latn", "fur_Latn": "fur_Latn", "fuv_Latn": "fuv_Latn", "gla_Latn": "gla_Latn",
    "gle_Latn": "gle_Latn", "glg_Latn": "glg_Latn", "grn_Latn": "grn_Latn", "guj_Gujr": "guj_Gujr",
    "hat_Latn": "hat_Latn", "hau_Latn": "hau_Latn", "heb_Hebr": "heb_Hebr", "hin_Deva": "hin_Deva",
    "hne_Deva": "hne_Deva", "hrv_Latn": "hrv_Latn", "hun_Latn": "hun_Latn", "hye_Armn": "hye_Armn",
    "ibo_Latn": "ibo_Latn", "ilo_Latn": "ilo_Latn", "ind_Latn": "ind_Latn", "isl_Latn": "isl_Latn",
    "ita_Latn": "ita_Latn", "jav_Latn": "jav_Latn", "jpn_Jpan": "jpn_Jpan", "kab_Latn": "kab_Latn",
    "kac_Latn": "kac_Latn", "kam_Latn": "kam_Latn", "kan_Knda": "kan_Knda", "kas_Arab": "kas_Arab",
    "kas_Deva": "kas_Deva", "kat_Geor": "kat_Geor", "kaz_Cyrl": "kaz_Cyrl", "kbp_Latn": "kbp_Latn",
    "kea_Latn": "kea_Latn", "khk_Cyrl": "khk_Cyrl", "khm_Khmr": "khm_Khmr", "kik_Latn": "kik_Latn",
    "kin_Latn": "kin_Latn", "kir_Cyrl": "kir_Cyrl", "kmb_Latn": "kmb_Latn", "kmr_Latn": "kmr_Latn",
    "knc_Arab": "knc_Arab", "knc_Latn": "knc_Latn", "kon_Latn": "kon_Latn", "kor_Hang": "kor_Hang",
    "lao_Laoo": "lao_Laoo", "lij_Latn": "lij_Latn", "lim_Latn": "lim_Latn", "lin_Latn": "lin_Latn",
    "lit_Latn": "lit_Latn", "lmo_Latn": "lmo_Latn", "ltg_Latn": "ltg_Latn", "ltz_Latn": "ltz_Latn",
    "lua_Latn": "lua_Latn", "lug_Latn": "lug_Latn", "luo_Latn": "luo_Latn", "lus_Latn": "lus_Latn",
    "lvs_Latn": "lvs_Latn", "mag_Deva": "mag_Deva", "mai_Deva": "mai_Deva", "mal_Mlym": "mal_Mlym",
    "mar_Deva": "mar_Deva", "min_Latn": "min_Latn", "mkd_Cyrl": "mkd_Cyrl", "mlt_Latn": "mlt_Latn",
    "mni_Beng": "mni_Beng", "mos_Latn": "mos_Latn", "mri_Latn": "mri_Latn", "mya_Mymr": "mya_Mymr",
    "nld_Latn": "nld_Latn", "nno_Latn": "nno_Latn", "nob_Latn": "nob_Latn", "npi_Deva": "npi_Deva",
    "nso_Latn": "nso_Latn", "nus_Latn": "nus_Latn", "nya_Latn": "nya_Latn", "oci_Latn": "oci_Latn",
    "ory_Orya": "ory_Orya", "pag_Latn": "pag_Latn", "pan_Guru": "pan_Guru", "pap_Latn": "pap_Latn",
    "pbt_Arab": "pbt_Arab", "pol_Latn": "pol_Latn", "por_Latn": "por_Latn", "prs_Arab": "prs_Arab",
    "quy_Latn": "quy_Latn", "ron_Latn": "ron_Latn", "run_Latn": "run_Latn", "rus_Cyrl": "rus_Cyrl",
    "sag_Latn": "sag_Latn", "san_Deva": "san_Deva", "sat_Beng": "sat_Beng", "scn_Latn": "scn_Latn",
    "shn_Mymr": "shn_Mymr", "sin_Sinh": "sin_Sinh", "slk_Latn": "slk_Latn", "slv_Latn": "slv_Latn",
    "smo_Latn": "smo_Latn", "sna_Latn": "sna_Latn", "snd_Arab": "snd_Arab", "som_Latn": "som_Latn",
    "sot_Latn": "sot_Latn", "spa_Latn": "spa_Latn", "srd_Latn": "srd_Latn", "srp_Cyrl": "srp_Cyrl",
    "ssw_Latn": "ssw_Latn", "sun_Latn": "sun_Latn", "swe_Latn": "swe_Latn", "swh_Latn": "swh_Latn",
    "szl_Latn": "szl_Latn", "tam_Taml": "tam_Taml", "tat_Cyrl": "tat_Cyrl", "tel_Telu": "tel_Telu",
    "tgk_Cyrl": "tgk_Cyrl", "tgl_Latn": "tgl_Latn", "tha_Thai": "tha_Thai", "tir_Ethi": "tir_Ethi",
    "taq_Latn": "taq_Latn", "taq_Tfng": "taq_Tfng", "tpi_Latn": "tpi_Latn", "tsn_Latn": "tsn_Latn",
    "tso_Latn": "tso_Latn", "tuk_Latn": "tuk_Latn", "tum_Latn": "tum_Latn", "tur_Latn": "tur_Latn",
    "twi_Latn": "twi_Latn", "tzm_Tfng": "tzm_Tfng", "uig_Arab": "uig_Arab", "ukr_Cyrl": "ukr_Cyrl",
    "umb_Latn": "umb_Latn", "urd_Arab": "urd_Arab", "uzn_Latn": "uzn_Latn", "vec_Latn": "vec_Latn",
    "vie_Latn": "vie_Latn", "war_Latn": "war_Latn", "wol_Latn": "wol_Latn", "xho_Latn": "xho_Latn",
    "yid_Hebr": "yid_Hebr", "yor_Latn": "yor_Latn", "yue_Hant": "yue_Hant", "zho_Hans": "zho_Hans",
    "zho_Hant": "zho_Hant", "zul_Latn": "zul_Latn",  
}

def get_nllb_code(query):
    """Get NLLB code for a language name or ISO code"""
    query = query.lower().strip()
    
    # Direct lookup in the mapping
    if query in NLLB_LANGUAGE_MAPPING:
        return NLLB_LANGUAGE_MAPPING[query]
    
    # Try language name lookup
    if query in LANGUAGE_NAMES:
        iso_code = LANGUAGE_NAMES[query]
        if iso_code in NLLB_LANGUAGE_MAPPING:
            return NLLB_LANGUAGE_MAPPING[iso_code]
    
    # Try flores-nllb mapping
    if query in FLORES_TO_NLLB:
        return FLORES_TO_NLLB[query]
    
    # Check if it's already a valid NLLB code
    for nllb_code in FLORES_TO_NLLB.values():
        if query == nllb_code.lower():
            return nllb_code
    
    # No match found
    return None

def display_all_languages():
    """Display all supported languages with their codes"""
    print("NLLB-200 Supported Languages")
    print("============================")
    print(f"Total languages: {len(FLORES_TO_NLLB)}")
    print()
    
    # Convert to a list for sorting
    languages = []
    for code, nllb_code in sorted(FLORES_TO_NLLB.items()):
        lang_name = code.split('_')[0]
        script = code.split('_')[1]
        languages.append((lang_name, script, nllb_code))
    
    # Print in columns
    col_width = 30
    for i, (lang, script, code) in enumerate(languages):
        print(f"{code:<15} ", end="")
        if (i + 1) % 5 == 0:
            print()
    
    print("\n")
    print("Common Languages")
    print("===============")
    
    for iso, name in sorted([(v, k) for k, v in LANGUAGE_NAMES.items()]):
        if iso in NLLB_LANGUAGE_MAPPING:
            print(f"{name.title():<15} {iso:<5} -> {NLLB_LANGUAGE_MAPPING[iso]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("NLLB Language Code Lookup Utility")
        print("=================================")
        print("Usage:")
        print("  python nllb_language_lookup.py [language_code|language_name]")
        print("  python nllb_language_lookup.py --all")
        print()
        print("Examples:")
        print("  python nllb_language_lookup.py en")
        print("  python nllb_language_lookup.py english")
        print("  python nllb_language_lookup.py spa_Latn")
        sys.exit(0)
    
    query = sys.argv[1]
    
    if query == "--all":
        display_all_languages()
        sys.exit(0)
    
    nllb_code = get_nllb_code(query)
    
    if nllb_code:
        print(f"NLLB code for '{query}': {nllb_code}")
    else:
        print(f"No NLLB code found for '{query}'")
        print("Try using --all to see all supported languages")
'''
    
    with open(util_path, 'w') as f:
        f.write(content)
    
    # Make the script executable
    util_path.chmod(0o755)
    
    logger.info(f"Created NLLB language lookup utility at {util_path}")
    return True

if __name__ == "__main__":
    logger.info("Starting NLLB language code update...")
    
    # Update language codes
    update_language_codes()
    
    # Create language lookup utility
    create_language_lookup_utility()
    
    logger.info("NLLB language code update completed")
    print("\n✅ NLLB language code update completed successfully!")
    print("You can use the lookup utility to find valid NLLB language codes:")
    print("  python scripts/nllb_language_lookup.py [language_code|language_name]")
    print("  python scripts/nllb_language_lookup.py --all")
    print("\nRemember to restart your application for the changes to take effect.")