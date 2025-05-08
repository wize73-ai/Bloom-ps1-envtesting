# NLLB language mapping module
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
