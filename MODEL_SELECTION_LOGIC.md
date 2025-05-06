# Translation Model Selection Logic

## Overview

This document explains how CasaLingua selects translation models and the current configuration that prioritizes MBART over MT5 based on our translation quality findings.

## Current Model Selection Logic

The translation system decides which model to use through the following process:

1. **Primary Logic in `translator.py:435-444`**:
   ```python
   # If no model_id provided or if it's explicitly set to mt5_translation, use MBART instead
   if model_id is None or model_id == "mt5_translation":
       # Get MBART language codes
       mbart_source_lang = self._get_mbart_language_code(source_language)
       mbart_target_lang = self._get_mbart_language_code(target_language)
       
       # Use MBART as primary model
       logger.info(f"Using MBART as primary translation model for {source_language} to {target_language}")
       model_id = "mbart_translation"
   ```

   This code ensures that MBART is used by default, even when MT5 is explicitly requested. The translation pipeline automatically switches to MBART in these cases.

2. **Fallback Mechanism in `translator.py:481-517`**:
   ```python
   # Check if the translation was successful
   if model_id != "mbart_translation" and use_fallback and self._is_poor_quality_translation(result.translated_text, text):
       logger.warning(f"Primary translation model produced poor quality result, attempting fallback for {source_language} to {target_language}")
       
       # Try fallback with MBART model
       # ...code to use MBART as fallback...
   ```

   If any non-MBART model produces poor quality translation (detected by `_is_poor_quality_translation`), the system will fall back to MBART.

3. **Quality Detection in `translator.py:541-591`**:
   The `_is_poor_quality_translation` method detects problematic translations using several heuristics:
   - Empty translations
   - Translations identical to the source text
   - Very short translations (when source was longer)
   - Outputs containing language codes (hallucination)
   - Outputs containing special tokens
   - Outputs with severe token repetition

## Model Registry Configuration

In `config/model_registry.json`, we have both models configured:

```json
"mbart_translation": {
    "model_name": "facebook/mbart-large-50-many-to-many-mmt",
    "tokenizer_name": "facebook/mbart-large-50-many-to-many-mmt",
    "task": "translation",
    "type": "transformers",
    "model_class": "MBartForConditionalGeneration",
    "framework": "transformers"
},
"translation": {
    "model_name": "google/mt5-base",
    "tokenizer_name": "google/mt5-base",
    "task": "translation",
    "type": "transformers",
    "framework": "transformers"
}
```

While "translation" still points to the MT5 model, the code overrides this and uses "mbart_translation" by default.

## Rationale for MBART Prioritization

Based on extensive testing documented in `TRANSLATION_FINDINGS.md`, we found that:

1. **MBART Produces Superior Quality**: Despite upgrading from MT5-small to MT5-base, MBART consistently produces far better translations across all test cases.

2. **MT5 Quality Issues Persist**: MT5-base continues to produce incomplete or low-quality translations, often producing fragments, repetition of source text, or completely empty translations.

3. **Specific Examples**:
   - English to Spanish with MT5: "The quick brown fox :" (incomplete)
   - Same sentence with MBART: "La fox marrón rápida salta sobre el can loco." (complete and accurate)

4. **Performance Trade-off**: MBART is slower than MT5 (average 1.66s vs 0.62s per translation), but the quality difference makes this trade-off worthwhile.

## Implementation Notes

The current implementation automatically uses MBART in two cases:
1. When no specific model is requested
2. When MT5 is requested but produces poor quality output

This ensures optimal translation quality while maintaining compatibility with existing code that might explicitly request MT5.