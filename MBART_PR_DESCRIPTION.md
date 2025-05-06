# Prioritize MBART for Improved Translation Quality

## Summary

This PR makes MBART the default and primary translation model in CasaLingua based on quality testing that showed it consistently outperforms MT5, even after upgrading to MT5-base. The changes ensure all translation requests utilize MBART for optimal quality.

## Changes

1. **Updated model_registry.json configuration**:
   - Made MBART the primary translation model by assigning it to the "translation" key
   - Moved MT5 to "mt5_translation" and marked it as a fallback model
   - Added explicit "is_primary" and "is_fallback" flags to clarify roles

2. **Updated translator.py implementation**:
   - Modified model selection logic to work with the new configuration
   - Updated references from "mbart_translation" to "translation" to reflect the new configuration
   - Ensured the fallback mechanism properly references the updated model names

3. **Added documentation**:
   - Created MODEL_SELECTION_LOGIC.md explaining the model selection process
   - Updated API documentation (docs/api/translation.md) to highlight MBART's superior quality
   - Added quality comparison information to help users understand the benefits

4. **Added test script**:
   - Created test_mbart_primary_selection.py to verify MBART is correctly used by default
   - Test ensures both implicit and explicit model selection works properly

## Motivation

Extensive translation quality testing showed MBART significantly outperforms MT5 across multiple languages and content types:

- **Better accuracy**: MBART properly translates full sentences and maintains context
- **Improved fluency**: More natural-sounding output that resembles native speakers
- **Superior handling of idioms**: Better preserves figurative language and cultural expressions
- **Complete translations**: Avoids the truncated or incomplete translations sometimes seen with MT5

This change ensures users receive the highest quality translations by default without having to explicitly request MBART.

## Test Plan

1. Run the test_mbart_primary_selection.py script to verify MBART is used by default
2. Test the translation API with various language pairs to verify proper model selection
3. Verify translation quality in both simple and complex sentences
4. Confirm the API documentation accurately represents the updated model configuration

## Documentation Updates

- Updated translation API documentation to highlight MBART as the primary model
- Added information about model quality comparisons
- Updated request and response examples to reflect the new configuration

## Related Issues

- Fixes ISSUES-MBART.md Issue 5: Update TranslationModelWrapper for MBART support