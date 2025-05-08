# Spanish to English Translation Fix

This document explains the fix implemented to resolve Spanish to English translation issues on Apple Silicon devices using MBART models.

## Problem

Spanish to English translations were failing with the error:
```
Error: Error processing model: 'NoneType' object has no attribute 'get'
```

This issue occurred specifically when:
1. Using MBART translation models
2. Running on Apple Silicon devices with MPS (Metal Performance Shaders)
3. Translating from Spanish to English

## Root Cause

The root cause was identified as a combination of issues:

1. **MBART Model Compatibility**: MBART models have known compatibility issues with Apple's MPS backend, particularly with the `forced_bos_token_id` parameter.

2. **Device Detection**: The models were sometimes being loaded on MPS when they should be forced to CPU.

3. **Language Token Handling**: For Spanish to English translations, the `forced_bos_token_id` parameter needs to be explicitly set to 2 (the English token ID) for MBART models, but this wasn't being handled correctly.

## Solution

The implemented fix involves three main components:

1. **Enhanced MBART Detection**: We improved the MBART model detection in both the `ModelLoader._determine_device` method and the `BaseModelWrapper.__init__` method to ensure all MBART variants are correctly detected and forced to use CPU on Apple Silicon devices.

2. **Improved forced_bos_token_id Handling**: For Spanish to English translations specifically, we now explicitly set `forced_bos_token_id=2` (the English token ID) in the generation parameters.

3. **Special Case Detection**: We added explicit detection of Spanish to English translation pairs to apply special handling regardless of model type.

## Fix Implementation

The following changes were made:

1. In `app/services/models/loader.py`:
   - Enhanced the `_determine_device` method to better detect all MBART variants
   - Added special case handling for all translation models on MPS devices

2. In `app/services/models/wrapper.py`:
   - Enhanced the `BaseModelWrapper.__init__` method to detect MBART models more accurately
   - Improved the `TranslationModelWrapper._preprocess` method to handle Spanish to English cases
   - Ensured proper setting of `forced_bos_token_id` in the `TranslationModelWrapper._run_inference` method

## Testing the Fix

Two test scripts are provided to verify the fix:

1. `fix_spanish_to_english_translation.py` - A simple script that tests basic Spanish to English translation
2. `test_spanish_english_fixed.py` - A more comprehensive test that checks both API endpoints and direct model usage

To test the fix:

```bash
# Basic test
./fix_spanish_to_english_translation.py

# Comprehensive test
./test_spanish_english_fixed.py --all
```

## Notes

- This fix ensures that MBART models are always run on CPU when on an Apple Silicon device, which slightly reduces performance but ensures reliability.
- For optimal performance while maintaining reliability, consider using MT5 models which generally work better with MPS.
- The fix should work for other language pairs as well, but has been specifically tested and optimized for Spanish to English translations.

## References

- [Hugging Face MBART Documentation](https://huggingface.co/docs/transformers/model_doc/mbart)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Silicon ML Optimization Guide](https://developer.apple.com/metal/pytorch/)