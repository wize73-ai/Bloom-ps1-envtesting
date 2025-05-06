# MBART Model Loading Fixes

This document describes the fixes implemented to address MBART model loading issues.

## Issues Fixed

1. **MBART Model Detection**: Improved detection logic in TranslationModelWrapper to correctly identify MBART models by checking for "mbart" in the model type string (case-insensitive).

2. **Specialized Tokenizers**: Added explicit imports and handling for MBART tokenizers (MBart50TokenizerFast and MBartTokenizer) to ensure they're available and properly loaded.

3. **Model Loading Logic**: Enhanced the model loading code to use MBartForConditionalGeneration directly for MBART models rather than the generic AutoModelForSeq2SeqLM, addressing compatibility issues.

## Implementation Details

### 1. Tokenizer Import Fix
Added explicit imports for MBART tokenizers:
```python
# Explicitly import MBART tokenizers to ensure they're available
from transformers import MBart50TokenizerFast, MBartTokenizer
```

### 2. Model Type Detection Fix
Improved MBART model detection in TranslationModelWrapper:
```python
is_mbart = (self.model.__class__.__name__ == "MBartForConditionalGeneration" or 
            (hasattr(self.model, "config") and 
             hasattr(self.model.config, "model_type") and 
             getattr(self.model.config, "model_type", "") == "mbart") or
             "mbart" in self._get_model_type().lower())  # Check for any model type containing mbart
```

### 3. Specific Model Class Loading
Added specialized handling for MBART models:
```python
# Check if this is an MBART model based on model name or task
if "mbart" in model_config.model_name.lower() or model_config.task == "mbart_translation":
    # Use specific MBartForConditionalGeneration for MBART models
    logger.info(f"Loading MBART model: {model_config.model_name}")
    model = MBartForConditionalGeneration.from_pretrained(
        model_config.model_name, 
        **model_kwargs
    )
```

### 4. Tokenizer Loading Fix
Enhanced tokenizer loading with MBART-specific handling:
```python
# Special handling for MBART models which need specific tokenizers
if "mbart" in model_config.model_name.lower() or model_config.task == "mbart_translation":
    logger.info(f"Loading specialized MBART tokenizer for {model_config.tokenizer_name}")
    
    # Check if it's MBART-50 (many-to-many)
    if "50" in model_config.model_name or "many-to-many" in model_config.model_name:
        try:
            return MBart50TokenizerFast.from_pretrained(
                model_config.tokenizer_name, 
                **tokenizer_kwargs
            )
        except Exception:
            # Fallback to AutoTokenizer if specialized loading fails
```

## Testing

A test script (`test_mbart_loading_fix.py`) has been created to verify that MBART models now load properly. The test:

1. Initializes the model loader
2. Attempts to load the MBART translation model
3. Creates a TranslationModelWrapper with the loaded model and tokenizer
4. Tests a simple translation to verify functionality

Run the test script with:
```bash
./test_mbart_loading_fix.sh
```

## Further Improvements

Potential additional improvements that could be made:

1. Add more comprehensive error handling for MBART-specific errors
2. Implement automatic fallback to CPU for large MBART models when GPU memory is insufficient
3. Add support for multilingual token handling across different MBART variants