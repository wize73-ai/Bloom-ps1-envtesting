# Transformer Import Fix

## Issue

The application was experiencing an error during model loading with the following exception:

```
UnboundLocalError: local variable 'AutoModel' referenced before assignment
```

This error occurred because the model loader was trying to use `AutoModel` and other transformer classes even when the `transformers` library wasn't available, despite having a flag (`HAVE_TRANSFORMERS`) to check for this condition.

## Analysis

The model loader was designed to handle missing dependencies by setting classes like `AutoModel` to `None` when the `transformers` library couldn't be imported:

```python
try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
    # ... other imports
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    # Define empty classes to avoid NameError when referenced
    AutoModel = None
    AutoTokenizer = None
    # ... other classes set to None
```

However, the code wasn't properly checking the `HAVE_TRANSFORMERS` flag before using these classes in various load methods:

- `_load_transformers_model`
- `_load_tokenizer`
- `_load_sentence_transformer`
- `_load_onnx_model`

This resulted in a `UnboundLocalError` when trying to reference these `None` variables as if they were actual classes.

## Solution

The fix has three main components:

1. **Improved import handling at the module level**: We now only import the library, but don't create placeholder `None` variables for missing classes:

```python
try:
    import transformers
    HAVE_TRANSFORMERS = True
    # Import all necessary classes here so they're available in the global scope
    from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
    # ... other imports
except ImportError:
    HAVE_TRANSFORMERS = False
    logger.warning("transformers not available - model loading will be limited")
    # Don't define empty classes here, we'll check HAVE_TRANSFORMERS before usage
```

2. **Early check for dependencies in loading methods**: Each method now checks for the required dependencies before attempting to use them:

```python
def _load_transformers_model(self, model_config: ModelConfig, device: str) -> Any:
    # First check if transformers is available
    if not HAVE_TRANSFORMERS:
        raise ImportError("transformers library is required for loading models")

    # Import necessary components to ensure they're available
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM,
        # ... other classes
    )
    
    # Rest of the method...
```

3. **Local imports within loading methods**: Instead of relying on module-level imports that might not be available, we import necessary classes directly within each loading method scope:

```python
def _load_tokenizer(self, model_config: ModelConfig) -> Any:
    # First check if transformers is available
    if not HAVE_TRANSFORMERS:
        raise ImportError("transformers library is required for loading tokenizers")
        
    # Import necessary tokenizer classes
    from transformers import AutoTokenizer, MBart50TokenizerFast, MBartTokenizer
    
    # Rest of the method...
```

This approach ensures that:
- Each method can only be called if the required dependencies are available 
- The model loader safely handles missing dependencies without crashing
- The code structure provides clear error messages about missing dependencies
- The model loading process is more robust with proper dependency checks

## Testing

A test script (`test_transformers_fix.py`) has been created to verify the fix. It initializes the model loader and attempts to load different model types to ensure everything works correctly with the new import structure.

## Summary

This fix addresses the `UnboundLocalError` by properly checking for the availability of dependencies before trying to use them and using local imports within methods for better isolation and error handling. It improves the robustness of the model loading process and provides clearer error messages when dependencies are missing.