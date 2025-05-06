# Model Reliability Fixes for CasaLingua

This document outlines the issues and fixes implemented to improve model reliability in the CasaLingua API service.

## Issues Identified

1. **Missing Model Configuration in Registry**:
   - The code was referring to "mbart_translation" when loading translation models
   - However, "mbart_translation" was not defined in the model registry
   - This caused the system to fall back to a generic MT5 model configuration
   - Log message: "⚠️ WARNING - No model configuration found for 'mbart_translation', using generic fallback"

2. **Model Persistence Issues**:
   - Models were being unloaded too quickly due to a short cleanup interval (300 seconds)
   - This forced re-loading of models for each new request
   - The "mbart_translation" model was not included in preload_models list

3. **Inconsistent Model Selection**:
   - Different parts of the codebase referenced different model names for the same functionality
   - Some parts referred to "translation" while others used "mbart_translation"

## Fixes Implemented

1. **Added Missing Model Configuration**:
   - Added "mbart_translation" configuration to the model registry
   - Used same settings as "translation" model to ensure consistency
   - Specified the model class, tokenizer, and task correctly

```json
"mbart_translation": {
    "model_name": "facebook/mbart-large-50-many-to-many-mmt",
    "tokenizer_name": "facebook/mbart-large-50-many-to-many-mmt",
    "task": "mbart_translation",
    "type": "transformers",
    "model_class": "AutoModelForSeq2SeqLM",
    "framework": "transformers",
    "is_primary": true,
    "tokenizer_kwargs": {
        "src_lang": "en_XX",
        "tgt_lang": "es_XX"
    }
}
```

2. **Improved Model Persistence**:
   - Increased model_cleanup_interval from 300s to 3600s (1 hour)
   - Added "mbart_translation" to the preload_models list
   - This ensures the model stays loaded between requests

3. **Ensured Consistent Model Loading**:
   - Kept existing references to both "translation" and "mbart_translation"
   - Made them point to the same model definition for consistency
   - This approach avoids changing existing code while ensuring reliable operation

## Expected Benefits

1. **Reliable Model Loading**:
   - No more fallbacks to the generic MT5 model
   - Proper loading of the MBART model with correct configuration

2. **Improved Performance**:
   - Reduced latency by keeping models loaded between requests
   - No need to reload the model for each request
   - Better translation quality by using the intended model

3. **Reduced Resource Usage**:
   - Fewer model loads/unloads means less CPU and memory churn
   - Fewer token generations due to more reliable processing

## Testing Strategy

1. **Translation Endpoint Testing**:
   - Test single translation with simple inputs
   - Test batch translation with multiple inputs
   - Test with various language pairs

2. **Performance Monitoring**:
   - Monitor model loading logs to ensure proper model selection
   - Verify the model is kept loaded between requests
   - Check for improved latency in successive requests

3. **Load Testing**:
   - Test under concurrent load to ensure model remains stable
   - Verify performance doesn't degrade over time