# Veracity Checking Implementation

This document describes the implementation of the veracity checking functionality in CasaLingua. This feature provides quality assessment and verification of translations and other language operations to ensure accuracy and consistency.

## Overview

The veracity checking system consists of two main components:
1. **VeracityAuditor**: The core verification engine that assesses model outputs
2. **BaseModelWrapper Integration**: Built-in support for automatic veracity checking in all model wrappers

The system can verify:
- Translation quality and accuracy
- Text simplification appropriateness
- Missing numerical information
- Entity preservation
- Content integrity
- Semantic equivalence

## Components

### VeracityAuditor

Located in `app/audit/veracity.py`, the `VeracityAuditor` class provides comprehensive verification of language model outputs with the following features:

- **Translation Verification**: Assesses translation accuracy and completeness
- **Simplification Verification**: Evaluates the quality of text simplifications
- **Content Integrity Checks**: Verifies that important information like numbers and entities are preserved
- **Semantic Verification**: Uses embeddings to compare meaning between source and target texts
- **Quality Metrics**: Tracks and reports on quality statistics across language pairs
- **Configurable Thresholds**: Adjustable sensitivity and criteria for verification

### BaseModelWrapper Integration

The `BaseModelWrapper` class (in `app/services/models/wrapper_base.py`) includes built-in veracity checking:

- **Automatic Verification**: Every model output is automatically checked for quality
- **Synchronous and Asynchronous Support**: Both sync and async methods for different operation modes
- **Graceful Degradation**: Falls back to basic verification when advanced methods are unavailable
- **Performance Monitoring**: Tracks and reports verification performance metrics
- **Standardized Interface**: Consistent access to veracity metrics for all models

## Implementation Details

### Quality Assessment

The veracity system performs several types of checks:

1. **Basic Validation**:
   - Empty content detection
   - Untranslated content detection
   - Language character verification
   - Length ratio verification

2. **Semantic Verification**:
   - Embedding-based similarity calculation
   - Comparison with reference embeddings (when available)
   - Detection of meaning distortion

3. **Content Integrity**:
   - Numerical information preservation
   - Entity preservation
   - Detection of potential hallucinated content

### Verification Results

Each verification operation returns a comprehensive assessment including:

- **Verification Status**: Whether the output passes quality standards
- **Quality Score**: Numerical assessment (0.0-1.0) of output quality
- **Confidence**: The system's confidence in its assessment
- **Issues**: List of identified problems with severity levels
- **Metrics**: Quantitative measures of various quality aspects
- **Processing Time**: Time taken for verification

### Model Wrapper Integration

The model wrapper automatically performs verification and includes:

- **VeracityMetrics**: Standardized metrics format for consistency
- **API Integration**: Seamless integration with the wrapper's processing pipeline
- **Stability**: Enhanced error handling and recovery mechanisms
- **Metadata**: Rich metadata about verification results for analysis

## Usage Examples

### Basic Verification

```python
from app.audit.veracity import VeracityAuditor

# Create and initialize the auditor
auditor = VeracityAuditor()
await auditor.initialize()

# Verify a translation
result = await auditor.verify_translation(
    source_text="I have 5 apples.",
    translation="Tengo 5 manzanas.",
    source_lang="en",
    target_lang="es"
)

# Check verification result
if result["verified"]:
    print("Translation passed verification!")
else:
    print("Issues found:", result["issues"])
```

### Integration with Model Wrapper

The veracity checking is automatically performed when using model wrappers:

```python
from app.services.models.manager import ModelManager
from app.services.models.wrapper_base import ModelInput

# Load model
model_manager = ModelManager()
model = await model_manager.load_model("translation_model")

# Process text with automatic verification
input_data = ModelInput(
    text="Hello, how are you?",
    source_language="en",
    target_language="es"
)

result = await model.process_async(input_data)

# Access veracity information
veracity_score = result["veracity_score"]
veracity_data = result["metadata"]["veracity"]

print(f"Translation quality score: {veracity_score}")
if veracity_data["checks_failed"]:
    print("Failed checks:", veracity_data["checks_failed"])
```

## Performance Considerations

- **Resource Usage**: Semantic verification is more resource-intensive and may be skipped for high-volume operations
- **Caching**: Reference embeddings are cached for better performance
- **Sampling**: For long texts, samples are taken to maintain performance while ensuring quality assessment
- **Asynchronous Operation**: Async methods allow verification to run in parallel with other operations

## Configuration

The veracity system can be configured through the application configuration system:

```json
{
  "veracity": {
    "enabled": true,
    "threshold": 0.75,
    "max_sample_size": 1000,
    "min_confidence": 0.7,
    "reference_embeddings_path": "/path/to/embeddings.json",
    "language_pairs": [
      ["en", "es"],
      ["en", "fr"]
    ]
  }
}
```

## Testing

Comprehensive test coverage for veracity functionality is available in:
- `tests/unit/test_veracity.py`

A demonstration script is available at:
- `scripts/demo_veracity_checker.py`