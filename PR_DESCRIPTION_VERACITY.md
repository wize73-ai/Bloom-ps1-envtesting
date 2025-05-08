# Implement Veracity Checking System

This PR implements a comprehensive veracity checking system for CasaLingua that enables automatic quality assessment and verification of language model outputs. The system ensures translations, simplifications, and other language operations meet quality standards by detecting issues like missing information, semantic distortion, and content integrity problems.

## Summary

- Add `VeracityAuditor` class for assessing quality of model outputs
- Integrate veracity checking into the model wrapper system
- Implement comprehensive validation checks for translations and simplifications
- Add unit tests and a demonstration script for the veracity system

## Details

### New Components

1. **VeracityAuditor** (`app/audit/veracity.py`):
   - Comprehensive verification engine for language model outputs
   - Support for translation and simplification verification
   - Content integrity checks (numbers, entities, hallucinations)
   - Semantic verification using embeddings
   - Quality statistics tracking

2. **BaseModelWrapper Integration** (`app/services/models/wrapper_base.py`):
   - Automatic veracity checking for all model outputs
   - Support for both sync and async verification
   - Standardized veracity metrics and reporting
   - Graceful degradation when advanced checks aren't available

3. **Testing and Documentation**:
   - Unit tests for veracity functionality (`tests/unit/test_veracity.py`)
   - Demonstration script (`scripts/demo_veracity_checker.py`)
   - Implementation documentation (`docs/enhancements/veracity_implementation.md`)

### Key Features

- **Comprehensive Assessment**: Checks multiple aspects of output quality (semantic accuracy, content integrity, etc.)
- **Automatic Integration**: Works automatically with all model wrappers
- **Configurable Thresholds**: Adjustable sensitivity for different use cases
- **Performance Optimized**: Sampling and caching for efficient verification
- **Rich Metrics**: Detailed metrics and scores for quality analysis

### How It Works

1. When a model processes input, the wrapper automatically invokes the veracity checker
2. The veracity system performs various checks on the output
3. Results are attached to the model output as metadata
4. Applications can use this information to guarantee output quality

The system is particularly valuable for:
- Detecting errors in translations (missing numbers, entities, etc.)
- Ensuring simplified text maintains the original meaning
- Identifying potential hallucinations or content issues
- Providing quality metrics for monitoring system performance

## Testing Done

- Unit tests for all veracity checking functionality
- Integration tests with model wrapper system
- Manual testing with demonstration script for various scenarios
- Verified edge cases (empty text, identical input/output, etc.)

## Related Work

This PR builds on our previous work improving model stability and metrics reporting by adding quality verification. It complements our enhanced prompt engineering efforts by providing objective quality assessment.

## Next Steps

- Expand reference data for more accurate semantic verification
- Add more language-specific verification rules
- Implement domain-specific verification for legal, medical, etc.
- Add dashboard for monitoring quality metrics