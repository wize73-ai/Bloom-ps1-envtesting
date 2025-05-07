# Enhanced Models Implementation

This document describes the implementation of enhanced model prompting capabilities for language detection and simplification models, as well as fixes for veracity and audit score reporting issues.

## Overview

The implementation enhances the CasaLingua system by:

1. Adding model-specific prompt engineering for language detector models
2. Adding multi-level simplification with model-specific prompt optimization
3. Fixing issues with veracity auditing and metrics reporting
4. Adding comprehensive testing tools for the enhanced functionality

## Language Detector Enhancements

### LanguageDetectorPromptEnhancer

The `LanguageDetectorPromptEnhancer` class provides model-specific prompt engineering for language detection models. It optimizes prompts based on model capabilities:

```python
class LanguageDetectorPromptEnhancer:
    """
    Enhances language detection prompts with model-specific optimizations to
    improve accuracy and performance.
    """
    
    def __init__(self):
        # Model capability profiles
        self.model_capabilities = {
            "mt5": {
                "instruction_style": "simple",
                "confidence_support": True,
                "instruction_format": "prefix",
                "handles_code_mixed": False,
                "prefers_examples": True,
                "language_coverage": "high",
            },
            "mbart": { ... },
            "gpt": { ... },
            # Additional models...
        }
```

Key features:

- Model-specific prompt templates
- Adaptive instruction complexity
- Confidence scoring optimizations
- Support for ambiguous and code-mixed text
- Language-specific features to improve detection

### Integration with Language Detector

The enhanced prompt system has been integrated with the existing `LanguageDetector` class:

```python
async def detect_language(self, text: str, detailed: bool = False) -> Dict[str, Any]:
    # ...existing code...
    
    # Import prompt enhancer if available
    try:
        from app.services.models.language_detector_prompt_enhancer import LanguageDetectorPromptEnhancer
        prompt_enhancer = LanguageDetectorPromptEnhancer()
        enhanced_prompt = True
    except ImportError:
        enhanced_prompt = False
        logger.debug("Language detector prompt enhancer not available")
    
    # Enhance prompt if available
    if enhanced_prompt:
        # Get model info to retrieve model ID
        model_info = await self.model_manager.get_model_info(self.model_type)
        model_id = model_info.get("model_id", "default") if model_info else "default"
        
        # Enhance prompt with model-specific optimization
        enhanced_data = prompt_enhancer.enhance_prompt(text, model_id, {"detailed": detailed})
        
        # Update input data with enhanced prompt
        if "prompt" in enhanced_data:
            input_data["prompt"] = enhanced_data["prompt"]
        
        # Update parameters with optimized values
        if "parameters" in enhanced_data:
            input_data["parameters"].update(enhanced_data["parameters"])
    
    # ...continue processing...
```

## Simplification Enhancements

### SimplifierPromptEnhancer

The `SimplifierPromptEnhancer` class provides model-specific prompt engineering for text simplification models:

```python
class SimplifierPromptEnhancer:
    """
    Enhances text simplification prompts with model-specific optimizations to
    improve quality across different simplification levels and domains.
    """
    
    def __init__(self):
        # Model capability profiles
        self.model_capabilities = {
            "mt5": {
                "instruction_style": "concise",
                "level_granularity": "coarse",
                "instruction_format": "prefix",
                "prefers_examples": True,
                "domain_support": "limited",
                "handles_formatting": False,
                "readability_metrics": False
            },
            # Additional models...
        }
        
        # Simplification level descriptions
        self.level_descriptions = {
            1: {
                "name": "Academic",
                "description": "Minimal simplification for academic/professional audiences",
                "grade_level": 12,
                "features": [ ... ]
            },
            # Additional levels...
        }
        
        # Domain-specific simplification strategies
        self.domain_strategies = {
            "legal": {
                "focus_areas": [ ... ],
                "key_terms": { ... }
            },
            # Additional domains...
        }
```

Key features:

- Multi-level simplification targeting (5 levels)
- Domain-specific simplification strategies (legal, medical, technical, financial, educational)
- Grade-level targeting with educational guidelines
- Model-specific prompt optimization
- Readability metrics integration

### Integration with Simplification Pipeline

The enhanced prompt system has been integrated with the existing `SimplificationPipeline` class:

```python
async def simplify(self, text: str, language: str, level: int = 3, grade_level: Optional[int] = None, options: Dict[str, Any] = None) -> Dict[str, Any]:
    # ...existing code...
    
    # Import prompt enhancer if available
    try:
        from app.services.models.simplifier_prompt_enhancer import SimplifierPromptEnhancer
        prompt_enhancer = SimplifierPromptEnhancer()
        enhanced_prompt = True
    except ImportError:
        enhanced_prompt = False
        logger.debug("Simplifier prompt enhancer not available")
    
    # Enhance prompt if available
    if enhanced_prompt:
        # Enhance prompt with model-specific optimization
        enhanced_data = prompt_enhancer.enhance_prompt(
            text=text,
            model_name=model_id,
            level=level,
            grade_level=target_grade,
            language=language,
            domain=options.get("domain"),
            preserve_formatting=options.get("preserve_formatting", True)
        )
        
        # Update input data with enhanced prompt
        if "prompt" in enhanced_data:
            input_data["prompt"] = enhanced_data["prompt"]
        
        # Update parameters with optimized values
        if "parameters" in enhanced_data:
            input_data["parameters"].update(enhanced_data["parameters"])
    
    # ...continue processing...
```

## Metrics and Audit Fixes

### EnhancedMetricsCollector

The `EnhancedMetricsCollector` class fixes issues with veracity and audit score reporting:

```python
class EnhancedMetricsCollector(MetricsCollector):
    """
    Enhanced metrics collector that ensures proper reporting of
    veracity and audit scores by fixing inconsistencies in the metrics system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced metrics collector."""
        super().__init__(config)
        self.veracity_metrics = {}
        
    async def install_hooks(self):
        """Install hooks to ensure metrics are properly reported."""
        # ...hook installation code...
```

Key features:

- Hooks into the existing metrics system to ensure proper reporting
- Tracks veracity metrics for translations and simplifications
- Ensures audit logs are correctly generated
- Provides additional metrics for quality assessment

### Setup Function

A helper function to set up the enhanced metrics collector:

```python
async def setup_enhanced_metrics() -> bool:
    """
    Setup the enhanced metrics collector.
    
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Create enhanced metrics collector
        enhanced_metrics = EnhancedMetricsCollector()
        
        # Install hooks
        result = await enhanced_metrics.install_hooks()
        
        # Log result
        if result:
            logger.info("Enhanced metrics collector successfully setup")
        else:
            logger.warning("Failed to setup enhanced metrics collector")
            
        return result
    except Exception as e:
        logger.error(f"Error setting up enhanced metrics collector: {str(e)}")
        return False
```

## Test Scripts

### Language Detection Test

The `test_enhanced_language_detection.py` script tests the enhanced language detection system:

```python
async def test_language_detection(text: str, model_name: Optional[str] = None, detailed: bool = False) -> Dict[str, Any]:
    """Test the enhanced language detection system."""
    # ...implementation...
```

### Simplification Test

The `test_enhanced_simplification.py` script tests the enhanced simplification system:

```python
async def test_simplification(text: str, level: int = 3, language: str = "en", domain: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Test the enhanced text simplification system."""
    # ...implementation...

async def compare_simplification_levels(text: str, language: str = "en", domain: Optional[str] = None):
    """Compare simplification across all 5 levels."""
    # ...implementation...

async def test_domain_simplification():
    """Test simplification for different domains."""
    # ...implementation...
```

### Metrics and Audit Fix Test

The `test_metrics_audit_fix.py` script tests the fixes to the metrics and audit system:

```python
async def test_metrics_system(fix_enabled: bool = True) -> Dict[str, Any]:
    """Test the metrics system with or without the fix."""
    # ...implementation...

async def test_audit_logger() -> Dict[str, Any]:
    """Test the audit logger system."""
    # ...implementation...

async def test_veracity_auditor() -> Dict[str, Any]:
    """Test the veracity auditor system."""
    # ...implementation...

async def compare_metrics(save_results: bool = False) -> Dict[str, Any]:
    """Compare metrics with and without the fix."""
    # ...implementation...
```

## Integration and Usage

### Adding Enhanced Prompting to Your Pipeline

To use the enhanced prompting system in your own code:

```python
# For language detection
from app.services.models.language_detector_prompt_enhancer import LanguageDetectorPromptEnhancer

# Create enhancer
prompt_enhancer = LanguageDetectorPromptEnhancer()

# Enhance prompt for a specific model
enhanced_data = prompt_enhancer.enhance_prompt(
    text="Text to detect language",
    model_name="your-model-name",
    parameters={"detailed": True}
)

# For simplification
from app.services.models.simplifier_prompt_enhancer import SimplifierPromptEnhancer

# Create enhancer
prompt_enhancer = SimplifierPromptEnhancer()

# Enhance prompt for a specific model
enhanced_data = prompt_enhancer.enhance_prompt(
    text="Text to simplify",
    model_name="your-model-name",
    level=3,
    grade_level=8,
    language="en",
    domain="legal",
    preserve_formatting=True
)
```

### Enabling Metrics Fixes

To enable the metrics and audit fixes in your application:

```python
from app.audit.metrics_fix import setup_enhanced_metrics

# Setup enhanced metrics at application startup
async def startup():
    # Other initialization code...
    
    # Setup enhanced metrics
    await setup_enhanced_metrics()
    
    # Continue with startup...
```

## Benefits and Improvements

1. **Improved Language Detection:**
   - Higher accuracy for language detection across all models
   - Better handling of code-mixed text
   - More reliable confidence scores

2. **Enhanced Simplification:**
   - Consistent progression across simplification levels
   - Domain-specific simplification with specialized terminology handling
   - Grade-level targeting for educational content

3. **Fixed Metrics and Audit Reporting:**
   - Veracity scores now properly reported
   - Audit logs correctly generated for all operations
   - Comprehensive metrics for assessing quality

4. **Better Testing and Validation:**
   - Dedicated test scripts for verifying enhancements
   - Comparison tools to validate improvements
   - Documentation of the enhanced functionality

## Conclusion

The enhanced model prompting system and metrics fixes significantly improve the quality and reliability of the CasaLingua language processing pipeline. The model-specific approach ensures optimal performance across different model types, while the fixes to the metrics and audit system ensure proper reporting and tracking of system performance.