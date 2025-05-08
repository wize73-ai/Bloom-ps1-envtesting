# Advanced Model-Aware Translation Prompts

This document describes the advanced model-aware translation prompt system in CasaLingua, which improves translation quality through sophisticated prompt engineering tailored to each translation model's unique characteristics.

## Overview

The advanced translation prompt system uses model-specific templates, domain knowledge, formality control, and specialized handling for challenging language pairs to optimize translation quality for each specific model without changing the underlying models themselves.

## Key Benefits

- **Model-Specific Optimization**: Different prompting strategies based on each model's strengths and weaknesses
- **Domain-Specific Terminology**: Specialized vocabulary for legal, medical, technical, and other domains
- **Formality Control**: Precise control over output tone (formal, informal, neutral)
- **Context Integration**: Intelligent incorporation of context in model-appropriate ways
- **Language Pair Specialization**: Custom handling for challenging language pairs
- **Quality Assurance Prompts**: Model-aware quality hints to avoid common translation errors
- **Dynamic Parameter Tuning**: Automatically optimized generation parameters for each model and task

## Using Advanced Translation Prompts

You can use the advanced translation prompts through the standard translation API. The system is enabled by default but can be disabled if needed.

### Basic Usage

```python
from app.api.schemas.translation import TranslationRequest

# Create a translation request with domain, formality and model hints
request = TranslationRequest(
    text="El inquilino debe pagar el depósito de seguridad antes de ocupar la vivienda.",
    source_language="es",
    target_language="en",
    domain="legal",  # Specify domain for domain-specific translation
    formality="formal",  # Control formality level
    parameters={
        "enhance_prompts": True,  # Enable prompt enhancement (default is True)
        "model_name": "mbart-large-50-many-to-many-mmt"  # Provide specific model name for better optimization
    }
)
```

### Parameters

The following parameters can be used to control prompt enhancement:

| Parameter | Type | Description |
|-----------|------|-------------|
| domain | string | Domain for domain-specific vocabulary and phrasing |
| formality | string | Formality level (formal, informal, neutral) |
| context | List[string] | Context strings to inform translation |
| enhance_prompts | boolean | Enable/disable prompt enhancement (default: true) |
| use_prompt_prefix | boolean | Enable prompt prefixes for MBART models (default: false) |
| model_name | string | Specific model name for tailored optimizations |
| add_prompt_prefix | boolean | Force usage of prompt prefixes (useful for difficult language pairs) |

### Supported Domains

- **legal**: Legal documents and contracts with specialized terminology
- **housing_legal**: Housing-specific legal documents with housing terms
- **medical**: Medical texts with precise clinical terminology
- **technical**: Technical documentation with consistent technical terms
- **casual**: Casual conversation with natural, idiomatic language

### Supported Formality Levels

- **formal**: Formal, official, or professional tone with proper honorifics
- **informal**: Casual, friendly, or conversational tone with contractions
- **neutral**: Balanced, accessible tone suitable for general audiences

## Model-Specific Optimizations

The system applies different strategies based on the model:

### MBART Models

- **Instruction Style**: Minimal instructions (MBART often performs better with shorter prompts)
- **Parameter Optimization**: Higher beam count and lower temperature for formal content
- **Language Pair Handling**: Special token handling for challenging language pairs
- **Domain Strengths**: Generally better at legal, technical, and formal content

### MT5 Models

- **Instruction Style**: Detailed instructions with examples and hints
- **Parameter Optimization**: More temperature variation based on content type
- **Language Pair Handling**: More robust to language structure differences
- **Domain Strengths**: Generally better at casual, creative, and idiomatic content

## Implementation Details

The advanced translation prompt system consists of:

1. **TranslationPromptEnhancer**: Creates model-specific prompts and parameters based on detailed model profiles
2. **TranslationModelWrapper**: Integrates with the enhancer to apply model-aware prompts
3. **Model Capability Database**: Knowledge base of model strengths, weaknesses, and preferences
4. **Language Pair Proficiency Scores**: Model-specific scores for each language pair
5. **Domain Specialization Metrics**: Quantification of each model's performance in different domains
6. **Dynamic Parameter Optimization**: Automatic tuning of generation parameters based on model, language pair, and domain

## Testing the Advanced Prompts

You can test the advanced model-aware prompts using these scripts:

```bash
# Test model-specific prompt generation and parameter tuning
python scripts/test_model_specific_translation.py

# View model selection recommendations based on task
python scripts/test_model_specific_translation.py --model-selection

# Test Spanish to English enhancements
python scripts/test_enhanced_spanish_english.py
```

## Examples of Model-Specific Prompts

### MT5 Detailed Legal Instructions
```
Translate from Spanish to fluent English Avoid direct word-for-word translation Use natural English phrasing Maintain the original meaning and tone This is a legal document that requires precise translation. Maintain legal terminology. Pay special attention to key terms like: contract, agreement, party, clause, provision translate the following Spanish legal text to precise English, maintaining all legal terminology, formal tone, and document structure. Pay special attention to legal terms of art that should not be simplified: El inquilino debe pagar el depósito de seguridad antes de ocupar la vivienda.
```

### MBART Minimal Technical Instructions
```
Accurate translation Preserve meaning Natural phrasing This is technical documentation. Maintain consistent technical terminology. Watch for proper translation of: database, server, algorithm translate technical: Configure the database connection parameters in the configuration file.
```

## Model Selection Based on Task

The system can automatically select the optimal model based on the specific translation task:

| Language Pair | Domain     | Recommended Model              | 
|---------------|------------|--------------------------------|
| es-en         | legal      | mbart-large-50-many-to-many-mmt |
| es-en         | casual     | mt5-base                       |
| en-fr         | technical  | mbart-large-50-many-to-many-mmt | 
| zh-en         | medical    | mt5-base                       |

## References

- [Translation API Documentation](/docs/api/translation.md)
- [CasaLingua Translation Pipeline](/docs/architecture/pipelines.md)
- [Model Performance Metrics](/docs/models/translation-model-metrics.md)