"""
Translation Prompt Enhancement Module for CasaLingua

This module provides sophisticated model-aware prompting to enhance translation quality
through advanced prompt engineering, model specialization, and context integration.

It provides:
1. Model-specific prompt templates and strategies
2. Domain-specific translation guidance
3. Formality-aware prompting
4. Context integration strategies
5. Special handling for problematic language pairs
6. Quality assurance hints tailored to each model's strengths and weaknesses
7. Dynamic parameter optimization based on model characteristics
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Configure logging
logger = logging.getLogger(__name__)

class TranslationPromptEnhancer:
    """
    Enhances prompts for translation models to improve quality
    through model-aware prompt engineering, context integration,
    and domain-specific instructions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the translation prompt enhancer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load model proficiencies if available
        self.model_proficiencies = self._load_model_proficiencies()
        
        # Track known models and their capabilities
        self.model_capabilities = {
            "mbart-large-50-many-to-many-mmt": {
                "strengths": ["formal_text", "longer_sentences", "european_languages"],
                "weaknesses": ["asian_languages", "creative_text", "idiomatic_expressions"],
                "preferred_domains": ["legal", "technical", "medical"],
                "instruction_style": "minimal",  # MBART often performs better with minimal instructions
                "max_prompt_tokens": 50  # MBART usually doesn't benefit from very long prompts
            },
            "mt5-base": {
                "strengths": ["casual_text", "short_sentences", "instruction_following"],
                "weaknesses": ["legal_terminology", "complex_sentences"],
                "preferred_domains": ["casual", "general"],
                "instruction_style": "detailed",  # MT5 benefits from detailed instructions
                "max_prompt_tokens": 200  # MT5 can use longer prompts effectively
            },
            "mt5-small": {
                "strengths": ["casual_text", "short_sentences", "instruction_following"],
                "weaknesses": ["legal_terminology", "complex_sentences", "rare_vocabulary"],
                "preferred_domains": ["casual", "general"],
                "instruction_style": "detailed",
                "max_prompt_tokens": 150
            },
            "default": {
                "strengths": ["general_translation"],
                "weaknesses": [],
                "preferred_domains": ["general"],
                "instruction_style": "balanced",
                "max_prompt_tokens": 100
            }
        }
        
        # Language pair difficulty ratings (higher = more difficult)
        self.language_pair_difficulty = {
            "en-es": 2,  # Relatively straightforward
            "es-en": 3,  # More challenging due to structure differences
            "en-fr": 2,
            "fr-en": 2,
            "en-de": 3,  # More challenging due to compound words, word order
            "de-en": 3,
            "en-zh": 5,  # Very different language structures
            "zh-en": 5,
            "en-ja": 5,
            "ja-en": 5,
            "en-ar": 4,  # Different script, right-to-left
            "ar-en": 4,
            "default": 3
        }
        
        # Domain-specific prompt templates with model-specific variations
        self.domain_templates = {
            "legal": {
                "mt5": "translate {source_lang} legal text to {target_lang} preserving legal terminology and formal tone: {text}",
                "mt5-detailed": "translate the following {source_lang} legal text to precise {target_lang}, maintaining all legal terminology, formal tone, and document structure. Pay special attention to legal terms of art that should not be simplified: {text}",
                "mbart": "translate {source_lang} legal document to {target_lang} maintaining legal terms",
                "mbart-minimal": "translate legal: {text}",
                "prefix": "This is a legal document that requires precise translation. Maintain legal terminology.",
                "suffix": "Ensure all legal terms are translated accurately.",
                "key_terms": ["contract", "agreement", "party", "clause", "provision", "termination", "liability", "jurisdiction", "tenant", "landlord"]
            },
            "medical": {
                "mt5": "translate {source_lang} medical text to {target_lang} preserving medical terminology: {text}",
                "mt5-detailed": "translate the following {source_lang} medical text to accurate {target_lang}, preserving all medical terminology, anatomical references, and technical precision. Medical terms should retain their clinical meaning: {text}",
                "mbart": "translate {source_lang} medical document to {target_lang} with precise terminology",
                "mbart-minimal": "translate medical: {text}",
                "prefix": "This is a medical text. Preserve medical terminology and maintain accuracy.",
                "suffix": "Ensure medical terms are translated with technical precision.",
                "key_terms": ["diagnosis", "treatment", "symptom", "prognosis", "chronic", "acute", "condition", "medication", "dosage", "therapy"]
            },
            "technical": {
                "mt5": "translate {source_lang} technical documentation to {target_lang} keeping technical terms consistent: {text}",
                "mt5-detailed": "translate the following {source_lang} technical documentation to {target_lang}, maintaining consistent terminology throughout. Keep format markers and code references unchanged. Technical terms should have the same meaning in the target language: {text}",
                "mbart": "translate {source_lang} technical text to {target_lang} with consistent terminology",
                "mbart-minimal": "translate technical: {text}",
                "prefix": "This is technical documentation. Maintain consistent technical terminology.",
                "suffix": "Ensure technical terms remain consistent throughout.",
                "key_terms": ["database", "server", "algorithm", "software", "hardware", "interface", "protocol", "configuration", "system", "network"]
            },
            "casual": {
                "mt5": "translate {source_lang} casual conversation to {target_lang} with natural, conversational tone: {text}",
                "mt5-detailed": "translate the following {source_lang} casual conversation to {target_lang} using a natural, conversational tone. Preserve humor, informality, and the personal style of the speaker: {text}",
                "mbart": "translate {source_lang} casual text to {target_lang} with informal tone",
                "mbart-minimal": "translate casual: {text}",
                "prefix": "This is casual conversation. Use natural, conversational language.",
                "suffix": "Keep the translation natural and conversational.",
                "key_terms": ["hey", "thanks", "cool", "awesome", "yeah", "what's up", "hang out", "grab", "catch up", "see you"]
            },
            "housing_legal": {
                "mt5": "translate {source_lang} housing legal document to {target_lang} with appropriate housing terminology: {text}",
                "mt5-detailed": "translate the following {source_lang} housing legal document to {target_lang} with precise housing-specific terminology. Preserve references to housing regulations, tenant rights, and property conditions: {text}",
                "mbart": "translate {source_lang} housing legal document to {target_lang} precisely",
                "mbart-minimal": "translate housing legal: {text}",
                "prefix": "This is a housing legal document. Use appropriate housing and legal terminology.",
                "suffix": "Maintain housing-specific legal terminology throughout.",
                "key_terms": ["tenant", "landlord", "lease", "rent", "security deposit", "eviction", "occupancy", "premises", "property", "housing"]
            }
        }
        
        # Formality levels with model-specific variations
        self.formality_templates = {
            "formal": {
                "mt5": "translate {source_lang} to formal {target_lang}: {text}",
                "mt5-detailed": "translate the following {source_lang} text to formal {target_lang}, using proper honorifics, avoiding contractions, and maintaining a professional, respectful tone throughout: {text}",
                "mbart": "translate {source_lang} to {target_lang} with formal register",
                "mbart-minimal": "translate formal: {text}",
                "prefix": "Use formal language and honorifics where appropriate.",
                "suffix": "Ensure the translation maintains a formal tone.",
                "register_markers": ["hereby", "aforementioned", "pursuant to", "shall", "must", "kindly", "respectfully"]
            },
            "informal": {
                "mt5": "translate {source_lang} to informal {target_lang}: {text}",
                "mt5-detailed": "translate the following {source_lang} text to informal, friendly {target_lang}, using contractions, everyday expressions, and a conversational tone: {text}",
                "mbart": "translate {source_lang} to {target_lang} with informal register",
                "mbart-minimal": "translate informal: {text}",
                "prefix": "Use casual, everyday language in this translation.",
                "suffix": "Keep the tone friendly and informal.",
                "register_markers": ["hey", "yeah", "cool", "gonna", "wanna", "kinda", "you know", "like", "stuff"]
            },
            "neutral": {
                "mt5": "translate {source_lang} to neutral {target_lang}: {text}",
                "mt5-detailed": "translate the following {source_lang} text to neutral {target_lang}, avoiding overly formal or casual language, maintaining a balanced, accessible tone suitable for general audiences: {text}",
                "mbart": "translate {source_lang} to {target_lang} with neutral tone",
                "mbart-minimal": "translate neutral: {text}",
                "prefix": "Use neutral, balanced language for this translation.",
                "suffix": "Maintain a neutral tone throughout.",
                "register_markers": ["approximately", "generally", "typically", "often", "may", "should", "could"]
            }
        }
        
        # Quality checklist templates, customized by model and language pair
        self.quality_hints = {
            "es-en": {
                "mt5": [
                    "Translate from Spanish to fluent English",
                    "Avoid direct word-for-word translation",
                    "Use natural English phrasing",
                    "Maintain the original meaning and tone"
                ],
                "mbart": [
                    "Spanish to natural English",
                    "Avoid literal translation",
                    "Keep meaning intact"
                ]
            },
            "en-es": {
                "mt5": [
                    "Translate from English to fluent Spanish",
                    "Pay attention to gender agreement in Spanish",
                    "Use appropriate verb tenses",
                    "Maintain the original meaning and tone"
                ],
                "mbart": [
                    "English to proper Spanish",
                    "Mind gender agreement",
                    "Correct verb conjugation"
                ]
            },
            "default": {
                "mt5": [
                    "Translate accurately while maintaining natural language flow",
                    "Preserve the original meaning, tone, and intent",
                    "Adapt cultural references appropriately"
                ],
                "mbart": [
                    "Accurate translation",
                    "Preserve meaning",
                    "Natural phrasing"
                ]
            }
        }
        
        # Special handling for problematic language pairs with model-specific strategies
        self.special_handling = {
            "es-en": {
                "mt5": "translate Spanish to clear, natural English avoiding common translation pitfalls: {text}",
                "mt5-detailed": "translate the following Spanish text to clear, natural English, avoiding overly literal translation, false cognates, and maintaining proper English syntax. Pay special attention to idiomatic expressions: {text}",
                "mbart": "translate Spanish to English with natural phrasing",
                "mbart-minimal": "translate Spanish to idiomatic English: {text}",
                "prefix": "Translate this Spanish text to natural, fluent English. Avoid literal translations. Use idiomatic English expressions where appropriate.",
                "suffix": "Ensure the English translation sounds natural, not translated.",
                "check_terms": ["muy", "que", "hoy", "bien", "estar", "tener", "hacer"], # Terms to check for direct translation
                "example_pairs": [
                    {"source": "Estoy muy feliz", "target": "I am very happy"},
                    {"source": "Tengo 30 años", "target": "I am 30 years old"},
                    {"source": "Hace calor", "target": "It's hot"}
                ]
            },
            "zh-en": {
                "mt5": "translate Chinese to clear, grammatical English with proper articles and plurals: {text}",
                "mt5-detailed": "translate the following Chinese text to clear, grammatical English, paying special attention to articles (a/an/the), plural forms, and proper English word order. Chinese lacks these features, so they must be added correctly: {text}",
                "mbart": "translate Chinese to English with proper grammar",
                "mbart-minimal": "translate Chinese to grammatical English: {text}",
                "prefix": "Translate this Chinese text to proper English. Add articles and plural forms correctly. Use standard English sentence structure.",
                "suffix": "Ensure the English translation has correct grammar and natural phrasing."
            }
        }
        
        # Model-specific default generation parameters for different language pairs
        self.model_default_params = {
            "mbart-large-50-many-to-many-mmt": {
                "es-en": {
                    "num_beams": 8,
                    "do_sample": False,
                    "length_penalty": 1.0,
                    "early_stopping": True
                },
                "en-es": {
                    "num_beams": 6,
                    "do_sample": False,
                    "length_penalty": 1.0
                },
                "default": {
                    "num_beams": 4,
                    "length_penalty": 1.0
                }
            },
            "mt5-base": {
                "es-en": {
                    "num_beams": 5,
                    "do_sample": True,
                    "temperature": 0.8,
                    "top_p": 0.9
                },
                "default": {
                    "num_beams": 4,
                    "temperature": 0.9,
                    "top_p": 0.9
                }
            },
            "default": {
                "default": {
                    "num_beams": 4,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        }
        
        logger.info("Advanced translation prompt enhancer initialized")
        
    def _load_model_proficiencies(self) -> Dict[str, Dict[str, Any]]:
        """
        Load model proficiency data from config or default to built-in values.
        
        Returns:
            Dictionary of model proficiencies
        """
        # Check if we have a custom model proficiency file
        proficiency_path = self.config.get("model_proficiency_path")
        
        if proficiency_path and os.path.exists(proficiency_path):
            try:
                with open(proficiency_path, 'r', encoding='utf-8') as f:
                    proficiencies = json.load(f)
                logger.info(f"Loaded model proficiencies from {proficiency_path}")
                return proficiencies
            except Exception as e:
                logger.error(f"Error loading model proficiencies: {str(e)}")
        
        # Default model proficiencies (quality scores by language pair)
        # Higher score = better quality (0-10 scale)
        return {
            "mbart-large-50-many-to-many-mmt": {
                "en-es": 8.5,
                "es-en": 8.2,
                "en-fr": 8.7,
                "fr-en": 8.6,
                "en-de": 8.3,
                "de-en": 8.4,
                "en-zh": 7.0,
                "zh-en": 7.2,
                "en-ru": 7.8,
                "ru-en": 7.9,
                "en-ar": 7.1,
                "ar-en": 7.0
            },
            "mt5-base": {
                "en-es": 8.0,
                "es-en": 8.3,
                "en-fr": 8.1,
                "fr-en": 8.2,
                "en-de": 7.8,
                "de-en": 7.9,
                "en-zh": 7.5,
                "zh-en": 7.6,
                "en-ru": 7.4,
                "ru-en": 7.5,
                "en-ar": 7.2,
                "ar-en": 7.3
            }
        }
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """
        Get capability information for a specific model.
        
        Args:
            model_name: The model name to look up
            
        Returns:
            Dictionary of model capabilities
        """
        # Look for exact match
        if model_name in self.model_capabilities:
            return self.model_capabilities[model_name]
        
        # Look for partial match
        for model_key in self.model_capabilities:
            if model_key in model_name:
                return self.model_capabilities[model_key]
        
        # Fall back to default
        return self.model_capabilities["default"]
    
    def get_model_instruction_style(self, model_name: str) -> str:
        """
        Get the optimal instruction style for a model.
        
        Args:
            model_name: The model name to look up
            
        Returns:
            Instruction style (minimal, detailed, balanced)
        """
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get("instruction_style", "balanced")
    
    def get_language_pair_proficiency(self, model_name: str, source_lang: str, target_lang: str) -> float:
        """
        Get the proficiency score for a model and language pair.
        
        Args:
            model_name: The model name
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Proficiency score (0-10 scale)
        """
        lang_pair = f"{source_lang}-{target_lang}"
        
        # Check if we have this model and language pair in proficiencies
        if model_name in self.model_proficiencies:
            model_scores = self.model_proficiencies[model_name]
            if lang_pair in model_scores:
                return model_scores[lang_pair]
        
        # Look for partial match on model name
        for model_key in self.model_proficiencies:
            if model_key in model_name:
                model_scores = self.model_proficiencies[model_key]
                if lang_pair in model_scores:
                    return model_scores[lang_pair]
        
        # Return default proficiency estimate
        return 7.0  # Reasonable default score
    
    def get_domain_quality_for_model(self, model_name: str, domain: str) -> float:
        """
        Estimate how well a model handles a specific domain.
        
        Args:
            model_name: Model name
            domain: Domain name (legal, medical, etc.)
            
        Returns:
            Quality score (0-10 scale)
        """
        capabilities = self.get_model_capabilities(model_name)
        preferred_domains = capabilities.get("preferred_domains", [])
        
        # If this domain is in the model's preferred domains, higher score
        if domain in preferred_domains:
            return 9.0
        
        # Domain-specific scoring based on model capabilities
        if "mbart" in model_name.lower():
            domain_scores = {
                "legal": 8.5,
                "medical": 8.2,
                "technical": 8.7,
                "housing_legal": 8.4,
                "casual": 7.5
            }
        elif "mt5" in model_name.lower():
            domain_scores = {
                "legal": 7.8,
                "medical": 7.6,
                "technical": 8.0,
                "housing_legal": 7.7,
                "casual": 8.5
            }
        else:
            domain_scores = {
                "legal": 7.5,
                "medical": 7.5,
                "technical": 7.5,
                "housing_legal": 7.5,
                "casual": 7.5
            }
        
        # Return domain score or default
        return domain_scores.get(domain, 7.5)
    
    def enhance_mt5_prompt(self, 
                           text: str, 
                           source_lang: str, 
                           target_lang: str,
                           domain: Optional[str] = None,
                           formality: Optional[str] = None,
                           context: Optional[List[str]] = None,
                           parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an enhanced MT5 prompt with model-aware instructions and context.
        
        Args:
            text: The text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain for domain-specific translations
            formality: Optional formality level
            context: Optional context for the translation
            parameters: Additional parameters
            
        Returns:
            Enhanced MT5 prompt
        """
        parameters = parameters or {}
        
        # Get model name and instruction style preference
        model_name = parameters.get("model_name", "mt5")
        instruction_style = self.get_model_instruction_style(model_name)
        
        # Create language pair key
        lang_pair = f"{source_lang}-{target_lang}"
        lang_pair_difficulty = self.language_pair_difficulty.get(lang_pair, self.language_pair_difficulty["default"])
        
        # Start with basic template
        base_template = "translate {source_lang} to {target_lang}: {text}"
        
        # Choose template based on model's instruction style preference
        if instruction_style == "detailed" and domain and domain in self.domain_templates:
            # Use detailed template if available
            if "mt5-detailed" in self.domain_templates[domain]:
                base_template = self.domain_templates[domain]["mt5-detailed"]
            else:
                base_template = self.domain_templates[domain]["mt5"]
        elif domain and domain in self.domain_templates:
            base_template = self.domain_templates[domain]["mt5"]
        
        # Apply formality template with instruction style consideration
        if formality and formality in self.formality_templates:
            if instruction_style == "detailed" and "mt5-detailed" in self.formality_templates[formality]:
                base_template = self.formality_templates[formality]["mt5-detailed"]
            else:
                base_template = self.formality_templates[formality]["mt5"]
            
        # Apply special handling for problematic language pairs with instruction style consideration
        if lang_pair in self.special_handling:
            if instruction_style == "detailed" and "mt5-detailed" in self.special_handling[lang_pair]:
                base_template = self.special_handling[lang_pair]["mt5-detailed"]
            else:
                base_template = self.special_handling[lang_pair]["mt5"]
        
        # Create prefix with quality hints, using model-specific hints when available
        prefix = []
        if lang_pair in self.quality_hints:
            # Use model-specific quality hints if available
            if "mt5" in self.quality_hints[lang_pair]:
                prefix.extend(self.quality_hints[lang_pair]["mt5"])
            else:
                prefix.extend(self.quality_hints[lang_pair])
        else:
            # Use default quality hints
            if "mt5" in self.quality_hints["default"]:
                prefix.extend(self.quality_hints["default"]["mt5"])
            else:
                prefix.extend(self.quality_hints["default"])
        
        # Add domain-specific prefix
        if domain and domain in self.domain_templates:
            prefix.append(self.domain_templates[domain]["prefix"])
            
            # If the model needs detailed instructions, add domain key terms guidance
            if instruction_style == "detailed" and "key_terms" in self.domain_templates[domain]:
                key_terms = self.domain_templates[domain]["key_terms"]
                if key_terms:
                    terms_str = ", ".join(key_terms[:5])  # Don't use too many terms
                    prefix.append(f"Pay special attention to key terms like: {terms_str}")
        
        # Add formality-specific prefix with register markers for detailed instructions
        if formality and formality in self.formality_templates:
            prefix.append(self.formality_templates[formality]["prefix"])
            
            # Add register markers for detailed instructions
            if instruction_style == "detailed" and "register_markers" in self.formality_templates[formality]:
                markers = self.formality_templates[formality]["register_markers"]
                if markers:
                    markers_str = ", ".join(markers[:3])  # Just a few examples
                    prefix.append(f"Use expressions like: {markers_str}")
        
        # Add example translations for problematic language pairs if detailed instructions preferred
        if instruction_style == "detailed" and lang_pair in self.special_handling:
            special_handling = self.special_handling[lang_pair]
            if "example_pairs" in special_handling and len(special_handling["example_pairs"]) > 0:
                # Add just one example for reference
                example = special_handling["example_pairs"][0]
                prefix.append(f"Example: '{example['source']}' translates to '{example['target']}'")
        
        # Add context instructions if provided
        if context and len(context) > 0:
            context_text = " ".join(context)
            if len(context_text) > 200 and instruction_style != "detailed":
                # Truncate context for non-detailed instruction styles
                context_text = context_text[:200] + "..."
            prefix.append(f"Use this context to inform translation: {context_text}")
        
        # Get model capabilities and adjust prefix for strengths/weaknesses
        capabilities = self.get_model_capabilities(model_name)
        
        # Add guidance based on model weaknesses for this particular language pair or domain
        weaknesses = capabilities.get("weaknesses", [])
        if "complex_sentences" in weaknesses and lang_pair_difficulty >= 3:
            prefix.append("Break complex sentences into simpler ones in the translation.")
        if "idiomatic_expressions" in weaknesses and lang_pair == "es-en":
            prefix.append("Replace Spanish idioms with equivalent English expressions, not literal translations.")
        if "legal_terminology" in weaknesses and domain == "legal":
            prefix.append("Pay special attention to legal terminology - be precise.")
        
        # Limit prompt length based on model capabilities
        max_tokens = capabilities.get("max_prompt_tokens", 100)
        
        # Build the final enhanced prompt
        prefix_str = " ".join(prefix)
        
        # Truncate prefix if it exceeds the maximum tokens (approximate)
        if len(prefix_str.split()) > max_tokens:
            # Keep approximately max_tokens words
            prefix_words = prefix_str.split()[:max_tokens]
            prefix_str = " ".join(prefix_words)
        
        # Replace placeholders in the template
        prompt_template = base_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
            text="{text}"  # Leave the text placeholder for now
        )
        
        # Build final prompt with prefix, prompt template, and text
        enhanced_prompt = f"{prefix_str} {prompt_template}".format(text=text)
        
        logger.debug(f"Enhanced MT5 prompt for {model_name}: {enhanced_prompt[:100]}...")
        return enhanced_prompt
    
    def get_mbart_generation_params(self,
                                   source_lang: str,
                                   target_lang: str,
                                   domain: Optional[str] = None,
                                   formality: Optional[str] = None,
                                   parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create enhanced generation parameters for MBART models based on model characteristics.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain for domain-specific translations
            formality: Optional formality level
            parameters: Additional parameters
            
        Returns:
            Dictionary of generation parameters
        """
        gen_kwargs = {}
        if parameters and "generation_kwargs" in parameters:
            gen_kwargs = parameters["generation_kwargs"].copy()
        
        # Get model name to use model-specific optimizations
        model_name = parameters.get("model_name", "mbart") if parameters else "mbart"
        
        # Create language pair key
        lang_pair = f"{source_lang}-{target_lang}"
        lang_pair_difficulty = self.language_pair_difficulty.get(lang_pair, self.language_pair_difficulty["default"])
        
        # Find the best model default parameters
        if model_name in self.model_default_params:
            model_params = self.model_default_params[model_name]
            # Check if we have language pair-specific parameters
            if lang_pair in model_params:
                # Apply language pair-specific defaults
                for key, value in model_params[lang_pair].items():
                    if key not in gen_kwargs:
                        gen_kwargs[key] = value
            else:
                # Apply model default parameters
                for key, value in model_params.get("default", {}).items():
                    if key not in gen_kwargs:
                        gen_kwargs[key] = value
        else:
            # Look for partial model name match in default params
            matched = False
            for model_key in self.model_default_params:
                if model_key in model_name:
                    model_params = self.model_default_params[model_key]
                    # Apply language pair-specific or default parameters
                    params_to_apply = model_params.get(lang_pair, model_params.get("default", {}))
                    for key, value in params_to_apply.items():
                        if key not in gen_kwargs:
                            gen_kwargs[key] = value
                    matched = True
                    break
            
            # If no match found, use generic defaults
            if not matched:
                generic_params = self.model_default_params["default"]["default"]
                for key, value in generic_params.items():
                    if key not in gen_kwargs:
                        gen_kwargs[key] = value
        
        # Set basic defaults if still not provided
        if "max_length" not in gen_kwargs:
            gen_kwargs["max_length"] = 512
        
        if "num_beams" not in gen_kwargs:
            # Adjust beam count based on language pair difficulty
            gen_kwargs["num_beams"] = 4 + lang_pair_difficulty
        
        # Apply domain-specific parameter adjustments
        if domain:
            if domain == "legal" or domain == "housing_legal":
                # Higher accuracy for legal content
                if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] < 6:
                    gen_kwargs["num_beams"] = 6
                if "length_penalty" not in gen_kwargs:
                    gen_kwargs["length_penalty"] = 1.2  # Encourage slightly longer translations for legal content
                if "repetition_penalty" not in gen_kwargs:
                    gen_kwargs["repetition_penalty"] = 1.2  # Avoid repetition
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False  # More deterministic for legal content
            
            elif domain == "medical":
                # Higher accuracy for medical content
                if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] < 5:
                    gen_kwargs["num_beams"] = 5
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False  # More deterministic for medical content
                if "length_penalty" not in gen_kwargs:
                    gen_kwargs["length_penalty"] = 1.1  # Encourage slightly longer translations for medical precision
            
            elif domain == "technical":
                # Precision for technical content
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] < 5:
                    gen_kwargs["num_beams"] = 5
                if "length_penalty" not in gen_kwargs:
                    gen_kwargs["length_penalty"] = 1.0
            
            elif domain == "casual":
                # More natural-sounding for casual content
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = True
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0.9
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95
        
        # Apply formality-specific parameter adjustments
        if formality:
            if formality == "formal":
                # More precise and conservative for formal content
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0.7
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.9
                if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] < 5:
                    gen_kwargs["num_beams"] = 5
            
            elif formality == "informal":
                # More varied and natural for informal content
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0.9
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = True
        
        # Apply model-specific adjustments based on language pair
        if "mbart" in model_name.lower() and lang_pair == "es-en":
            # MBART-specific optimizations for Spanish-English
            if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] < 8:
                gen_kwargs["num_beams"] = 8
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "early_stopping" not in gen_kwargs:
                gen_kwargs["early_stopping"] = True
        
        # Get model proficiency for this language pair
        proficiency = self.get_language_pair_proficiency(model_name, source_lang, target_lang)
        
        # For lower proficiency language pairs, increase sampling diversity and beam width
        if proficiency < 7.5:
            if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] < 6:
                gen_kwargs["num_beams"] = 6
            if "length_penalty" not in gen_kwargs:
                gen_kwargs["length_penalty"] = 1.1  # Encourage slightly longer translations
        
        return gen_kwargs
    
    def create_domain_prompt_prefix(self,
                                  source_lang: str,
                                  target_lang: str,
                                  domain: Optional[str] = None,
                                  formality: Optional[str] = None,
                                  model_name: str = "default") -> str:
        """
        Create a model-specific domain and formality prompt prefix.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            domain: Optional domain for domain-specific translations
            formality: Optional formality level
            model_name: Model name for model-specific adjustments
            
        Returns:
            Prompt prefix string
        """
        # Get model's preferred instruction style
        instruction_style = self.get_model_instruction_style(model_name)
        prefix_parts = []
        
        # Determine language pair
        lang_pair = f"{source_lang}-{target_lang}"
        
        # For MBART models that prefer minimal instructions, use shorter prefix
        if "mbart" in model_name.lower() and instruction_style == "minimal":
            # Minimal domain prefix
            if domain and domain in self.domain_templates and "mbart-minimal" in self.domain_templates[domain]:
                prefix_parts.append(self.domain_templates[domain]["mbart-minimal"].replace("{text}", ""))
            
            # Minimal formality prefix
            if formality and formality in self.formality_templates and "mbart-minimal" in self.formality_templates[formality]:
                prefix_parts.append(self.formality_templates[formality]["mbart-minimal"].replace("{text}", ""))
            
            # Minimal language pair quality hint
            if lang_pair in self.quality_hints and "mbart" in self.quality_hints[lang_pair]:
                # Just add one hint for minimal style
                prefix_parts.append(self.quality_hints[lang_pair]["mbart"][0])
            
            # Return minimal prefix
            return " ".join(prefix_parts)
        
        # Standard or detailed prefix construction
        # Add domain-specific prefix
        if domain and domain in self.domain_templates:
            prefix_parts.append(self.domain_templates[domain]["prefix"])
        
        # Add formality-specific prefix
        if formality and formality in self.formality_templates:
            prefix_parts.append(self.formality_templates[formality]["prefix"])
        
        # Add language pair specific quality hints
        if lang_pair in self.quality_hints:
            if "mbart" in self.quality_hints[lang_pair]:
                # Use model-specific hints if available
                prefix_parts.extend(self.quality_hints[lang_pair]["mbart"])
            else:
                # Use default hints for the language pair
                prefix_parts.extend(self.quality_hints[lang_pair][:2])  # Limit to two hints
        else:
            # Use default quality hints
            if "mbart" in self.quality_hints["default"]:
                prefix_parts.extend(self.quality_hints["default"]["mbart"])
            else:
                prefix_parts.extend(self.quality_hints["default"][:1])  # Just add the first generic hint
        
        # Add specific advice for challenging language pairs
        if lang_pair in self.special_handling:
            special = self.special_handling[lang_pair]
            if "check_terms" in special and len(special["check_terms"]) > 0:
                prefix_parts.append(f"Watch for proper translation of: {', '.join(special['check_terms'][:3])}")
        
        # Combine all prefix parts
        prefix = " ".join(prefix_parts)
        
        # Limit prefix length for models that prefer shorter instructions
        if instruction_style != "detailed":
            prefix_words = prefix.split()
            if len(prefix_words) > 50:  # Arbitrary limit for non-detailed instructions
                prefix = " ".join(prefix_words[:50])
        
        return prefix
    
    def enhance_translation_input(self,
                                input_data: Dict[str, Any],
                                model_type: str) -> Dict[str, Any]:
        """
        Enhance translation input data with model-specific prompts and parameters.
        
        Args:
            input_data: Input data dictionary
            model_type: Model type (mbart, mt5, etc.)
            
        Returns:
            Enhanced input data
        """
        enhanced_input = input_data.copy()
        
        # Extract required fields
        text = input_data.get("text", "")
        source_lang = input_data.get("source_language", "en")
        target_lang = input_data.get("target_language", "es")
        
        # Extract optional fields
        parameters = input_data.get("parameters", {})
        domain = parameters.get("domain")
        formality = parameters.get("formality")
        context = input_data.get("context")
        
        # Extract or determine full model name
        model_name = parameters.get("model_name", model_type)
        
        # Get model's preferred instruction style
        instruction_style = self.get_model_instruction_style(model_name)
        
        # Handle different model types
        if "mt5" in model_type.lower():
            # For MT5 models, enhance the prompt text with model-specific templates
            if isinstance(text, list):
                enhanced_texts = []
                for t in text:
                    enhanced_prompt = self.enhance_mt5_prompt(
                        t, source_lang, target_lang, domain, formality, context, 
                        {**parameters, "model_name": model_name}
                    )
                    enhanced_texts.append(enhanced_prompt)
                enhanced_input["text"] = enhanced_texts
            else:
                enhanced_prompt = self.enhance_mt5_prompt(
                    text, source_lang, target_lang, domain, formality, context,
                    {**parameters, "model_name": model_name}
                )
                enhanced_input["text"] = enhanced_prompt
        
        elif "mbart" in model_type.lower():
            # For MBART models, add model-specific generation parameters
            enhanced_params = enhanced_input.get("parameters", {}).copy()
            
            # Add enhanced generation parameters with model awareness
            gen_kwargs = self.get_mbart_generation_params(
                source_lang, target_lang, domain, formality,
                {**parameters, "model_name": model_name}
            )
            enhanced_params["generation_kwargs"] = gen_kwargs
            
            # Apply prompt prefix based on model's instruction style
            # Some MBART models can benefit from text prefixes
            use_prompt_prefix = parameters.get("add_prompt_prefix", False)
            
            # For certain language pairs, always use prompt prefix regardless of setting
            lang_pair = f"{source_lang}-{target_lang}"
            if lang_pair in self.special_handling:
                # Override for difficult language pairs
                use_prompt_prefix = True
            
            # Apply prompt prefix if needed
            if use_prompt_prefix:
                prefix = self.create_domain_prompt_prefix(
                    source_lang, target_lang, domain, formality, model_name
                )
                
                if isinstance(text, list):
                    enhanced_texts = []
                    for t in text:
                        # For minimal instruction style, use shorter combined format
                        if instruction_style == "minimal":
                            prefix_and_text = f"{prefix} {t}"
                        else:
                            prefix_and_text = f"{prefix}\n\n{t}"
                        enhanced_texts.append(prefix_and_text)
                    enhanced_input["text"] = enhanced_texts
                else:
                    # For minimal instruction style, use shorter combined format
                    if instruction_style == "minimal":
                        enhanced_input["text"] = f"{prefix} {text}"
                    else:
                        enhanced_input["text"] = f"{prefix}\n\n{text}"
            
            enhanced_input["parameters"] = enhanced_params
        
        # For all models, add enhanced metadata to help with post-processing
        if "parameters" not in enhanced_input:
            enhanced_input["parameters"] = {}
        
        # Add domain and formality hints
        if domain:
            enhanced_input["parameters"]["domain_hint"] = domain
        if formality:
            enhanced_input["parameters"]["formality_level"] = formality
        
        # Add language pair data to help post-processing
        enhanced_input["parameters"]["language_pair"] = f"{source_lang}-{target_lang}"
        enhanced_input["parameters"]["language_pair_difficulty"] = self.language_pair_difficulty.get(
            f"{source_lang}-{target_lang}", 
            self.language_pair_difficulty["default"]
        )
        
        # Add model proficiency data to help with fallback decisions
        proficiency = self.get_language_pair_proficiency(model_name, source_lang, target_lang)
        enhanced_input["parameters"]["model_proficiency"] = proficiency
        
        # For domain-specific translation, add key terms to watch for in post-processing
        if domain and domain in self.domain_templates and "key_terms" in self.domain_templates[domain]:
            enhanced_input["parameters"]["domain_key_terms"] = self.domain_templates[domain]["key_terms"]
        
        # Log the enhancement
        logger.debug(f"Enhanced translation input for {model_name} model with {instruction_style} instruction style")
        
        return enhanced_input