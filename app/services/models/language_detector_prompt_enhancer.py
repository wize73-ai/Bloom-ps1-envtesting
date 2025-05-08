"""
Language Detector Prompt Enhancement Module for CasaLingua

This module provides sophisticated model-aware prompting to enhance language detection quality
through advanced prompt engineering, model specialization, and feature detection optimization.

It provides:
1. Model-specific prompt templates and strategies
2. Language feature detection optimization
3. Code-mixed text handling strategies
4. Quality assurance hints tailored to each model's strengths and weaknesses
5. Dynamic parameter optimization based on model characteristics
6. Special handling for difficult-to-detect languages
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Configure logging
logger = logging.getLogger(__name__)

class LanguageDetectorPromptEnhancer:
    """
    Enhances prompts for language detection models to improve quality
    through model-aware prompt engineering, feature recognition,
    and specialized handling of difficult cases.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the language detector prompt enhancer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Track known models and their capabilities
        self.model_capabilities = {
            "xlm-roberta-base": {
                "languages": ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja", "ko", 
                              "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh"],
                "strengths": ["european_languages", "high_resource_languages", "long_texts"],
                "weaknesses": ["code_mixed_text", "very_short_texts", "rare_languages"],
                "instruction_style": "minimal",
                "supports_detailed_results": True,
                "max_prompt_tokens": 50
            },
            "xlm-roberta-large": {
                "languages": ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja", "ko", 
                              "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh"],
                "strengths": ["european_languages", "asian_languages", "multilingual_texts"],
                "weaknesses": ["very_short_texts", "rare_languages"],
                "instruction_style": "minimal",
                "supports_detailed_results": True,
                "max_prompt_tokens": 50
            },
            "bert-base-multilingual-cased": {
                "languages": ["en", "es", "de", "fr", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar"],
                "strengths": ["high_resource_languages", "structured_text"],
                "weaknesses": ["low_resource_languages", "informal_text", "code_mixed_text"],
                "instruction_style": "explicit",
                "supports_detailed_results": True,
                "max_prompt_tokens": 100
            },
            "LID-model": {
                "strengths": ["many_languages", "dialect_detection", "short_texts"],
                "weaknesses": ["rare_scripts", "formal_documents"],
                "instruction_style": "detailed",
                "supports_detailed_results": True,
                "max_prompt_tokens": 150
            },
            "fasttext-lid": {
                "strengths": ["short_texts", "web_content", "social_media_text"],
                "weaknesses": ["formal_documents", "mixed_scripts"],
                "instruction_style": "none",  # FastText doesn't use prompts
                "supports_detailed_results": True,
                "max_prompt_tokens": 0
            },
            "default": {
                "strengths": ["general_language_detection"],
                "weaknesses": [],
                "instruction_style": "balanced",
                "supports_detailed_results": True,
                "max_prompt_tokens": 100
            }
        }
        
        # Language detection difficulty ratings (higher = more difficult)
        self.language_detection_difficulty = {
            "en": 1,  # English is usually easy to detect
            "es": 1,  # Spanish is usually easy to detect
            "fr": 1,  # French is usually easy to detect
            "de": 1,  # German is usually easy to detect
            "zh": 2,  # Chinese requires sufficient text
            "ja": 2,  # Japanese requires sufficient text
            "ko": 2,  # Korean requires sufficient text
            "ru": 2,  # Russian uses Cyrillic script
            "ar": 2,  # Arabic uses Arabic script
            "hi": 3,  # Hindi may be confused with similar languages
            "ur": 3,  # Urdu may be confused with similar languages
            "fa": 3,  # Persian may be confused with similar languages
            "default": 2
        }
        
        # Language feature detection strategies
        self.language_features = {
            "script": {
                "latin": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ro", "sv", "da", "no", "fi"],
                "cyrillic": ["ru", "uk", "bg", "sr", "mk"],
                "arabic": ["ar", "fa", "ur"],
                "cjk": ["zh", "ja", "ko"],
                "devanagari": ["hi", "mr", "ne", "sa"],
                "thai": ["th"],
                "greek": ["el"]
            },
            "common_words": {
                "en": ["the", "and", "is", "in", "to", "of", "that", "for", "it", "with"],
                "es": ["el", "la", "los", "las", "y", "de", "en", "que", "un", "una"],
                "fr": ["le", "la", "les", "des", "et", "de", "un", "une", "en", "que"],
                "de": ["der", "die", "das", "und", "ist", "in", "zu", "den", "mit", "für"],
                "pt": ["o", "a", "os", "as", "e", "de", "um", "uma", "que", "para"],
                "it": ["il", "la", "i", "le", "e", "di", "un", "una", "che", "per"],
                "nl": ["de", "het", "een", "in", "en", "van", "is", "op", "voor", "met"]
            },
            "distinctive_patterns": {
                "en": r"\b(ing|ly|ion)\b",
                "es": r"\b(ción|mente|dad)\b",
                "fr": r"\b(ment|tion|tait)\b",
                "de": r"\b(ung|lich|keit)\b",
                "pt": r"\b(ção|mente|dade)\b",
                "it": r"\b(zione|mente|ità)\b",
                "nl": r"\b(heid|lijk|ing)\b"
            }
        }
        
        # Prompt templates for different instruction styles and models
        self.prompt_templates = {
            "minimal": "detect language: {text}",
            "balanced": "identify the language of this text: {text}",
            "detailed": "analyze this text and determine the primary language used. Consider script, common words, and linguistic features: {text}",
            "explicit": "detect which language is used in the following text, providing a confidence score between 0 and 1. If multiple languages are present, identify the primary language: {text}",
            "code_mixed": "this text may contain multiple languages. identify the primary language and any other languages present: {text}"
        }
        
        # Special handling templates for difficult-to-detect languages
        self.special_handling = {
            "short_text": "detect language from this short text, focusing on distinctive words and character patterns: {text}",
            "code_mixed": "detect primary language from this code-mixed text, which may contain multiple languages: {text}",
            "similar_scripts": "distinguish between similar script languages in this text, paying attention to distinctive patterns: {text}"
        }
        
        # Quality hints for different models and scenarios
        self.quality_hints = {
            "xlm-roberta": [
                "focus on distinctive script features",
                "detect language from common word patterns",
                "consider vocabulary and grammar patterns"
            ],
            "bert-multilingual": [
                "identify language from word frequencies",
                "analyze token distributions",
                "detect based on statistical patterns"
            ],
            "short_text": [
                "focus on script and character patterns",
                "identify distinctive characters",
                "look for language-specific markers"
            ],
            "code_mixed": [
                "identify the dominant language",
                "detect script mixing patterns",
                "separate code-switched segments"
            ],
            "default": [
                "analyze text for language features",
                "identify primary language"
            ]
        }
        
        logger.info("Language detector prompt enhancer initialized")
    
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
            Instruction style (minimal, balanced, detailed, explicit, none)
        """
        capabilities = self.get_model_capabilities(model_name)
        return capabilities.get("instruction_style", "balanced")
    
    def detect_language_features(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to detect language-specific features.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of detected language features
        """
        import re
        
        features = {
            "length": len(text),
            "word_count": len(text.split()),
            "detected_scripts": set(),
            "possible_languages": [],
            "is_code_mixed": False
        }
        
        # Detect scripts present in the text
        script_patterns = {
            "latin": r'[a-zA-Z]',
            "cyrillic": r'[а-яА-Я]',
            "arabic": r'[\u0600-\u06FF]',
            "devanagari": r'[\u0900-\u097F]',
            "thai": r'[\u0E00-\u0E7F]',
            "greek": r'[\u0370-\u03FF]',
            "chinese": r'[\u4E00-\u9FFF]',
            "japanese_hiragana": r'[\u3040-\u309F]',
            "japanese_katakana": r'[\u30A0-\u30FF]',
            "japanese_kanji": r'[\u4E00-\u9FFF]',  # Overlaps with Chinese
            "korean": r'[\uAC00-\uD7A3]'
        }
        
        for script, pattern in script_patterns.items():
            if re.search(pattern, text):
                features["detected_scripts"].add(script)
                
                # Add possible languages based on script
                if script in self.language_features["script"]:
                    features["possible_languages"].extend(self.language_features["script"][script])
                elif script == "chinese" or script == "japanese_kanji":
                    features["possible_languages"].extend(["zh", "ja"])
                elif script == "japanese_hiragana" or script == "japanese_katakana":
                    features["possible_languages"].append("ja")
                elif script == "korean":
                    features["possible_languages"].append("ko")
        
        # Check for common words of different languages
        for lang, common_words in self.language_features["common_words"].items():
            word_matches = 0
            text_lower = text.lower()
            for word in common_words:
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, text_lower):
                    word_matches += 1
            
            if word_matches >= 2:  # If at least 2 common words are found
                if lang not in features["possible_languages"]:
                    features["possible_languages"].append(lang)
        
        # Check for distinctive language patterns
        for lang, pattern in self.language_features["distinctive_patterns"].items():
            if re.search(pattern, text, re.IGNORECASE):
                if lang not in features["possible_languages"]:
                    features["possible_languages"].append(lang)
        
        # Check if text is potentially code-mixed
        if len(features["detected_scripts"]) > 1:
            features["is_code_mixed"] = True
        
        # Make unique list of possible languages
        features["possible_languages"] = list(set(features["possible_languages"]))
        
        # Convert set to list for JSON serialization
        features["detected_scripts"] = list(features["detected_scripts"])
        
        return features
    
    def enhance_prompt(self, text: str, model_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance the language detection prompt for a specific model.
        
        Args:
            text: Text to detect language
            model_name: Name/identifier of the model
            parameters: Additional parameters for customization
            
        Returns:
            Dictionary with enhanced prompt and parameters
        """
        parameters = parameters or {}
        detailed = parameters.get("detailed", False)
        
        # Get model capabilities
        capabilities = self.get_model_capabilities(model_name)
        
        # Create appropriate prompt based on model capabilities
        prompt = self._create_detection_prompt(text, capabilities, detailed)
        
        # Optimize parameters for this model
        optimized_params = self._optimize_parameters(capabilities, parameters)
        
        return {
            "prompt": prompt,
            "parameters": optimized_params
        }
    
    def _create_detection_prompt(self, text: str, capabilities: Dict[str, Any], detailed: bool) -> str:
        """
        Create a model-appropriate detection prompt.
        
        Args:
            text: Text to detect language
            capabilities: Model capabilities
            detailed: Whether to request detailed results
            
        Returns:
            Optimized prompt string
        """
        # Analyze text features
        features = self.detect_language_features(text)
        
        # Determine appropriate instruction style
        instruction_style = capabilities.get("instruction_style", "balanced")
        
        # Skip prompt creation for models that don't use prompts
        if instruction_style == "none":
            return text
        
        # Choose appropriate template based on text features and model
        if features["is_code_mixed"]:
            # Use code-mixed template for mixed script texts
            template = self.prompt_templates["code_mixed"]
        elif features["length"] < 20 or features["word_count"] < 5:
            # Use short text template for very short texts
            template = self.special_handling["short_text"]
        elif len(features["detected_scripts"]) == 0:
            # Fallback for non-letter content
            template = self.prompt_templates["explicit"]
        else:
            # Use template based on model's preferred instruction style
            template = self.prompt_templates[instruction_style]
        
        # For models that support detailed results, add specific instructions if needed
        if detailed and capabilities.get("supports_detailed_results", False):
            if instruction_style == "detailed" or instruction_style == "explicit":
                template = template.replace("{text}", "Provide detailed language identification with confidence scores for: {text}")
            elif instruction_style == "balanced":
                template = "identify language with confidence scores: {text}"
        
        # Add quality hints for models that benefit from them
        if instruction_style == "detailed" or instruction_style == "explicit":
            hints = []
            
            # Add model-specific hints
            for model_key in self.quality_hints:
                if model_key in model_name:
                    hints.extend(self.quality_hints[model_key][:2])  # Add up to 2 hints
                    break
            
            # Add scenario-specific hints
            if features["is_code_mixed"]:
                hints.extend(self.quality_hints["code_mixed"][:1])
            elif features["length"] < 30:
                hints.extend(self.quality_hints["short_text"][:1])
            
            # If no specific hints were added, use default
            if not hints:
                hints.extend(self.quality_hints["default"])
            
            # Add hints to prompt
            hint_text = " ".join(hints)
            
            # Add hints before the main text
            template = f"{hint_text}. {template}"
        
        # Format template with text
        prompt = template.format(text=text)
        
        # Limit prompt length based on model capabilities
        max_tokens = capabilities.get("max_prompt_tokens", 100)
        
        # Simple tokenization for length limiting
        words = prompt.split()
        if len(words) > max_tokens:
            # Keep the instruction part and truncate the text if needed
            instruction_part = " ".join(words[:min(30, max_tokens // 2)])
            text_limit = max_tokens - len(instruction_part.split())
            text_part = " ".join(words[-text_limit:])
            prompt = f"{instruction_part} {text_part}"
        
        return prompt
    
    def _optimize_parameters(self, capabilities: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters for the specific model and text.
        
        Args:
            capabilities: Model capabilities
            parameters: User-provided parameters
            
        Returns:
            Optimized parameter dictionary
        """
        # Start with user parameters
        optimized = parameters.copy()
        
        # Add model-specific optimizations if not specified by user
        if "temperature" not in optimized:
            # Lower temperature for more deterministic language detection
            optimized["temperature"] = 0.3
        
        if "top_p" not in optimized:
            # Focus on most likely tokens
            optimized["top_p"] = 0.9
        
        # For models that support detailed results, ensure format is appropriate
        if "detailed" in optimized and optimized["detailed"]:
            if "output_format" not in optimized and capabilities.get("supports_detailed_results", False):
                optimized["output_format"] = "json"
        
        return optimized