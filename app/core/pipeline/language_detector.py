"""
Language Detector for CasaLingua

This module provides language detection capabilities for text inputs.
"""

import asyncio
import logging
import re
import time
from typing import Dict, Any, List, Optional, Set

from app.utils.logging import get_logger

logger = get_logger("casalingua.core.language_detector")

class LanguageDetector:
    """
    Detects the language of text input.
    
    This module uses transformer-based models to identify the language
    of input text, supporting a wide range of languages.
    """
    
    def __init__(
        self,
        model_manager,
        config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the language detector.
        
        Args:
            model_manager: Model manager for accessing detection models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.model_type = "language_detection"
        self.initialized = False
        
        # Language patterns as fallback
        self.language_patterns = {
            "en": re.compile(r'\b(the|and|is|to|a|in|that|of|for|it|with|as|was|on)\b', re.IGNORECASE),
            "es": re.compile(r'\b(el|la|los|las|y|es|de|en|que|por|con|para|un|una)\b', re.IGNORECASE),
            "fr": re.compile(r'\b(le|la|les|des|et|est|de|en|que|pour|avec|dans|un|une)\b', re.IGNORECASE),
            "de": re.compile(r'\b(der|die|das|und|ist|zu|von|in|den|für|mit|dem|ein|eine)\b', re.IGNORECASE),
            "it": re.compile(r'\b(il|lo|la|i|gli|le|e|è|di|a|in|che|per|con|su|un|una)\b', re.IGNORECASE),
            "pt": re.compile(r'\b(o|a|os|as|e|é|de|em|que|para|com|um|uma|no|na)\b', re.IGNORECASE),
            "nl": re.compile(r'\b(de|het|een|in|en|van|is|voor|op|dat|die|met|zijn|heeft)\b', re.IGNORECASE),
            "sv": re.compile(r'\b(och|att|det|som|på|är|av|för|med|den|till|jag|har)\b', re.IGNORECASE),
            "ru": re.compile(r'\b(и|в|не|на|я|что|быть|с|он|а|но|как|это)\b', re.IGNORECASE),
            "zh": re.compile(r'[\u4e00-\u9fff]'),  # Chinese characters
            "ja": re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'),  # Japanese hiragana and katakana
            "ko": re.compile(r'[\uac00-\ud7af]')  # Korean Hangul
        }
        
        # Cache for language detection
        self.cache = {}
        self.cache_limit = 1000
        
        logger.info("Language detector initialized (not yet loaded)")
    
    async def initialize(self) -> None:
        """
        Initialize the language detector.
        
        This ensures models are loaded and ready.
        """
        if self.initialized:
            logger.debug("Language detector already initialized")
            return
        
        logger.info("Initializing language detector")
        
        try:
            # Try to load the model to check if it's available
            model_info = await self.model_manager.load_model(self.model_type)
            
            if model_info:
                logger.info(f"Language detection model loaded: {model_info.get('model_id', 'unknown')}")
            else:
                logger.warning("Language detection model not available, using fallback methods")
                
            # Try to import fasttext if available
            try:
                # Import for detection rather than initial loading
                import langdetect
                logger.info("LangDetect library available for fallback")
            except ImportError:
                logger.warning("LangDetect library not available for fallback")
            
            # Try to import nltk if available
            try:
                import nltk
                # Check if required data is available
                try:
                    nltk.data.find('tokenizers/punkt')
                    logger.info("NLTK available for fallback")
                except LookupError:
                    logger.warning("NLTK data not available, try running nltk.download('punkt')")
            except ImportError:
                logger.warning("NLTK not available for fallback")
        
        except Exception as e:
            logger.warning(f"Error initializing language detector: {str(e)}")
            logger.warning("Using pattern-based fallback for language detection")
        
        self.initialized = True
        logger.info("Language detector initialization complete")
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the input text.
        
        Args:
            text: Text to detect language
            
        Returns:
            Dict containing:
            - detected_language: ISO language code
            - confidence: Detection confidence (0.0-1.0)
            - possible_languages: List of other possible languages
        """
        if not self.initialized:
            await self.initialize()
        
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for language detection")
            return {
                "detected_language": "en",  # Default to English
                "confidence": 0.0,
                "possible_languages": []
            }
        
        # Check cache
        cache_key = text[:100].lower()  # Use first 100 chars for caching
        if cache_key in self.cache:
            logger.debug("Using cached language detection result")
            return self.cache[cache_key]
        
        # Start timing
        start_time = time.time()
        
        try:
            # Prepare input data
            input_data = {
                "text": text,
                "parameters": {"detailed": True}
            }
            
            # Try to use model manager
            result = await self.model_manager.run_model(
                self.model_type,
                "process",
                input_data
            )
            
            # Process result
            if isinstance(result, dict) and "result" in result:
                detection_result = result["result"]
                
                # Format standardization
                if isinstance(detection_result, dict):
                    # Single language result
                    detected_lang = detection_result.get("language", "en")
                    confidence = detection_result.get("confidence", 0.0)
                    possible_languages = []
                    
                elif isinstance(detection_result, list) and detection_result:
                    # Multiple languages with probabilities
                    detected_lang = detection_result[0].get("language", "en")
                    confidence = detection_result[0].get("confidence", 0.0)
                    
                    # Extract other possible languages
                    possible_languages = [
                        {"language": item.get("language"), "confidence": item.get("confidence", 0.0)}
                        for item in detection_result[1:4]  # Get top 3 alternatives
                    ]
                else:
                    # Unexpected format, use fallback
                    logger.warning(f"Unexpected language detection result format: {detection_result}")
                    return await self._fallback_detection(text)
                
                # Create result
                detection_result = {
                    "detected_language": detected_lang,
                    "confidence": confidence,
                    "possible_languages": possible_languages
                }
                
                # Update cache
                self._update_cache(cache_key, detection_result)
                
                return detection_result
            
            # Model didn't return expected format
            logger.warning("Language detection model returned unexpected result")
            return await self._fallback_detection(text)
            
        except Exception as e:
            logger.warning(f"Error using model for language detection: {str(e)}")
            
            # Use fallback method
            return await self._fallback_detection(text)
        
    async def _fallback_detection(self, text: str) -> Dict[str, Any]:
        """
        Fallback language detection using pattern matching and libraries.
        
        Args:
            text: Text to detect language
            
        Returns:
            Dict with detection results
        """
        logger.debug("Using fallback language detection")
        
        detected_lang = "en"  # Default
        confidence = 0.0
        possible_languages = []
        
        # First try langdetect if available
        try:
            import langdetect
            from langdetect import detect_langs
            
            # Get language probabilities
            langs = detect_langs(text)
            
            if langs:
                # Sort by probability
                langs.sort(key=lambda x: x.prob, reverse=True)
                
                # Get top language
                detected_lang = langs[0].lang
                confidence = langs[0].prob
                
                # Get alternatives
                possible_languages = [
                    {"language": lang.lang, "confidence": lang.prob}
                    for lang in langs[1:4]
                ]
                
                logger.debug(f"LangDetect detected language: {detected_lang} with confidence {confidence}")
                
                # Return result from langdetect
                result = {
                    "detected_language": detected_lang,
                    "confidence": confidence,
                    "possible_languages": possible_languages
                }
                
                return result
                
        except ImportError:
            logger.debug("LangDetect not available, using pattern matching")
        except Exception as e:
            logger.warning(f"Error using LangDetect: {str(e)}")
        
        # Fall back to pattern-based detection
        best_match = None
        highest_count = 0
        match_counts = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = len(pattern.findall(text))
            match_counts[lang] = matches
            
            if matches > highest_count:
                highest_count = matches
                best_match = lang
        
        # Require a minimum number of matches
        if highest_count < 2:
            detected_lang = "en"  # Default to English
            confidence = 0.1
        else:
            detected_lang = best_match
            confidence = min(0.8, highest_count / 20)  # Scale confidence based on matches
        
        # Get possible alternatives
        sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)
        possible_languages = [
            {"language": lang, "confidence": min(0.7, count / 20)}
            for lang, count in sorted_matches[1:4]
            if count > 0
        ]
        
        logger.debug(f"Pattern matching detected language: {detected_lang}")
        
        # Create result
        result = {
            "detected_language": detected_lang,
            "confidence": confidence,
            "possible_languages": possible_languages
        }
        
        # Update cache
        self._update_cache(text[:100].lower(), result)
        
        return result
    
    def _update_cache(self, key: str, result: Dict[str, Any]) -> None:
        """
        Update the detection cache.
        
        Args:
            key: Cache key
            result: Detection result
        """
        # Implement cache size limit
        if len(self.cache) >= self.cache_limit:
            # Remove a random item
            try:
                self.cache.pop(next(iter(self.cache)))
            except:
                pass
        
        # Add to cache
        self.cache[key] = result