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
    
    async def detect_language(self, text: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Detect the language of the input text.
        
        Args:
            text: Text to detect language
            detailed: Whether to include detailed information
            
        Returns:
            Dict containing:
            - detected_language: ISO language code
            - confidence: Detection confidence (0.0-1.0)
            - possible_languages: List of other possible languages (if detailed)
        """
        if not self.initialized:
            await self.initialize()
        
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for language detection")
            return {
                "detected_language": "en",  # Default to English
                "language": "en",  # For compatibility
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
        
        # Try pattern-based detection first as a fallback result
        # This ensures we have something to return even if the model fails
        fallback_result = await self._fallback_detection(text)
        
        try:
            # Import prompt enhancer if available
            try:
                from app.services.models.language_detector_prompt_enhancer import LanguageDetectorPromptEnhancer
                prompt_enhancer = LanguageDetectorPromptEnhancer()
                enhanced_prompt = True
            except ImportError:
                enhanced_prompt = False
                logger.debug("Language detector prompt enhancer not available")
            
            # Prepare input data for the model
            input_data = {
                "text": text,
                "parameters": {"detailed": detailed}
            }
            
            # Enhance prompt if available
            if enhanced_prompt:
                try:
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
                        
                    logger.debug("Using enhanced prompt for language detection")
                except Exception as e:
                    logger.warning(f"Error enhancing language detection prompt: {str(e)}")
            
            # Try to use model manager
            try:
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
                        detected_lang = detection_result.get("language", fallback_result["detected_language"])
                        confidence = detection_result.get("confidence", fallback_result["confidence"])
                        possible_languages = []
                        
                    elif isinstance(detection_result, list) and detection_result:
                        # Multiple languages with probabilities
                        detected_lang = detection_result[0].get("language", fallback_result["detected_language"])
                        confidence = detection_result[0].get("confidence", fallback_result["confidence"])
                        
                        # Extract other possible languages
                        possible_languages = [
                            {"language": item.get("language"), "confidence": item.get("confidence", 0.0)}
                            for item in detection_result[1:4]  # Get top 3 alternatives
                        ]
                    else:
                        # Unexpected format, use fallback
                        logger.warning(f"Unexpected language detection result format: {detection_result}")
                        return fallback_result
                    
                    # Create result - include extra fields for compatibility
                    detection_result = {
                        "detected_language": detected_lang,
                        "language": detected_lang,  # For compatibility
                        "confidence": confidence,
                        "possible_languages": possible_languages,
                        "processing_time": time.time() - start_time
                    }
                    
                    # Add model metadata if available
                    if isinstance(result, dict):
                        if "performance_metrics" in result:
                            detection_result["performance_metrics"] = result["performance_metrics"]
                        if "memory_usage" in result:
                            detection_result["memory_usage"] = result["memory_usage"]
                        if "operation_cost" in result:
                            detection_result["operation_cost"] = result["operation_cost"]
                        if "accuracy_score" in result:
                            detection_result["accuracy_score"] = result["accuracy_score"]
                        if "truth_score" in result:
                            detection_result["truth_score"] = result["truth_score"]
                    
                    # Track metrics if verification data available
                    try:
                        # Track accuracy metrics to audit logger
                        from app.audit.logger import AuditLogger
                        audit_logger = AuditLogger()
                        await audit_logger.log_language_detection(
                            text_length=len(text),
                            detected_language=detected_lang, 
                            confidence=confidence,
                            model_id=model_id if enhanced_prompt else "unknown",
                            processing_time=detection_result["processing_time"],
                            metadata={
                                "accuracy_score": detection_result.get("accuracy_score", confidence),
                                "enhanced_prompt": enhanced_prompt
                            }
                        )
                    except (ImportError, Exception) as e:
                        logger.debug(f"Could not log language detection metrics: {str(e)}")
                    
                    # Update cache
                    self._update_cache(cache_key, detection_result)
                    
                    return detection_result
                
                # Model didn't return expected format, use our prepared fallback
                logger.warning("Language detection model returned unexpected result")
                return fallback_result
                
            except Exception as e:
                logger.warning(f"Error using model for language detection: {str(e)}")
                return fallback_result
                
        except Exception as e:
            logger.error(f"Critical error in language detection: {str(e)}")
            # Use the fallback result we generated earlier
            return fallback_result
        
    async def _fallback_detection(self, text: str) -> Dict[str, Any]:
        """
        Fallback language detection using pattern matching and libraries.
        
        Args:
            text: Text to detect language
            
        Returns:
            Dict with detection results
        """
        start_time = time.time()
        logger.debug("Using fallback language detection")
        
        detected_lang = "en"  # Default
        confidence = 0.0
        possible_languages = []
        
        # First try langdetect if available
        try:
            import langdetect
            from langdetect import detect_langs
            
            # Set seed for deterministic results
            langdetect.DetectorFactory.seed = 0
            
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
                
                # Return result from langdetect with enhanced info
                result = {
                    "detected_language": detected_lang,
                    "language": detected_lang,  # For compatibility
                    "confidence": confidence,
                    "possible_languages": possible_languages,
                    "detection_method": "langdetect",
                    "processing_time": time.time() - start_time
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
        
        # Create result with enhanced information
        processing_time = time.time() - start_time
        result = {
            "detected_language": detected_lang,
            "language": detected_lang,  # For compatibility
            "confidence": confidence,
            "possible_languages": possible_languages,
            "detection_method": "pattern_matching",
            "processing_time": processing_time,
            # Add mock enhanced metrics for compatibility with model output
            "performance_metrics": {
                "tokens_per_second": len(text.split()) / max(0.001, processing_time),
                "latency_ms": processing_time * 1000
            },
            "memory_usage": {
                "peak_mb": 50.0,
                "allocated_mb": 30.0
            },
            "operation_cost": 0.00001 * len(text), # Mock cost
            "accuracy_score": confidence,
            "truth_score": confidence * 0.9
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