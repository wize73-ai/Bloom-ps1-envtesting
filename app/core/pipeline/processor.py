"""
Unified Processor for CasaLingua

This module provides a unified interface for all language processing operations
within the CasaLingua platform.
"""

import os
import re
import json
import time
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from uuid import uuid4
from datetime import datetime

# Instance variable for global access
_processor_instance = None

# Import pipeline components
from app.core.pipeline.translator import TranslationPipeline
from app.core.pipeline.language_detector import LanguageDetector
from app.core.pipeline.anonymizer import AnonymizationPipeline
from app.core.pipeline.simplifier import SimplificationPipeline
from app.core.pipeline.summarizer import SummarizationPipeline
from app.core.pipeline.tts import TTSPipeline
from app.core.pipeline.stt import STTPipeline
from app.core.document.pdf import PDFProcessor
from app.core.document.docx import DOCXProcessor
from app.core.document.ocr import OCRProcessor
from app.core.rag.expert import RAGExpert
# Comment out the problematic import for now
# from app.core.rag.content_fetcher import ContentFetcher, ContentProcessor
from app.audit.logger import AuditLogger
from app.audit.metrics import MetricsCollector
from app.audit.veracity import VeracityAuditor

from app.utils.logging import get_logger
from app.utils.helpers import generate_session_token, sanitize_filename

# Define the logger for the processor
logger = get_logger("casalingua.processor")

class UnifiedProcessor:
    """
    Unified processor for all language processing operations.
    
    This class orchestrates various pipeline components to provide a
    seamless interface for performing language operations including:
    - Language detection
    - Translation
    - Simplification
    - Anonymization
    - Summarization
    - Text-to-Speech
    - Speech-to-Text
    - Document processing
    
    It also handles metrics, logging, and performance optimization.
    """
    
    def __init__(
            self, 
            model_manager, 
            config: Dict[str, Any] = None, 
            registry_config: Optional[Dict[str, Any]] = None
        ):
        """
        Initialize the unified processor.
        
        Args:
            model_manager: Model manager instance for accessing models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.config = config or {}
        self.model_manager = model_manager
        self.registry_config = registry_config or {}
        self.initialized = False
        
        # Initialize pipeline components to None - will be created on demand
        self.translation_pipeline = None
        self.language_detector = None
        self.anonymizer = None
        self.simplifier = None
        self.summarizer = None
        self.tts_pipeline = None
        self.stt_pipeline = None
        
        # Document processing
        self.pdf_processor = None
        self.docx_processor = None
        self.ocr_processor = None
        
        # RAG components
        self.rag_expert = None
        
        # Auditing and metrics - these will be set externally
        self.audit_logger = None  # Will be set from main.py
        self.metrics = None  # Will be set from main.py
        self.metrics_collector = None  # For backward compatibility
        self.veracity_auditor = VeracityAuditor(self.config)
        
        # Session storage
        self.session_dir = Path(self.config.get("session_dir", "sessions"))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache directories
        self.cache_dir = Path(self.config.get("cache_dir", "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio output directory
        self.audio_dir = Path(self.config.get("audio_dir", "audio"))
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("UnifiedProcessor initialized (components not yet loaded)")
    
    async def initialize(self) -> None:
        """
        Initialize pipeline components.
        
        This performs lazy initialization of pipeline components.
        """
        if self.initialized:
            logger.info("UnifiedProcessor already initialized")
            return
        
        logger.info("Initializing UnifiedProcessor components")
        
        # Perform concurrent initialization of core components
        init_tasks = [
            self._initialize_translation_pipeline(),
            self._initialize_language_detector(),
            self._initialize_stt_pipeline()
            # Other initializations commented out for now
            # self._initialize_simplifier(),
            # self._initialize_anonymizer(),
            # self._initialize_summarizer(),
            # self._initialize_tts_pipeline(),
        ]
        
        # Wait for core components to initialize
        await asyncio.gather(*init_tasks)
        
        self.initialized = True
        logger.info("UnifiedProcessor initialization complete")
    
    async def _initialize_translation_pipeline(self) -> None:
        """Initialize the translation pipeline component."""
        logger.info("Initializing translation pipeline")
        self.translation_pipeline = TranslationPipeline(
            self.model_manager,
            self.config,
            registry_config=self.registry_config
        )
        await self.translation_pipeline.initialize()
        logger.info("Translation pipeline initialization complete")
    
    async def _initialize_language_detector(self) -> None:
        """Initialize the language detector component."""
        logger.info("Initializing language detector")
        self.language_detector = LanguageDetector(
            self.model_manager,
            self.config
        )
        await self.language_detector.initialize()
        logger.info("Language detector initialization complete")
    
    async def _initialize_anonymizer(self) -> None:
        """Initialize the anonymizer component."""
        logger.info("Initializing anonymizer")
        self.anonymizer = AnonymizationPipeline(
            self.model_manager,
            self.config
        )
        await self.anonymizer.initialize()
        logger.info("Anonymizer initialization complete")
    
    async def _initialize_simplifier(self) -> None:
        """Initialize the simplifier component."""
        logger.info("Initializing simplifier")
        self.simplifier = SimplificationPipeline(
            self.model_manager,
            self.config
        )
        await self.simplifier.initialize()
        logger.info("Simplifier initialization complete")
    
    async def _initialize_summarizer(self) -> None:
        """Initialize the summarizer component."""
        logger.info("Initializing summarizer")
        self.summarizer = SummarizationPipeline(
            self.model_manager,
            self.config
        )
        await self.summarizer.initialize()
        logger.info("Summarizer initialization complete")
    
    async def _initialize_tts_pipeline(self) -> None:
        """Initialize the TTS pipeline component."""
        logger.info("Initializing TTS pipeline")
        self.tts_pipeline = TTSPipeline(
            self.model_manager,
            self.config,
            registry_config=self.registry_config
        )
        await self.tts_pipeline.initialize()
        logger.info("TTS pipeline initialization complete")
    
    async def _initialize_stt_pipeline(self) -> None:
        """Initialize the STT pipeline component."""
        logger.info("Initializing STT pipeline")
        self.stt_pipeline = STTPipeline(
            self.model_manager,
            self.config,
            registry_config=self.registry_config
        )
        await self.stt_pipeline.initialize()
        logger.info("STT pipeline initialization complete")
    
    async def _initialize_document_processors(self) -> None:
        """Initialize document processing components."""
        logger.info("Initializing document processors")
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(self.config)
        
        # Initialize DOCX processor
        self.docx_processor = DOCXProcessor(self.config)
        
        # Initialize OCR processor
        self.ocr_processor = OCRProcessor(
            self.model_manager,
            self.config
        )
        await self.ocr_processor.initialize()
        
        logger.info("Document processors initialization complete")
    
    async def _initialize_rag_components(self) -> None:
        """Initialize RAG components."""
        logger.info("Initializing RAG components")
        
        # Initialize RAG expert
        self.rag_expert = RAGExpert(
            self.model_manager,
            self.config
        )
        await self.rag_expert.initialize()
        
        # The ContentFetcher is commented out for now since it was causing issues
        # We'll add it back later
        
        logger.info("RAG components initialization complete")
    
    async def detect_language(
        self,
        text: str,
        confidence_threshold: float = 0.6,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect the language of a text.
        
        Args:
            text: The text to detect language
            confidence_threshold: Minimum confidence threshold
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with detected language and confidence scores
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.language_detector:
            logger.warning("Language detector not initialized, initializing now")
            await self._initialize_language_detector()
        
        # Start timing
        start_time = time.time()
        
        # Detect language
        try:
            detection_result = await self.language_detector.detect_language(text)
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            
            # Return failure result
            processing_time = time.time() - start_time
            return {
                "detected_language": "unknown",
                "confidence": 0.0,
                "possible_languages": [],
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing time to the result
        detection_result["processing_time"] = processing_time
        
        # Log the detection if user_id is provided
        if user_id:
            await self.audit_logger.log_language_detection(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                text_length=len(text),
                detected_language=detection_result["detected_language"],
                confidence=detection_result["confidence"],
                processing_time=processing_time
            )
        
        return detection_result
    
    async def translate_text(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: str = "en",
        quality_level: str = "standard",
        model_id: Optional[str] = None,
        options: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text from one language to another.
        
        Args:
            text: The text to translate
            source_language: Source language code (auto-detect if None)
            target_language: Target language code
            quality_level: Translation quality level (standard, high, etc.)
            model_id: Specific model to use
            options: Additional translation options
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with translated text and metadata
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.translation_pipeline:
            logger.warning("Translation pipeline not initialized, initializing now")
            await self._initialize_translation_pipeline()
        
        options = options or {}
        
        # Start timing
        start_time = time.time()
        
        # Auto-detect source language if not provided
        source_lang = source_language
        if not source_lang:
            try:
                if not self.language_detector:
                    await self._initialize_language_detector()
                
                detection_result = await self.language_detector.detect_language(text)
                source_lang = detection_result["detected_language"]
                
                # Add detection info to options
                options["detection_confidence"] = detection_result["confidence"]
                options["possible_languages"] = detection_result.get("possible_languages", [])
                
                logger.debug(f"Auto-detected source language: {source_lang}")
            except Exception as e:
                logger.warning(f"Error auto-detecting language: {e}")
                source_lang = "en"  # Default to English if detection fails
        
        # Validate that the source and target languages are different
        if source_lang == target_language:
            logger.debug(f"Source language {source_lang} is the same as target language, returning original text")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_language,
                "processing_time": processing_time,
                "model_used": "none",
                "quality_level": quality_level,
                "same_language": True
            }
        
        # Translate the text
        try:
            # Create a TranslationRequest object as required by translation_pipeline.translate()
            from app.api.schemas.translation import TranslationRequest
            
            # Create the translation request
            translation_request = TranslationRequest(
                text=text,
                source_language=source_lang,
                target_language=target_language,
                model_name=model_id,
                preserve_formatting=options.get("preserve_formatting", True),
                formality=options.get("formality", None),
                glossary_id=options.get("glossary_id", None),
                domain=options.get("domain", None),
                context=options.get("context", [])
            )
            
            # Call the translate method with the request object
            try:
                translation_result = await self.translation_pipeline.translate(translation_request)
            except TypeError as e:
                if "source_lang" in str(e):
                    # Fix for the source_lang vs source_language parameter mismatch
                    logger.warning("Detected parameter mismatch in translate method, using direct parameter passing")
                    translation_result = await self.translation_pipeline.translate_text(
                        text=text,
                        source_language=source_lang,
                        target_language=target_language,
                        model_id=model_id,
                        glossary_id=options.get("glossary_id", None),
                        preserve_formatting=options.get("preserve_formatting", True),
                        formality=options.get("formality", None)
                    )
                else:
                    # Re-raise if it's a different error
                    raise
            
            # Convert result to dictionary for consistency
            if hasattr(translation_result, 'dict'):
                translation_result = translation_result.dict()
            elif not isinstance(translation_result, dict):
                # Handle case where result is not a dict and not a Pydantic model
                translation_result = {
                    "translated_text": str(translation_result),
                    "source_language": source_lang,
                    "target_language": target_language,
                    "confidence": 0.0
                }
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return failure result
            return {
                "translated_text": text,  # Return original text
                "source_language": source_lang,
                "target_language": target_language,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate additional metrics
        word_count = len(text.split())
        output_text = translation_result.get("translated_text", "")
        output_word_count = len(output_text.split())
        
        # Calculate approximate cost (based on input and output lengths)
        # Assuming a cost model of $0.001 per input word and $0.002 per output word
        operation_cost = (word_count * 0.001) + (output_word_count * 0.002)
        
        # Approximate accuracy/quality score based on model and language pair
        # This is a mock implementation - real systems would use more sophisticated metrics
        base_quality = 0.85  # Base quality score
        # Adjust based on language pair difficulty (simplified example)
        language_difficulty = {
            "en-es": 0.05,  # English-Spanish is common, higher quality
            "en-fr": 0.05,  # English-French is common, higher quality
            "en-de": 0.03,  # English-German is common, slightly lower
            "es-en": 0.05,  # Spanish-English is common, higher quality
            "fr-en": 0.05,  # French-English is common, higher quality
            "de-en": 0.03,  # German-English is common, slightly lower
        }
        lang_pair = f"{source_lang}-{target_language}"
        quality_adjustment = language_difficulty.get(lang_pair, 0.0)
        quality_score = min(0.98, base_quality + quality_adjustment)
        
        # Calculate a mock "truth score" that evaluates the translation's fidelity
        # In a real system, this would be based on human evaluation or reference translations
        truth_score = quality_score * 0.95  # Slightly lower than quality score
        
        # Add metrics to the result
        translation_result["processing_time"] = processing_time
        translation_result["word_count"] = word_count
        translation_result["output_word_count"] = output_word_count
        translation_result["operation_cost"] = operation_cost
        translation_result["accuracy_score"] = quality_score
        translation_result["truth_score"] = truth_score
        
        # Add performance metrics
        translation_result["performance_metrics"] = {
            "tokens_per_second": word_count / processing_time if processing_time > 0 else 0,
            "latency_ms": processing_time * 1000,
            "throughput": len(text) / processing_time if processing_time > 0 else 0
        }
        
        # Add memory usage metrics (mock data for now)
        translation_result["memory_usage"] = {
            "peak_mb": 150.0,
            "allocated_mb": 120.0,
            "util_percent": 75.0
        }
        
        # Audit and metrics
        if user_id:
            await self.audit_logger.log_translation(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                source_language=source_lang,
                target_language=target_language,
                text_length=len(text),
                processing_time=processing_time,
                model_id=translation_result.get("model_used", "unknown"),
                quality_score=quality_score,  # Use actual quality score
                metadata={
                    "operation_cost": operation_cost,
                    "accuracy_score": quality_score,
                    "truth_score": truth_score,
                    "word_count": word_count,
                    "output_word_count": output_word_count
                }
            )
            
            # Collect metrics
            # Use metrics if set, fallback to metrics_collector for backward compatibility
            (self.metrics or self.metrics_collector).record_translation_metrics(
                source_language=source_lang,
                target_language=target_language,
                text_length=len(text),
                processing_time=processing_time,
                model_id=translation_result.get("model_used", "unknown")
            )
        
        return translation_result
    
    async def simplify_text(
        self,
        text: str,
        level: str = "medium",
        target_language: Optional[str] = None,
        source_language: Optional[str] = None,
        model_id: Optional[str] = None,
        options: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simplify text to make it more accessible.
        
        Args:
            text: The text to simplify
            level: Simplification level (easy, medium, standard)
            target_language: Target language for simplified text
            source_language: Source language of text (auto-detect if None)
            model_id: Specific model to use
            options: Additional simplification options
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with simplified text and metadata
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.simplifier:
            logger.warning("Simplifier not initialized, initializing now")
            await self._initialize_simplifier()
        
        options = options or {}
        
        # Start timing
        start_time = time.time()
        
        # Auto-detect source language if not provided
        source_lang = source_language
        if not source_lang:
            try:
                if not self.language_detector:
                    await self._initialize_language_detector()
                
                detection_result = await self.language_detector.detect_language(text)
                source_lang = detection_result["detected_language"]
                logger.debug(f"Auto-detected source language: {source_lang}")
            except Exception as e:
                logger.warning(f"Error auto-detecting language: {e}")
                source_lang = "en"  # Default to English if detection fails
        
        # Use source language as target if not specified
        target_lang = target_language or source_lang
        
        # Check if we need to translate before simplifying
        if source_lang != target_lang:
            logger.debug(f"Translating from {source_lang} to {target_lang} before simplifying")
            
            try:
                if not self.translation_pipeline:
                    await self._initialize_translation_pipeline()
                
                translation_result = await self.translation_pipeline.translate(
                    text=text,
                    source_language=source_lang,
                    target_language=target_lang
                )
                
                # Use translated text for simplification
                text_to_simplify = translation_result["translated_text"]
                model_used_for_translation = translation_result.get("model_used", "unknown")
            except Exception as e:
                logger.warning(f"Error translating before simplification: {e}")
                text_to_simplify = text
                model_used_for_translation = "translation_failed"
        else:
            text_to_simplify = text
            model_used_for_translation = None
        
        # Simplify the text
        try:
            try:
                simplification_result = await self.simplifier.simplify_text(
                    text=text_to_simplify,
                    level=level,
                    language=target_lang,
                    model_id=model_id,
                    options=options
                )
            except TypeError as e:
                if "target_level" in str(e):
                    # Fix for the target_level vs level parameter mismatch
                    logger.warning("Detected parameter mismatch in simplify_text method, using target_level instead of level")
                    simplification_result = await self.simplifier.simplify_text(
                        text=text_to_simplify,
                        target_level=level,  # Use target_level instead of level
                        language=target_lang,
                        model_id=model_id,
                        options=options
                    )
                else:
                    # Re-raise if it's a different error
                    raise
        except Exception as e:
            logger.error(f"Error simplifying text: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return failure result with original text
            return {
                "simplified_text": text_to_simplify,  # Return text before simplification
                "source_language": source_lang,
                "target_language": target_lang,
                "simplification_level": level,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "simplified_text": simplification_result["simplified_text"],
            "source_language": source_lang,
            "target_language": target_lang,
            "original_length": len(text),
            "simplified_length": len(simplification_result["simplified_text"]),
            "simplification_level": level,
            "processing_time": processing_time,
            "model_used": simplification_result.get("model_used", "unknown"),
            "success": True
        }
        
        # Add translation info if applicable
        if model_used_for_translation:
            result["translation_performed"] = True
            result["translation_model"] = model_used_for_translation
        
        # Audit and metrics
        if user_id:
            self.audit_logger.log_simplification(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                language=target_lang,
                text_length=len(text),
                simplified_length=len(simplification_result["simplified_text"]),
                level=level,
                processing_time=processing_time,
                model_id=simplification_result.get("model_used", "unknown")
            )
            
            # Collect metrics
            # Use metrics if set, fallback to metrics_collector for backward compatibility
            (self.metrics or self.metrics_collector).record_simplification_metrics(
                language=target_lang,
                text_length=len(text),
                simplified_length=len(simplification_result["simplified_text"]),
                level=level,
                processing_time=processing_time,
                model_id=simplification_result.get("model_used", "unknown")
            )
        
        return result
    
    async def anonymize_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        mode: str = "redact",
        language: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Anonymize text by removing or masking personal information.
        
        Args:
            text: The text to anonymize
            entity_types: Types of entities to anonymize
            mode: Anonymization mode (redact, mask, replace)
            language: Text language (auto-detect if None)
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with anonymized text and metadata
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.anonymizer:
            logger.warning("Anonymizer not initialized, initializing now")
            await self._initialize_anonymizer()
        
        # Start timing
        start_time = time.time()
        
        # Default entity types if not provided
        entity_types = entity_types or ["PERSON", "EMAIL", "PHONE", "ADDRESS", "CREDIT_CARD", "SSN"]
        
        # Auto-detect language if not provided
        detect_lang = language
        if not detect_lang:
            try:
                if not self.language_detector:
                    await self._initialize_language_detector()
                
                detection_result = await self.language_detector.detect_language(text)
                detect_lang = detection_result["detected_language"]
                logger.debug(f"Auto-detected language for anonymization: {detect_lang}")
            except Exception as e:
                logger.warning(f"Error auto-detecting language: {e}")
                detect_lang = "en"  # Default to English if detection fails
        
        # Anonymize the text
        try:
            anonymization_result = await self.anonymizer.anonymize_text(
                text=text,
                entity_types=entity_types,
                mode=mode,
                language=detect_lang
            )
        except Exception as e:
            logger.error(f"Error anonymizing text: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return failure result
            return {
                "anonymized_text": text,  # Return original text
                "language": detect_lang,
                "entity_types": entity_types,
                "mode": mode,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing time to the result
        anonymization_result["processing_time"] = processing_time
        
        # Audit and metrics
        if user_id:
            self.audit_logger.log_anonymization(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                language=detect_lang,
                text_length=len(text),
                entity_count=len(anonymization_result.get("entities", [])),
                entity_types=entity_types,
                mode=mode,
                processing_time=processing_time
            )
        
        return anonymization_result
    
    async def summarize_text(
        self,
        text: str,
        length: str = "medium",
        language: Optional[str] = None,
        model_id: Optional[str] = None,
        options: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize text to a shorter version.
        
        Args:
            text: The text to summarize
            length: Summary length (short, medium, long)
            language: Text language (auto-detect if None)
            model_id: Specific model to use
            options: Additional summarization options
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with summarized text and metadata
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.summarizer:
            logger.warning("Summarizer not initialized, initializing now")
            await self._initialize_summarizer()
        
        options = options or {}
        
        # Start timing
        start_time = time.time()
        
        # Auto-detect language if not provided
        detect_lang = language
        if not detect_lang:
            try:
                if not self.language_detector:
                    await self._initialize_language_detector()
                
                detection_result = await self.language_detector.detect_language(text)
                detect_lang = detection_result["detected_language"]
                logger.debug(f"Auto-detected language for summarization: {detect_lang}")
            except Exception as e:
                logger.warning(f"Error auto-detecting language: {e}")
                detect_lang = "en"  # Default to English if detection fails
        
        # Summarize the text
        try:
            summarization_result = await self.summarizer.summarize(
                text=text,
                length=length,
                language=detect_lang,
                model_id=model_id,
                options=options
            )
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return failure result
            return {
                "summary": "",
                "language": detect_lang,
                "original_length": len(text),
                "summary_length": 0,
                "summary_ratio": 0,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Process the summary result (it might be a string or a dict)
        if isinstance(summarization_result, str):
            summary_text = summarization_result
            model_used = "default"
        elif isinstance(summarization_result, dict):
            summary_text = summarization_result.get("summary", "")
            model_used = summarization_result.get("model_used", "default")
        else:
            # Convert to string as a fallback
            summary_text = str(summarization_result)
            model_used = "default"
        
        # Prepare result
        result = {
            "summary": summary_text,
            "language": detect_lang,
            "original_length": len(text),
            "summary_length": len(summary_text),
            "summary_ratio": round(len(summary_text) / len(text), 3) if len(text) > 0 else 0,
            "summary_length_setting": length,
            "processing_time": processing_time,
            "model_used": model_used,
            "success": True
        }
        
        # Audit and metrics
        if user_id:
            self.audit_logger.log_summarization(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                language=detect_lang,
                text_length=len(text),
                summary_length=len(summary_text),
                summary_ratio=result["summary_ratio"],
                length_setting=length,
                processing_time=processing_time,
                model_id=model_used
            )
        
        return result
    
    async def text_to_speech(
        self,
        text: str,
        language: Optional[str] = None,
        voice_id: Optional[str] = None,
        output_format: str = "mp3",
        speed: float = 1.0,
        pitch: float = 0.0,
        options: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert text to speech.
        
        Args:
            text: The text to convert to speech
            language: Text language (auto-detect if None)
            voice_id: Voice ID to use
            output_format: Audio format (mp3, wav, etc.)
            speed: Speech speed factor
            pitch: Speech pitch adjustment
            options: Additional TTS options
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with audio data, filename, and metadata
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.tts_pipeline:
            logger.warning("TTS pipeline not initialized, initializing now")
            await self._initialize_tts_pipeline()
        
        options = options or {}
        
        # Start timing
        start_time = time.time()
        
        # Auto-detect language if not provided
        detect_lang = language
        if not detect_lang:
            try:
                if not self.language_detector:
                    await self._initialize_language_detector()
                
                detection_result = await self.language_detector.detect_language(text)
                detect_lang = detection_result["detected_language"]
                logger.debug(f"Auto-detected language for TTS: {detect_lang}")
            except Exception as e:
                logger.warning(f"Error auto-detecting language: {e}")
                detect_lang = "en"  # Default to English if detection fails
        
        # Generate speech from text
        try:
            tts_result = await self.tts_pipeline.synthesize(
                text=text,
                language=detect_lang,
                voice_id=voice_id,
                output_format=output_format,
                speed=speed,
                pitch=pitch,
                options=options
            )
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return failure result
            return {
                "audio_data": None,
                "language": detect_lang,
                "text_length": len(text),
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing time to the result
        tts_result["processing_time"] = processing_time
        
        # Generate a unique filename and save the audio
        audio_data = tts_result.get("audio_content")
        if audio_data:
            timestamp = int(time.time())
            filename = f"tts_{timestamp}_{sanitize_filename(text[:20])}.{output_format}"
            file_path = self.audio_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(audio_data)
            
            # Add file path to result
            tts_result["filename"] = filename
            tts_result["file_path"] = str(file_path)
        
        # Audit and metrics
        if user_id:
            self.audit_logger.log_text_to_speech(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                language=detect_lang,
                text_length=len(text),
                audio_size=len(audio_data) if audio_data else 0,
                voice_id=voice_id or "default",
                output_format=output_format,
                processing_time=processing_time,
                model_id=tts_result.get("model_used", "unknown")
            )
        
        return tts_result
    
    async def _process_audio(self, 
                           audio_content: bytes, 
                           options: Dict[str, Any],
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process audio content using the appropriate pipeline.
        
        Args:
            audio_content: Audio content as bytes
            options: Processing options
            metadata: Additional metadata
            
        Returns:
            Processing result
        """
        # Initialize STT pipeline if needed
        if not hasattr(self, "stt_pipeline") or self.stt_pipeline is None:
            logger.warning("STT pipeline not initialized, initializing now")
            await self._initialize_stt_pipeline()
        
        # Prepare transcription options
        transcription_options = {
            "model_id": options.get("model_id"),
            "enhanced_results": options.get("enhanced_results", False)
        }
        
        try:
            # Use STT pipeline for transcription
            transcription_result = await self.stt_pipeline.transcribe(
                audio_content=audio_content,
                language=options.get("source_language"),
                detect_language=options.get("detect_language", False),
                model_id=options.get("model_id"),
                options=transcription_options
            )
            
            # Add metadata
            transcription_result.update({
                "request_metadata": metadata
            })
            
            return transcription_result
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            
            # Return error result
            return {
                "status": "error",
                "error": f"Error processing audio: {str(e)}",
                "request_metadata": metadata
            }
    
    async def transcribe_speech(
        self,
        audio_content: bytes,
        language: Optional[str] = None,
        detect_language: bool = False,
        model_id: Optional[str] = None,
        options: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe speech from audio.
        
        Args:
            audio_content: Audio content as bytes
            language: Language code
            detect_language: Whether to detect language
            model_id: Specific model to use
            options: Additional options
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with transcription results
        """
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.stt_pipeline:
            logger.warning("STT pipeline not initialized, initializing now")
            await self._initialize_stt_pipeline()
        
        options = options or {}
        
        # Start timing
        start_time = time.time()
        
        # Prepare metadata
        metadata = {
            "user_id": user_id,
            "request_id": request_id or str(uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Transcribe audio
        try:
            transcription_result = await self.stt_pipeline.transcribe(
                audio_content=audio_content,
                language=language,
                detect_language=detect_language,
                model_id=model_id,
                options=options
            )
        except Exception as e:
            logger.error(f"Error transcribing speech: {str(e)}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return failure result
            return {
                "text": "",
                "language": language or "unknown",
                "confidence": 0.0,
                "processing_time": processing_time,
                "success": False,
                "error": str(e),
                "metadata": metadata
            }
        
        # Add processing time if not already present
        if "processing_time" not in transcription_result:
            transcription_result["processing_time"] = time.time() - start_time
        
        # Add metadata
        transcription_result["metadata"] = metadata
        
        # Audit and metrics
        if user_id:
            self.audit_logger.log_speech_to_text(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                language=transcription_result.get("language", language or "unknown"),
                audio_size=len(audio_content),
                text_length=len(transcription_result.get("text", "")),
                confidence=transcription_result.get("confidence", 0.0),
                processing_time=transcription_result.get("processing_time", 0.0),
                model_id=transcription_result.get("model_used", "unknown")
            )
        
        return transcription_result
    
    async def get_stt_languages(self) -> Dict[str, Any]:
        """
        Get supported languages for speech recognition.
        
        Returns:
            Dict with supported languages
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.stt_pipeline:
            logger.warning("STT pipeline not initialized, initializing now")
            await self._initialize_stt_pipeline()
            
        return await self.stt_pipeline.get_supported_languages()
    
    async def get_tts_voices(
        self,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get available voices for TTS.
        
        Args:
            language: Filter voices by language
            
        Returns:
            Dict with available voices
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.tts_pipeline:
            logger.warning("TTS pipeline not initialized, initializing now")
            await self._initialize_tts_pipeline()
            
        return await self.tts_pipeline.get_voices(language)
    
    def get_component(self, component_name: str) -> Any:
        """
        Get a specific component by name.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component instance or None if not found
        """
        components = {
            "translation_pipeline": self.translation_pipeline,
            "language_detector": self.language_detector,
            "anonymizer": self.anonymizer,
            "simplifier": self.simplifier,
            "summarizer": self.summarizer,
            "tts_pipeline": self.tts_pipeline,
            "stt_pipeline": self.stt_pipeline if hasattr(self, "stt_pipeline") else None,
            "pdf_processor": self.pdf_processor,
            "docx_processor": self.docx_processor,
            "ocr_processor": self.ocr_processor,
            "rag_expert": self.rag_expert,
            "audit_logger": self.audit_logger,
            "metrics_collector": self.metrics or self.metrics_collector,
            "veracity_auditor": self.veracity_auditor
        }
        
        return components.get(component_name)
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics and metrics.
        
        Returns:
            Dict with processor statistics
        """
        stats = {
            "initialized": self.initialized,
            "components": {
                "translation_pipeline": self.translation_pipeline is not None,
                "language_detector": self.language_detector is not None,
                "anonymizer": self.anonymizer is not None,
                "simplifier": self.simplifier is not None,
                "summarizer": self.summarizer is not None,
                "tts_pipeline": self.tts_pipeline is not None,
                "stt_pipeline": hasattr(self, "stt_pipeline") and self.stt_pipeline is not None,
                "pdf_processor": self.pdf_processor is not None,
                "docx_processor": self.docx_processor is not None,
                "ocr_processor": self.ocr_processor is not None,
                "rag_expert": self.rag_expert is not None
            },
            "metrics": (self.metrics or self.metrics_collector).get_metrics() if (self.metrics or self.metrics_collector) else {}
        }
        
        return stats
    
    async def process(
        self,
        content: Union[str, bytes],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generic processing method that routes to appropriate pipeline component.
        
        Args:
            content: The content to process (text or binary)
            options: Processing options including:
                operation: The operation to perform (translate, detect_language, etc.)
                source_language: Source language code
                target_language: Target language code (for translation)
                model_id/model_name: Model to use
                ... other operation-specific options
                
        Returns:
            Dict with processing results
        """
        operation = options.get("operation", "").lower()
        
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
        
        # Start timing
        start_time = time.time()
        
        try:
            # Route to appropriate operation
            if operation == "translate":
                if isinstance(content, str):
                    return await self.translate_text(
                        text=content,
                        source_language=options.get("source_language"),
                        target_language=options.get("target_language", "en"),
                        model_id=options.get("model_id") or options.get("model_name"),
                        options=options,
                        user_id=options.get("user_id"),
                        request_id=options.get("request_id")
                    )
            elif operation == "detect_language":
                if isinstance(content, str):
                    return await self.detect_language(
                        text=content,
                        confidence_threshold=options.get("confidence_threshold", 0.6),
                        user_id=options.get("user_id"),
                        request_id=options.get("request_id")
                    )
            elif operation == "simplify":
                if isinstance(content, str):
                    return await self.simplify_text(
                        text=content,
                        level=options.get("level", "medium"),
                        source_language=options.get("source_language"),
                        target_language=options.get("target_language"),
                        model_id=options.get("model_id"),
                        options=options,
                        user_id=options.get("user_id"),
                        request_id=options.get("request_id")
                    )
            elif operation == "anonymize":
                if isinstance(content, str):
                    mode = "redact"
                    if options.get("strategy") == "replace":
                        mode = "replace"
                    elif options.get("strategy") == "remove":
                        mode = "remove"
                        
                    return await self.anonymize_text(
                        text=content,
                        entity_types=options.get("entities"),
                        mode=mode,
                        language=options.get("language"),
                        user_id=options.get("user_id"),
                        request_id=options.get("request_id")
                    )
            elif operation == "speech_to_text" or operation == "transcribe":
                if isinstance(content, bytes):
                    return await self.transcribe_speech(
                        audio_content=content,
                        language=options.get("source_language"),
                        detect_language=options.get("detect_language", False),
                        model_id=options.get("model_id"),
                        options=options.get("parameters", {}),
                        user_id=options.get("user_id"),
                        request_id=options.get("request_id")
                    )
            elif operation == "analyze":
                if isinstance(content, str):
                    # Check if we have an analyze_text method
                    if hasattr(self, "analyze_text") and callable(getattr(self, "analyze_text")):
                        # Use the analyze_text method if it exists
                        return await self.analyze_text(
                            text=content,
                            language=options.get("language", "en"),
                            analyses=options.get("analyses", []),
                            model_id=options.get("model_id"),
                            options=options,
                            user_id=options.get("user_id"),
                            request_id=options.get("request_id")
                        )
                    # Create a basic analysis result with fallback implementations
                    result = {
                        "language": options.get("language", "en"),
                        "text": content,
                        "word_count": len(content.split()),
                        "sentence_count": content.count(".") + content.count("!") + content.count("?"),
                        "processing_time": time.time() - start_time
                    }
                    
                    # Add sentiment analysis fallback
                    if "sentiment" in options.get("analyses", []) or options.get("include_sentiment", False):
                        # Very basic sentiment analysis based on positive/negative word counting
                        positive_words = ["good", "great", "excellent", "amazing", "happy", "love", "like", "best"]
                        negative_words = ["bad", "terrible", "awful", "worst", "hate", "dislike", "poor", "sad"]
                        
                        text_lower = content.lower()
                        positive_count = sum(text_lower.count(word) for word in positive_words)
                        negative_count = sum(text_lower.count(word) for word in negative_words)
                        
                        total = positive_count + negative_count
                        if total == 0:
                            sentiment_score = 0.0
                        else:
                            sentiment_score = (positive_count - negative_count) / max(1, total)
                        
                        sentiment = {
                            "score": sentiment_score,
                            "magnitude": abs(sentiment_score),
                            "label": "positive" if sentiment_score > 0.25 else "negative" if sentiment_score < -0.25 else "neutral"
                        }
                        result["sentiment"] = sentiment
                    
                    # Add entities fallback
                    if "entities" in options.get("analyses", []) or options.get("include_entities", False):
                        result["entities"] = []
                        # This would need a more sophisticated implementation for real entity extraction
                    
                    # Add topics fallback
                    if "topics" in options.get("analyses", []) or options.get("include_topics", False):
                        result["topics"] = []
                        # This would need a more sophisticated implementation for real topic extraction
                    
                    # Add summary fallback
                    if "summary" in options.get("analyses", []) or options.get("include_summary", False):
                        # Extract first sentence and add "..." if content is longer than 1 sentence
                        sentences = content.split(".")
                        if len(sentences) > 1:
                            summary = sentences[0].strip() + "..."
                        else:
                            summary = content
                        result["summary"] = summary
                    
                    return result
            elif operation == "summarize":
                if isinstance(content, str):
                    # Basic fallback summarization by extracting key sentences
                    sentences = content.split(".")
                    total_sentences = len(sentences)
                    
                    # Calculate target summary length
                    max_length = options.get("max_length", None)
                    min_length = options.get("min_length", None)
                    
                    # If no max_length, use 1/3 of original text length
                    if not max_length:
                        max_length = total_sentences // 3 if total_sentences > 3 else 1
                    
                    # Use importance-based extraction (first and last sentences, plus any with keywords)
                    important_sentences = []
                    
                    # Add first sentence
                    if sentences and len(sentences) > 0:
                        important_sentences.append(sentences[0])
                    
                    # Add last sentence if different from first
                    if total_sentences > 1:
                        important_sentences.append(sentences[-1])
                    
                    # Basic summary joining selected sentences
                    summary = ". ".join(important_sentences)
                    if not summary.endswith("."):
                        summary += "."
                    
                    # Return summarization result
                    return {
                        "summary": summary,
                        "language": options.get("language", "en"),
                        "type": options.get("type", "extractive"),
                        "processing_time": time.time() - start_time,
                        "model_used": "fallback_summarizer"
                    }
            
            # If we get here, operation wasn't handled or content was wrong type
            return {
                "error": f"Operation '{operation}' not supported or content type is invalid",
                "processing_time": time.time() - start_time,
                "success": False
            }
            
        except Exception as e:
            logger.error(f"Error in process() operation '{operation}': {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    async def cleanup(self) -> None:
        """
        Clean up resources.
        
        This should be called before application shutdown.
        """
        logger.info("Cleaning up UnifiedProcessor resources")
        
        # Clean up pipelines
        if self.translation_pipeline:
            await self.translation_pipeline.cleanup()
        
        if self.language_detector:
            await self.language_detector.cleanup()
        
        if self.anonymizer:
            await self.anonymizer.cleanup()
        
        if self.simplifier:
            await self.simplifier.cleanup()
        
        if self.summarizer:
            await self.summarizer.cleanup()
        
        if self.tts_pipeline:
            await self.tts_pipeline.cleanup()
        
        if hasattr(self, "stt_pipeline") and self.stt_pipeline:
            await self.stt_pipeline.cleanup()
        
        # Clean up document processors
        if self.ocr_processor:
            await self.ocr_processor.cleanup()
        
        # Clean up RAG components
        if self.rag_expert:
            await self.rag_expert.cleanup()
        
        logger.info("UnifiedProcessor cleanup complete")


# Global accessor function to get the processor instance
async def get_pipeline_processor() -> UnifiedProcessor:
    """
    Get the global pipeline processor instance.
    
    Returns:
        UnifiedProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        logger.info("Creating new UnifiedProcessor instance")
        _processor_instance = UnifiedProcessor()
        await _processor_instance.initialize()
    
    return _processor_instance