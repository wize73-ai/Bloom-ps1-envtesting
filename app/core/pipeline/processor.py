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

# Import pipeline components
from app.core.pipeline.translator import TranslationPipeline
from app.core.pipeline.language_detector import LanguageDetector
# Commented out imports that might be missing
# from app.core.pipeline.anonymizer import Anonymizer
# from app.core.pipeline.simplifier import Simplifier
# from app.core.pipeline.summarizer import Summarizer
# from app.core.pipeline.tts import TTSPipeline
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
        self.anonymizer = Anonymizer(
            self.model_manager,
            self.config
        )
        await self.anonymizer.initialize()
        logger.info("Anonymizer initialization complete")
    
    async def _initialize_simplifier(self) -> None:
        """Initialize the simplifier component."""
        logger.info("Initializing simplifier")
        self.simplifier = Simplifier(
            self.model_manager,
            self.config
        )
        await self.simplifier.initialize()
        logger.info("Simplifier initialization complete")
    
    async def _initialize_summarizer(self) -> None:
        """Initialize the summarizer component."""
        logger.info("Initializing summarizer")
        self.summarizer = Summarizer(
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
            self.audit_logger.log_language_detection(
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
            translation_result = await self.translation_pipeline.translate(
                text=text,
                source_language=source_lang,
                target_language=target_language,
                quality_level=quality_level,
                model_id=model_id,
                options=options
            )
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
        
        # Add processing time to the result
        translation_result["processing_time"] = processing_time
        
        # Audit and metrics
        if user_id:
            self.audit_logger.log_translation(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                source_language=source_lang,
                target_language=target_language,
                text_length=len(text),
                processing_time=processing_time,
                model_id=translation_result.get("model_used", "unknown"),
                quality_level=quality_level
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
            simplification_result = await self.simplifier.simplify_text(
                text=text_to_simplify,
                level=level,
                language=target_lang,
                model_id=model_id,
                options=options
            )
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