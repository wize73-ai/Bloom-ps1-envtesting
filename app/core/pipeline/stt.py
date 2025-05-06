"""
Speech-to-Text Module for CasaLingua

This module provides speech recognition capabilities, converting audio
recordings into text in multiple languages with various models.
"""

import os
import io
import uuid
import logging
import asyncio
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, BinaryIO

from app.utils.logging import get_logger

logger = get_logger("casalingua.core.stt")

class STTPipeline:
    """
    Speech-to-Text Pipeline for converting audio to text.
    
    Features:
    - Support for multiple languages
    - Multiple model support (Whisper, Wav2Vec2, etc.)
    - Various input formats (MP3, WAV, OGG)
    - Caching for efficiency
    - Fallback mechanisms for reliability
    """
    
    def __init__(self, model_manager, config: Dict[str, Any] = None, registry_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the STT pipeline.
        
        Args:
            model_manager: Model manager for accessing STT models
            config: Configuration dictionary
            registry_config: Registry configuration for models
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.registry_config = registry_config or {}
        self.initialized = False
        
        # Model type for STT
        self.model_type = "speech_to_text"
        
        # Input settings
        self.supported_formats = ["mp3", "wav", "ogg", "flac", "m4a"]
        
        # Language settings
        self.default_language = "en"
        
        # Audio cache
        self.cache_enabled = self.config.get("stt_cache_enabled", True)
        self.cache_dir = Path(self.config.get("stt_cache_dir", "cache/stt"))
        self.cache_size_limit = self.config.get("stt_cache_size_mb", 500) * 1024 * 1024  # Convert MB to bytes
        
        # Ensure cache directory exists
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Temp directory for processing
        self.temp_dir = Path(self.config.get("temp_dir", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("STT pipeline created (not yet initialized)")
    
    async def initialize(self) -> None:
        """
        Initialize the STT pipeline.
        
        This loads necessary models and prepares the pipeline.
        """
        if self.initialized:
            logger.warning("STT pipeline already initialized")
            return
        
        logger.info("Initializing STT pipeline")
        
        # Verify STT models are available
        try:
            # Load the STT model through the model manager
            model_info = await self.model_manager.load_model(self.model_type)
            if model_info:
                logger.info(f"STT model loaded successfully")
            else:
                logger.warning("STT model loading failed, will use fallbacks")
        except Exception as e:
            logger.warning(f"Error loading STT model: {str(e)}")
            logger.warning("STT functionality will be limited to fallbacks")
            
        # Clean up cache if enabled
        if self.cache_enabled:
            await self._cleanup_cache()
            
            # Start background task for periodic cleanup
            asyncio.create_task(self._periodic_cache_cleanup())
        
        self.initialized = True
        logger.info("STT pipeline initialization complete")
    
    async def transcribe(self, 
                        audio_content: bytes, 
                        language: Optional[str] = None,
                        detect_language: bool = False,
                        model_id: Optional[str] = None,
                        audio_format: Optional[str] = None,
                        options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transcribe speech from audio content.
        
        Args:
            audio_content: Audio content as bytes
            language: Language code (e.g., 'en', 'es')
            model_id: Specific model to use
            detect_language: Whether to detect language from audio
            audio_format: Format of audio (mp3, wav, etc.)
            options: Additional options for transcription
            
        Returns:
            Dict with transcription results:
            - text: Transcribed text
            - language: Detected or specified language
            - confidence: Confidence score
            - segments: Time-aligned segments (if available)
            - duration: Audio duration in seconds
            - model_used: Name of model used
        """
        if not self.initialized:
            await self.initialize()
        
        options = options or {}
        logger.debug(f"Transcribing audio of length {len(audio_content)} bytes")
        
        # Generate cache key if caching is enabled
        cache_key = None
        if self.cache_enabled:
            cache_key = self._generate_cache_key(audio_content, language, model_id, options)
            
            # Check cache
            cached_result = await self._get_cached_transcription(cache_key)
            if cached_result:
                logger.debug(f"Using cached transcription")
                return cached_result
        
        # Determine audio format if not provided
        if not audio_format:
            audio_format = self._detect_audio_format(audio_content)
            logger.debug(f"Detected audio format: {audio_format}")
        
        # Save audio to temporary file for processing
        temp_file = self._save_audio_to_temp(audio_content, audio_format)
        
        try:
            # Prepare input data for model
            input_data = {
                "audio_file": str(temp_file),
                "audio_content": audio_content,
                "language": language,
                "detect_language": detect_language,
                "parameters": {
                    "model_id": model_id,
                    **options
                }
            }
            
            # Run STT model through model manager
            start_time = time.time()
            result = await self.model_manager.run_model(
                self.model_type,
                "transcribe",
                input_data
            )
            processing_time = time.time() - start_time
            
            # Extract result
            transcription = {}
            if isinstance(result, dict):
                if "result" in result:
                    if isinstance(result["result"], str):
                        # Simple string result
                        transcription = {
                            "text": result["result"],
                            "language": result.get("language", language or self.default_language),
                            "confidence": result.get("confidence", 1.0),
                            "model_used": result.get("model_used", self.model_type),
                            "processing_time": processing_time
                        }
                    elif isinstance(result["result"], dict):
                        # Dict result with more information
                        transcription = result["result"]
                        transcription["processing_time"] = processing_time
                        # Ensure model_used is set
                        if "model_used" not in transcription:
                            transcription["model_used"] = result.get("model_used", self.model_type)
                else:
                    # Try to interpret the result directly
                    transcription = {
                        "text": str(result),
                        "language": language or self.default_language,
                        "confidence": 0.5,
                        "model_used": self.model_type,
                        "processing_time": processing_time
                    }
            else:
                # Direct string result
                transcription = {
                    "text": str(result),
                    "language": language or self.default_language,
                    "confidence": 0.5,
                    "model_used": self.model_type,
                    "processing_time": processing_time
                }
            
            # Add additional information
            transcription["audio_format"] = audio_format
            transcription["audio_size"] = len(audio_content)
            
            # Cache the result if enabled
            if self.cache_enabled and cache_key:
                await self._cache_transcription(cache_key, transcription)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing speech: {str(e)}", exc_info=True)
            
            # Try fallback approach
            try:
                fallback_result = await self._fallback_transcription(
                    audio_content, language, temp_file, audio_format, options
                )
                
                if fallback_result:
                    # Mark as fallback
                    fallback_result["fallback"] = True
                    return fallback_result
                    
            except Exception as fallback_e:
                logger.error(f"Fallback transcription failed: {str(fallback_e)}", exc_info=True)
            
            # Return error if all approaches failed
            return {
                "status": "error",
                "error": f"Speech transcription failed: {str(e)}",
                "language": language or self.default_language,
                "text": ""
            }
        finally:
            # Clean up temporary file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
    
    async def get_supported_languages(self) -> Dict[str, Any]:
        """
        Get available languages for speech recognition.
        
        Returns:
            Dict with supported languages
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Try to get languages from model
            input_data = {}
            
            result = await self.model_manager.run_model(
                self.model_type,
                "get_languages",
                input_data
            )
            
            if isinstance(result, dict) and "result" in result:
                languages = result["result"]
                return {
                    "status": "success",
                    "languages": languages,
                    "default_language": self.default_language
                }
            
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}", exc_info=True)
        
        # Fallback - return basic language list
        return {
            "status": "success",
            "languages": [
                {"code": "en", "name": "English"},
                {"code": "es", "name": "Spanish"},
                {"code": "fr", "name": "French"},
                {"code": "de", "name": "German"},
                {"code": "it", "name": "Italian"},
                {"code": "pt", "name": "Portuguese"},
                {"code": "zh", "name": "Chinese"},
                {"code": "ja", "name": "Japanese"},
                {"code": "ko", "name": "Korean"},
                {"code": "ru", "name": "Russian"}
            ],
            "default_language": self.default_language
        }
    
    async def _fallback_transcription(self,
                                     audio_content: bytes,
                                     language: Optional[str],
                                     temp_file: Path,
                                     audio_format: str,
                                     options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fallback STT implementation when primary model fails.
        
        Args:
            audio_content: Audio content as bytes
            language: Language code
            temp_file: Path to temporary audio file
            audio_format: Audio format
            options: Additional options
            
        Returns:
            Dict with transcription results or None on failure
        """
        logger.info("Using fallback STT implementation")
        
        try:
            # Try to use SpeechRecognition library
            import speech_recognition as sr
            
            # Create recognizer
            r = sr.Recognizer()
            
            # Convert file to AudioData
            with sr.AudioFile(str(temp_file)) as source:
                audio_data = r.record(source)
            
            # Determine language code format for recognizers
            lang_code = language or self.default_language
            
            # Try Google Speech Recognition
            try:
                text = r.recognize_google(audio_data, language=lang_code)
                return {
                    "text": text,
                    "language": lang_code,
                    "confidence": 0.7,
                    "model_used": "google_speech",
                    "audio_format": audio_format
                }
            except Exception as google_e:
                logger.warning(f"Google Speech Recognition failed: {str(google_e)}")
            
            # Try Whisper (if available)
            try:
                text = r.recognize_whisper(audio_data, language=lang_code)
                return {
                    "text": text,
                    "language": lang_code,
                    "confidence": 0.6,
                    "model_used": "whisper_fallback",
                    "audio_format": audio_format
                }
            except Exception as whisper_e:
                logger.warning(f"Whisper recognition failed: {str(whisper_e)}")
            
        except ImportError:
            logger.warning("SpeechRecognition library not available for fallback")
            
        except Exception as e:
            logger.error(f"Error in fallback transcription: {str(e)}", exc_info=True)
        
        return None
    
    def _detect_audio_format(self, audio_content: bytes) -> str:
        """
        Detect the format of audio content from magic bytes.
        
        Args:
            audio_content: Audio content as bytes
            
        Returns:
            Detected audio format
        """
        # Check magic bytes to determine format
        if audio_content.startswith(b'ID3') or audio_content.startswith(b'\xff\xfb'):
            return "mp3"
        elif audio_content.startswith(b'RIFF'):
            return "wav"
        elif audio_content.startswith(b'OggS'):
            return "ogg"
        elif audio_content.startswith(b'fLaC'):
            return "flac"
        
        # Default to mp3 if unable to determine
        return "mp3"
    
    def _save_audio_to_temp(self, audio_content: bytes, audio_format: str) -> Path:
        """
        Save audio content to a temporary file.
        
        Args:
            audio_content: Audio content as bytes
            audio_format: Audio format
            
        Returns:
            Path to temporary file
        """
        # Create unique filename
        filename = f"stt_input_{uuid.uuid4()}.{audio_format}"
        file_path = self.temp_dir / filename
        
        # Write content to file
        with open(file_path, "wb") as f:
            f.write(audio_content)
        
        return file_path
    
    def _generate_cache_key(self,
                          audio_content: bytes,
                          language: Optional[str],
                          model_id: Optional[str],
                          options: Dict[str, Any]) -> str:
        """
        Generate a cache key for STT output.
        
        Args:
            audio_content: Audio content
            language: Language code
            model_id: Model ID
            options: Additional options
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create hash of audio content
        content_hash = hashlib.md5(audio_content).hexdigest()
        
        # Create string with parameters
        params = f"{language}|{model_id}|{str(options)}"
        params_hash = hashlib.md5(params.encode()).hexdigest()
        
        # Combine for final key
        return f"{content_hash}_{params_hash}"
    
    async def _get_cached_transcription(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached transcription if available.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached transcription or None
        """
        if not self.cache_enabled:
            return None
        
        # Check if cache file exists
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                # Read cache file
                import json
                with open(cache_file, "r") as f:
                    cached_result = json.load(f)
                
                # Update access time
                os.utime(cache_file, None)
                
                logger.debug(f"Using cached transcription from {cache_file}")
                return cached_result
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        
        return None
    
    async def _cache_transcription(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Cache transcription result for future use.
        
        Args:
            cache_key: Cache key
            result: Transcription result to cache
        """
        if not self.cache_enabled:
            return
        
        try:
            # Create cache file
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Write to cache
            import json
            with open(cache_file, "w") as f:
                json.dump(result, f)
                
            logger.debug(f"Cached transcription result to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching transcription: {str(e)}", exc_info=True)
    
    async def _cleanup_cache(self) -> None:
        """Clean up STT cache to stay within size limits."""
        if not self.cache_enabled:
            return
        
        try:
            # Get all cache files
            cache_files = list(self.cache_dir.glob("*.json"))
            
            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Check if we need to clean up
            if total_size <= self.cache_size_limit:
                return
            
            logger.info(f"STT cache size ({total_size / 1024 / 1024:.1f} MB) exceeds limit "
                       f"({self.cache_size_limit / 1024 / 1024:.1f} MB), cleaning up")
            
            # Sort by access time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_atime)
            
            # Remove files until we're under the limit
            for file in cache_files:
                if total_size <= self.cache_size_limit * 0.8:  # Clean up to 80% of limit
                    break
                
                file_size = file.stat().st_size
                file.unlink()
                total_size -= file_size
                logger.debug(f"Removed cache file: {file.name}")
            
            logger.info(f"STT cache cleanup complete, new size: {total_size / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error cleaning up STT cache: {str(e)}", exc_info=True)
    
    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up STT cache."""
        if not self.cache_enabled:
            return
        
        # Run cleanup every hour
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour
                await self._cleanup_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cache cleanup: {str(e)}", exc_info=True)
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def cleanup(self) -> None:
        """
        Clean up STT resources.
        
        This should be called before application shutdown.
        """
        logger.info("Cleaning up STT resources")
        
        # Clean up temp directory
        try:
            for temp_file in self.temp_dir.glob("stt_input_*"):
                temp_file.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
        
        # Final cache cleanup
        if self.cache_enabled:
            await self._cleanup_cache()