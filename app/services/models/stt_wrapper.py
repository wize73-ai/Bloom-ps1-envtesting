"""
Speech-to-Text model wrapper for CasaLingua.

This module provides wrapper functionality for speech recognition models,
supporting various architectures like Whisper, Wav2Vec2, etc.
"""

import os
import io
import time
import logging
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from app.utils.logging import get_logger
from app.services.models.wrapper_base import BaseModelWrapper

logger = get_logger("casalingua.models.stt_wrapper")

class STTModelWrapper(BaseModelWrapper):
    """
    Wrapper for speech-to-text models.
    
    Supports:
    - Whisper models (tiny, base, small, medium, large)
    - Wav2Vec2 models for specific languages
    - Other compatible ASR models
    """
    
    def __init__(
        self,
        model_manager=None,
        model_config: Dict[str, Any] = None,
        registry_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the STT model wrapper.
        
        Args:
            model_manager: Model manager instance
            model_config: Model configuration
            registry_config: Registry configuration
        """
        super().__init__(model_manager, model_config, registry_config)
        
        # Model type identifier
        self.model_type = "speech_to_text"
        
        # Default model
        self.default_model = "whisper_base"
        
        # Model options
        self.available_models = {
            "whisper_tiny": {"name": "openai/whisper-tiny", "multilingual": True},
            "whisper_base": {"name": "openai/whisper-base", "multilingual": True},
            "whisper_small": {"name": "openai/whisper-small", "multilingual": True},
            "whisper_medium": {"name": "openai/whisper-medium", "multilingual": True},
            "whisper_large": {"name": "openai/whisper-large-v2", "multilingual": True},
            "wav2vec2_en": {"name": "facebook/wav2vec2-base-960h", "multilingual": False},
            "wav2vec2_multilingual": {"name": "facebook/wav2vec2-xls-r-300m", "multilingual": True}
        }
        
        # Load model registry if available
        if registry_config and "speech_to_text" in registry_config:
            self.from_registry(registry_config["speech_to_text"])
        
        # Supported languages
        self.supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", 
            "nl", "pl", "tr", "ar", "hi", "cs", "da", "fi", "hu", "sv"
        ]
        
        # Initialize model and tokenizer
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self.is_initialized = False
        
        logger.info(f"STT model wrapper created for {self.model_type}")
    
    async def initialize(self) -> bool:
        """
        Initialize the model.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        if self.is_initialized:
            logger.debug("STT model already initialized")
            return True
        
        logger.info(f"Initializing STT model: {self.default_model}")
        
        try:
            # Import required libraries
            try:
                import torch
                import transformers
                from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
            except ImportError as e:
                logger.error(f"Required libraries not found: {str(e)}")
                return False
            
            # Get model config
            model_config = self.available_models.get(
                self.model_config.get("model_id", self.default_model),
                self.available_models[self.default_model]
            )
            
            model_name = model_config["name"]
            
            # Create pipeline
            if "whisper" in model_name:
                # Whisper model
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Move to device
                if not torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                
                logger.info(f"Loaded Whisper model: {model_name}")
                
            elif "wav2vec2" in model_name:
                # Wav2Vec2 model
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                logger.info(f"Loaded Wav2Vec2 model: {model_name}")
                
            else:
                # Generic ASR model
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                logger.info(f"Loaded ASR model: {model_name}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing STT model: {str(e)}", exc_info=True)
            return False
    
    async def transcribe(self, 
                        audio_file: Union[str, bytes, Path],
                        language: Optional[str] = None,
                        detect_language: bool = False,
                        options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Transcribe speech from audio.
        
        Args:
            audio_file: Path to audio file or audio content as bytes
            language: Language code
            detect_language: Whether to detect language
            options: Additional options
            
        Returns:
            Dict with transcription results
        """
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("Failed to initialize STT model")
        
        options = options or {}
        logger.debug(f"Transcribing audio: {audio_file if isinstance(audio_file, str) else 'bytes data'}")
        
        start_time = time.time()
        
        try:
            # Process audio file
            if isinstance(audio_file, (str, Path)):
                # File path provided
                audio_path = str(audio_file)
            else:
                # Audio content as bytes
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp.write(audio_file)
                    audio_path = tmp.name
            
            # Import required libraries
            import torch
            import librosa
            
            try:
                # Load audio with librosa
                y, sr = librosa.load(audio_path, sr=16000)
                
                # For Whisper model
                if hasattr(self, 'processor') and hasattr(self, 'model'):
                    input_features = self.processor(
                        y, 
                        sampling_rate=sr, 
                        return_tensors="pt"
                    ).input_features
                    
                    # Move to device
                    input_features = input_features.to(self.device)
                    
                    # Set language and task for generation
                    forced_decoder_ids = None
                    if language:
                        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                            language=language, task="transcribe"
                        )
                    
                    # Generate transcription
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            input_features,
                            forced_decoder_ids=forced_decoder_ids,
                            max_length=448
                        )
                    
                    # Decode transcription
                    transcription = self.processor.batch_decode(
                        predicted_ids, skip_special_tokens=True
                    )[0]
                    
                    # Detect language if needed
                    detected_language = None
                    if detect_language:
                        with torch.no_grad():
                            # Generate with language detection
                            detected_ids = self.model.generate(
                                input_features,
                                max_length=448,
                                language="",
                                task="transcribe"
                            )
                            
                            # Get language info
                            detected_language_token = self.processor.batch_decode(
                                detected_ids[:, 1:2], skip_special_tokens=False
                            )[0]
                            
                            # Extract language code
                            import re
                            lang_match = re.search(r'<|([a-z]{2,3})|>', detected_language_token)
                            detected_language = lang_match.group(1) if lang_match else None
                    
                    result = {
                        "text": transcription,
                        "language": detected_language or language or "en",
                        "confidence": 0.9,
                        "model_used": "whisper",
                        "audio_format": os.path.splitext(audio_path)[1][1:] if isinstance(audio_file, str) else "mp3"
                    }
                    
                # For pipeline-based models
                elif hasattr(self, 'pipe'):
                    # Transcribe with pipeline
                    pipe_result = self.pipe(
                        y,
                        chunk_length_s=30,
                        batch_size=8,
                        return_timestamps=options.get("return_timestamps", False)
                    )
                    
                    # Extract result
                    if isinstance(pipe_result, dict):
                        transcription = pipe_result.get("text", "")
                        chunks = pipe_result.get("chunks", [])
                    else:
                        transcription = pipe_result
                        chunks = []
                    
                    # Create segments if timestamps are available
                    segments = None
                    if chunks:
                        segments = []
                        for chunk in chunks:
                            segments.append({
                                "text": chunk["text"],
                                "start": chunk["timestamp"][0] if isinstance(chunk["timestamp"], list) else chunk["timestamp"][0][0],
                                "end": chunk["timestamp"][1] if isinstance(chunk["timestamp"], list) else chunk["timestamp"][-1][1],
                                "confidence": chunk.get("confidence", 0.9)
                            })
                    
                    result = {
                        "text": transcription,
                        "language": language or "en",
                        "confidence": 0.9,
                        "segments": segments,
                        "model_used": "pipeline",
                        "audio_format": os.path.splitext(audio_path)[1][1:] if isinstance(audio_file, str) else "mp3"
                    }
                    
                else:
                    raise RuntimeError("No valid model initialized")
                
            finally:
                # Clean up temp file if needed
                if not isinstance(audio_file, str) and 'audio_path' in locals():
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass
            
            # Add duration estimate
            result["duration"] = len(y) / sr
            
            # Add processing time
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            
            # Clean up temp file if needed
            if not isinstance(audio_file, str) and 'audio_path' in locals():
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass
            
            raise
    
    async def get_languages(self) -> Dict[str, Any]:
        """
        Get supported languages.
        
        Returns:
            Dict with supported languages
        """
        if not self.is_initialized:
            await self.initialize()
        
        return {
            "languages": [
                {"code": lang, "name": self._get_language_name(lang)}
                for lang in self.supported_languages
            ],
            "default_language": "en"
        }
    
    def _get_device(self) -> str:
        """
        Get the device to use for inference.
        
        Returns:
            Device string
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _get_language_name(self, code: str) -> str:
        """
        Get language name from code.
        
        Args:
            code: Language code
            
        Returns:
            Language name
        """
        language_map = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "ar": "Arabic",
            "hi": "Hindi",
            "cs": "Czech",
            "da": "Danish",
            "fi": "Finnish",
            "hu": "Hungarian",
            "sv": "Swedish"
        }
        
        return language_map.get(code, code)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data with the model.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing result
        """
        # Detect method from input
        method = input_data.get("method", "transcribe")
        
        # Call appropriate method
        if method == "transcribe":
            # Extract necessary parameters
            audio_file = input_data.get("audio_file")
            audio_content = input_data.get("audio_content")
            language = input_data.get("language")
            detect_language = input_data.get("detect_language", False)
            parameters = input_data.get("parameters", {})
            
            # Use file or content
            if audio_file:
                return await self.transcribe(audio_file, language, detect_language, parameters)
            elif audio_content:
                return await self.transcribe(audio_content, language, detect_language, parameters)
            else:
                raise ValueError("No audio file or content provided")
        
        elif method == "get_languages":
            return await self.get_languages()
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'model') and self.model:
            del self.model
        
        if hasattr(self, 'processor') and self.processor:
            del self.processor
        
        if hasattr(self, 'pipe') and self.pipe:
            del self.pipe
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.is_initialized = False
        logger.info("STT model resources cleaned up")