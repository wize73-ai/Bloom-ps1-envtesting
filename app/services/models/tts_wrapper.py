"""
Model Wrapper for Text-to-Speech (TTS) models

This wrapper handles TTS models using the transformers pipeline API or direct gTTS implementation
"""

import os
import io
import time
import logging
from typing import Dict, Any, Optional

# Import gtts for fallback
from gtts import gTTS

# Configure logging
logger = logging.getLogger(__name__)

class TTSWrapper:
    """
    Wrapper for Text-to-Speech models
    """
    
    def __init__(self, model_config: Dict[str, Any], model):
        """
        Initialize the TTS wrapper
        
        Args:
            model_config (Dict[str, Any]): Model configuration
            model: The model instance
        """
        self.model_config = model_config
        self.model = model
        self.name = model_config.get("model_name", "tts_model")
        
        # Determine if we're using a pipeline
        self.is_pipeline = model_config.get("type") == "pipeline" or model_config.get("use_pipeline", False)
        
        # Determine model type
        if hasattr(model, "task") and model.task:
            self.model_type = model.task
        else:
            self.model_type = model_config.get("pipeline_task", "text-to-speech")
            
        logger.info(f"Initialized TTS wrapper for model: {self.name}, is_pipeline: {self.is_pipeline}")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text input to generate speech
        
        Args:
            input_data (Dict[str, Any]): Input data with text and parameters
            
        Returns:
            Dict[str, Any]: Result with audio data
        """
        start_time = time.time()
        
        # Extract text and parameters
        text = input_data.get("text", "")
        if not text:
            logger.warning("Empty text provided to TTS model")
            return {"result": b"", "error": "No text provided"}
        
        source_language = input_data.get("source_language", "en")
        params = input_data.get("parameters", {})
        
        # Log the request
        logger.info(f"TTS request: text='{text[:30]}...', language={source_language}")
        
        # Try using the pipeline model
        try:
            if self.is_pipeline and self.model is not None:
                # Process using pipeline
                logger.info(f"Processing with TTS pipeline: {self.name}")
                
                # Prepare parameters for the pipeline
                pipeline_params = {
                    "text": text
                }
                
                # Add any additional parameters from request
                if "voice" in params:
                    pipeline_params["voice"] = params.get("voice")
                    
                if "speed" in params:
                    pipeline_params["rate"] = params.get("speed")
                    
                # Run the pipeline
                result = self.model(**pipeline_params)
                
                # Extract audio data from result
                if isinstance(result, dict) and "audio" in result:
                    audio_data = result["audio"]
                elif hasattr(result, "audio"):
                    audio_data = result.audio
                else:
                    # Try to extract bytes directly
                    audio_data = result
                
                # Check if we got audio data
                if audio_data is not None:
                    processing_time = time.time() - start_time
                    return {
                        "result": audio_data,
                        "metadata": {
                            "model_used": self.name,
                            "duration": processing_time,
                            "text_length": len(text),
                            "language": source_language
                        }
                    }
                else:
                    raise ValueError("No audio content returned from TTS pipeline")
                    
            else:
                # Fallback to gTTS since model is not available or failed
                logger.warning(f"TTS model not available as pipeline, falling back to gTTS")
                return self._fallback_gtts(text, source_language, params)
                
        except Exception as e:
            logger.error(f"Error in TTS processing: {str(e)}", exc_info=True)
            # Fallback to gTTS
            logger.info("Falling back to gTTS for speech synthesis")
            return self._fallback_gtts(text, source_language, params)
    
    def _fallback_gtts(self, text: str, language: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use gTTS as fallback for generating speech
        
        Args:
            text (str): Text to synthesize
            language (str): Language code
            params (Dict[str, Any]): Additional parameters
            
        Returns:
            Dict[str, Any]: Result with audio data
        """
        start_time = time.time()
        
        try:
            # Ensure language is in 2-letter format for gTTS
            if len(language) > 2 and '-' in language:
                language = language.split('-')[0]
                
            # Create audio buffer
            buffer = io.BytesIO()
            
            # Create gTTS object and save to buffer
            slow = params.get("speed", 1.0) < 0.8  # Use slow mode if speed is < 0.8
            tts = gTTS(text=text, lang=language, slow=slow)
            tts.write_to_fp(buffer)
            
            # Get audio content
            buffer.seek(0)
            audio_content = buffer.read()
            
            processing_time = time.time() - start_time
            
            return {
                "result": audio_content,
                "metadata": {
                    "model_used": "gtts_fallback",
                    "duration": processing_time,
                    "text_length": len(text),
                    "language": language,
                    "format": "mp3"
                }
            }
        except Exception as e:
            logger.error(f"Error generating speech with gTTS fallback: {str(e)}", exc_info=True)
            
            # Try to return emergency audio content
            try:
                # Check for pre-generated emergency files
                emergency_locations = [
                    "audio/tts_emergency_0.mp3",
                    "audio/tts_emergency_1.mp3",
                    "audio/tts_emergency_2.mp3",
                    "temp/tts_emergency_0.mp3",
                    "temp/emergency_fallback.mp3"
                ]
                
                # Try each location
                for location in emergency_locations:
                    if os.path.exists(location):
                        with open(location, "rb") as f:
                            audio_content = f.read()
                        
                        logger.info(f"Using pre-generated emergency audio: {location}")
                        return {
                            "result": audio_content,
                            "metadata": {
                                "model_used": "emergency_audio",
                                "duration": 1.0,  # Approximate duration
                                "text_length": len(text),
                                "language": language,
                                "format": "mp3"
                            }
                        }
                
                # If no emergency files found, return minimal audio
                logger.warning("No emergency audio files found, returning empty audio")
                
            except Exception as e2:
                logger.error(f"Error using emergency audio: {str(e2)}", exc_info=True)
            
            # Return minimal result to prevent complete failure
            return {
                "result": b'\xFF\xFB\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00' * 100,  # Minimal MP3 header
                "error": f"Failed to generate speech: {str(e)}",
                "metadata": {
                    "model_used": "empty_fallback",
                    "duration": 0.1,
                    "text_length": len(text),
                    "language": language,
                    "format": "mp3"
                }
            }
    
    def get_voices(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get available voices for TTS
        
        Args:
            input_data (Dict[str, Any]): Input data with parameters
            
        Returns:
            Dict[str, Any]: Result with available voices
        """
        # Get language filter
        parameters = input_data.get("parameters", {})
        language = parameters.get("language")
        
        # Default voices by language
        default_voices = {
            "en": "en-us-1",  # English
            "es": "es-es-1",  # Spanish
            "fr": "fr-fr-1",  # French
            "de": "de-de-1",  # German
            "it": "it-it-1",  # Italian
            "pt": "pt-br-1",  # Portuguese
            "nl": "nl-nl-1",  # Dutch
            "ru": "ru-ru-1",  # Russian
            "zh": "zh-cn-1",  # Chinese
            "ja": "ja-jp-1",  # Japanese
            "ko": "ko-kr-1",  # Korean
            "ar": "ar-sa-1",  # Arabic
            "hi": "hi-in-1",  # Hindi
        }
        
        # Filter voices by language if specified
        voices = []
        for lang, voice_id in default_voices.items():
            if language is None or lang == language:
                voices.append({
                    "id": voice_id,
                    "language": lang,
                    "name": f"{lang.upper()} Voice {voice_id.split('-')[-1]}",
                    "gender": "female" if int(voice_id.split('-')[-1]) % 2 == 0 else "male"
                })
        
        return {
            "result": {
                "voices": voices,
                "default_voice": default_voices.get(language or "en")
            }
        }