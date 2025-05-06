"""
Speech processing schemas for CasaLingua API.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
from datetime import datetime
from app.api.schemas.base import BaseResponse

class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    M4A = "m4a"

class STTRequest(BaseModel):
    """Speech-to-text request model."""
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en', 'es'). If not provided, the system will attempt to detect the language."
    )
    detect_language: bool = Field(
        default=False,
        description="Whether to automatically detect language from audio."
    )
    model_id: Optional[str] = Field(
        default=None,
        description="ID of the specific model to use for transcription."
    )
    audio_format: Optional[AudioFormat] = Field(
        default=None,
        description="Format of the audio file. If not provided, the system will attempt to detect it."
    )
    enhanced_results: bool = Field(
        default=False,
        description="Whether to include enhanced results such as timestamps, speaker identification, etc."
    )

class TranscriptionSegment(BaseModel):
    """Time-aligned segment of transcription."""
    text: str = Field(
        ...,
        description="Transcribed text for this segment."
    )
    start: float = Field(
        ...,
        description="Start time of segment in seconds."
    )
    end: float = Field(
        ...,
        description="End time of segment in seconds."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for this segment."
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker identifier if available."
    )

class STTResult(BaseModel):
    """Speech-to-text result model."""
    text: str = Field(
        ...,
        description="Transcribed text."
    )
    language: str = Field(
        ...,
        description="Detected or specified language."
    )
    confidence: float = Field(
        default=1.0,
        description="Overall confidence score."
    )
    segments: Optional[List[TranscriptionSegment]] = Field(
        default=None,
        description="Time-aligned segments if available."
    )
    duration: Optional[float] = Field(
        default=None,
        description="Audio duration in seconds."
    )
    model_used: str = Field(
        ...,
        description="Model used for transcription."
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds."
    )
    audio_format: str = Field(
        ...,
        description="Format of the audio."
    )
    fallback: Optional[bool] = Field(
        default=False,
        description="Whether a fallback model was used."
    )
    performance_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance metrics."
    )
    memory_usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Memory usage statistics."
    )
    operation_cost: Optional[float] = Field(
        default=None,
        description="Estimated operation cost."
    )

class STTResponse(BaseResponse):
    """Speech-to-text response model."""
    data: STTResult

class SupportedLanguageInfo(BaseModel):
    """Information about a supported language."""
    code: str = Field(
        ...,
        description="Language code."
    )
    name: str = Field(
        ...,
        description="Language name."
    )
    supported_models: Optional[List[str]] = Field(
        default=None,
        description="List of models supporting this language."
    )
    quality_rating: Optional[float] = Field(
        default=None,
        description="Quality rating for this language."
    )

class SupportedLanguagesResponse(BaseResponse):
    """Response model for supported languages."""
    data: Dict[str, Any] = Field(
        ...,
        description="Supported languages information."
    )