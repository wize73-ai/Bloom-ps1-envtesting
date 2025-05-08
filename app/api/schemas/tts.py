"""
Text-to-Speech schemas for CasaLingua API.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
from app.api.schemas.base import BaseResponse

class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"

class TTSRequest(BaseModel):
    """Text-to-speech request model."""
    text: str = Field(
        ...,
        description="Text to convert to speech."
    )
    language: str = Field(
        default="en",
        description="Language code (e.g., 'en', 'es')."
    )
    voice: Optional[str] = Field(
        default=None,
        description="Voice identifier. If not provided, the default voice for the language will be used."
    )
    speed: float = Field(
        default=1.0,
        description="Speech rate multiplier (0.5-2.0)."
    )
    pitch: float = Field(
        default=1.0,
        description="Voice pitch adjustment (0.5-2.0)."
    )
    output_format: AudioFormat = Field(
        default=AudioFormat.MP3,
        description="Output audio format."
    )

class VoiceInfo(BaseModel):
    """Information about a voice."""
    id: str = Field(
        ...,
        description="Voice identifier."
    )
    language: str = Field(
        ...,
        description="Language code."
    )
    name: str = Field(
        ...,
        description="Voice name."
    )
    gender: Optional[str] = Field(
        default=None,
        description="Voice gender (male/female)."
    )
    description: Optional[str] = Field(
        default=None,
        description="Voice description."
    )

class TTSResult(BaseModel):
    """Text-to-speech result model."""
    audio_url: str = Field(
        ...,
        description="URL to access the generated audio."
    )
    format: AudioFormat = Field(
        ...,
        description="Audio format."
    )
    language: str = Field(
        ...,
        description="Language used."
    )
    voice: str = Field(
        ...,
        description="Voice used."
    )
    duration: float = Field(
        ...,
        description="Audio duration in seconds."
    )
    text: str = Field(
        ...,
        description="Original text."
    )
    model_used: str = Field(
        ...,
        description="Model used for synthesis."
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds."
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

class TTSResponse(BaseResponse):
    """Text-to-speech response model."""
    data: TTSResult

class AvailableVoicesResponse(BaseResponse):
    """Response model for available voices."""
    data: Dict[str, Any] = Field(
        ...,
        description="Available voices information."
    )