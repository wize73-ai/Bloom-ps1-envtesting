from typing import Dict, List, Any, Optional
from pydantic import BaseModel, validator
from .base import BaseResponse

class TextAnalysisRequest(BaseModel):
    text: str
    language: Optional[str] = None
    include_sentiment: bool = True
    include_entities: bool = True
    include_topics: bool = False
    include_summary: bool = False
    analyses: Optional[List[str]] = None  # Added for compatibility with test scripts
    model_id: Optional[str] = None  # Added to fix the analyze endpoint

class TextAnalysisResult(BaseModel):
    text: str
    language: str
    sentiment: Optional[Dict[str, float]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    topics: Optional[List[Dict[str, float]]] = None
    summary: Optional[str] = None
    word_count: Optional[int] = None
    sentence_count: Optional[int] = None
    process_time: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, float]] = None
    operation_cost: Optional[float] = None
    accuracy_score: Optional[float] = None
    truth_score: Optional[float] = None

class TextAnalysisResponse(BaseResponse[TextAnalysisResult]):
    pass

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    output_format: str = "mp3"
    model_id: Optional[str] = None
    
    @validator('speed')
    def validate_speed(cls, v):
        if v < 0.5 or v > 2.0:
            raise ValueError('Speed must be between 0.5 and 2.0')
        return v
    
    @validator('pitch')
    def validate_pitch(cls, v):
        if v < 0.5 or v > 2.0:
            raise ValueError('Pitch must be between 0.5 and 2.0')
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v):
        supported_formats = ["mp3", "wav", "ogg"]
        if v not in supported_formats:
            raise ValueError(f'Output format must be one of {supported_formats}')
        return v

class TTSResult(BaseModel):
    text: str
    language: str
    voice: str
    audio_url: str
    audio_file: Optional[str] = None
    format: str
    duration: float = 0.0
    process_time: float = 0.0
    model_used: str = "tts"
    fallback: bool = False
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, float]] = None
    operation_cost: Optional[float] = None

class TTSResponse(BaseResponse[TTSResult]):
    pass
    
class VoiceInfoResponse(BaseResponse):
    pass