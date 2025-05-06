from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .base import BaseResponse

class VerificationIssue(BaseModel):
    """Issue found during verification"""
    type: str
    severity: str
    message: str
    details: Optional[Dict[str, Any]] = None

class VerificationRequest(BaseModel):
    """Request for translation verification"""
    source_text: str = Field(..., description="Original source text")
    translation: str = Field(..., description="Translation to verify")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")

class VerificationResult(BaseModel):
    verified: bool
    score: float
    confidence: float
    issues: Optional[List[Dict[str, Any]]]
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    metrics: Optional[Dict[str, Any]]
    process_time: float
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, float]] = None
    operation_cost: Optional[float] = None
    accuracy_score: Optional[float] = None
    truth_score: Optional[float] = None

class VerificationResponse(BaseResponse[VerificationResult]):
    pass
