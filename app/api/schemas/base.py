from typing import Optional, List, Any, Dict, Generic, TypeVar
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime

T = TypeVar('T')

class BaseRequest(BaseModel):
    """Base class for all request models."""
    pass

class StatusEnum(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ErrorDetail(BaseModel):
    code: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None

class MetadataModel(BaseModel):
    request_id: str
    timestamp: datetime
    version: str
    process_time: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, float]] = None
    operation_cost: Optional[float] = None
    accuracy_score: Optional[float] = None
    truth_score: Optional[float] = None

class BaseResponse(BaseModel, Generic[T]):
    status: StatusEnum
    message: str
    data: Optional[T] = None
    errors: Optional[List[ErrorDetail]] = None
    metadata: MetadataModel

    @validator("data", always=True)
    def validate_data_or_errors(cls, v, values):
        status = values.get("status")
        errors = values.get("errors")
        if status == StatusEnum.ERROR and not errors:
            raise ValueError("Errors must be provided when status is 'error'")
        if status != StatusEnum.ERROR and errors:
            raise ValueError("Errors should only be provided when status is 'error'")
        return v

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    uptime: float
    timestamp: datetime
    services: Dict[str, str]