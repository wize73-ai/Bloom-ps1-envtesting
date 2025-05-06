"""
Document Processing Schemas for CasaLingua

This module defines schemas for document processing operations,
including document upload, extraction, and processing.
"""

from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from .base import BaseResponse


class DocumentType(str, Enum):
    """Supported document types for processing"""
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    TXT = "txt"
    UNKNOWN = "unknown"


class DocumentProcessingOptions(BaseModel):
    """Options for document processing"""
    translate: bool = False
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    simplify: bool = False
    simplification_level: Optional[str] = "medium"
    anonymize: bool = False
    analyze: bool = False
    extract_tables: bool = False
    preserve_formatting: bool = True
    ocr_enabled: bool = True  # Enable OCR for images and scans
    model_id: Optional[str] = None


class DocumentExtractionRequest(BaseModel):
    """Request for extracting text from a document"""
    file_type: DocumentType
    language: Optional[str] = None
    ocr_enabled: bool = True


class DocumentProcessingRequest(BaseModel):
    """Request for processing a document"""
    file_type: DocumentType
    options: DocumentProcessingOptions = Field(default_factory=DocumentProcessingOptions)


class DocumentResult(BaseModel):
    """Result of document processing"""
    original_text: str
    processed_text: Optional[str] = None
    document_type: DocumentType
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    languages: Optional[List[str]] = None
    detection_confidence: Optional[float] = None
    processing_metrics: Optional[Dict[str, Any]] = None
    translations: Optional[Dict[str, str]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    entities: Optional[List[Dict[str, Any]]] = None


class DocumentResponse(BaseResponse[DocumentResult]):
    """Response for document processing operations"""
    pass


class TableData(BaseModel):
    """Table data extracted from documents"""
    headers: List[str]
    rows: List[List[str]]
    position: Optional[Dict[str, int]] = None
    page: Optional[int] = None


class DocumentAnalysisResult(BaseModel):
    """Result of document analysis"""
    document_type: DocumentType
    page_count: int
    word_count: int
    character_count: int
    languages: List[Dict[str, float]]
    tables: Optional[List[TableData]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    sentiment: Optional[Dict[str, float]] = None
    topics: Optional[List[Dict[str, float]]] = None
    processing_time: float = 0.0


class DocumentAnalysisResponse(BaseResponse[DocumentAnalysisResult]):
    """Response for document analysis"""
    pass