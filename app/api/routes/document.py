"""
Document Processing Routes for CasaLingua

This module defines API endpoints for document processing, including
text extraction, translation, and analysis. It uses a session manager
to store document content temporarily during user sessions.
"""

import time
import uuid
import logging
import mimetypes
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Query, Path, Request, status, Cookie, Response
from pydantic import BaseModel, Field, validator

from app.api.schemas.document import (
    DocumentType, 
    DocumentProcessingOptions, 
    DocumentExtractionRequest, 
    DocumentProcessingRequest,
    DocumentResult,
    DocumentResponse,
    DocumentAnalysisResult,
    DocumentAnalysisResponse
)
from app.api.schemas.base import BaseResponse, StatusEnum, MetadataModel, ErrorDetail
from app.api.middleware.auth import get_current_user
from app.utils.logging import get_logger
from app.services.storage.session_manager import SessionManager

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Get the session manager instance
session_manager = SessionManager()

# ----- Document Processing Endpoints -----

@router.post(
    "/extract",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract text from document",
    description="Extracts text content from a document file."
)
async def extract_document_text(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None, description="Language code for OCR processing"),
    ocr_enabled: bool = Form(True, description="Whether to use OCR for scanned documents"),
    session_id: Optional[str] = Cookie(None, description="Session identifier for document storage"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Extract text from a document without further processing.
    
    This endpoint handles PDF, DOCX, and image files, extracting
    the text content with optional OCR for images and scanned PDFs.
    Document content is stored in the user's session for later use.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Determine document type from file content type or extension
        content_type = file.content_type
        filename = file.filename
        
        # Map content type to document type
        if "pdf" in content_type.lower():
            document_type = "application/pdf"
        elif "word" in content_type.lower() or filename.lower().endswith((".docx", ".doc")):
            document_type = "application/docx"
        elif any(img_type in content_type.lower() for img_type in ["image", "png", "jpeg", "jpg"]):
            document_type = "image"
        else:
            document_type = content_type
            
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/document/extract",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "filename": filename,
                "document_type": document_type,
                "language": language,
                "ocr_enabled": ocr_enabled,
                "session_id": session_id
            }
        )
        
        # Read file content
        content = await file.read()
        
        # Store document in session
        document_id = str(uuid.uuid4())
        await session_manager.add_document(
            session_id=session_id,
            document_id=document_id,
            content=content,
            metadata={
                "filename": filename,
                "content_type": content_type,
                "document_type": document_type,
                "size": len(content),
                "user_id": current_user["id"],
                "created_at": time.time()
            }
        )
        
        # Extract text from document
        extraction_result = await processor.extract_document_text(
            document_content=content,
            document_type=document_type,
            options={
                "language": language,
                "ocr_enabled": ocr_enabled
            },
            filename=filename,
            request_id=request_id
        )
        
        # Check for errors
        if "error" in extraction_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=extraction_result["error"]
            )
        
        # Create result
        result = DocumentResult(
            original_text=extraction_result.get("text", ""),
            document_type=DocumentType(document_type.split("/")[-1] if "/" in document_type else document_type),
            page_count=extraction_result.get("metadata", {}).get("page_count", 1),
            word_count=extraction_result.get("word_count", 0),
            detection_confidence=extraction_result.get("metadata", {}).get("ocr_confidence")
        )
        
        # Store extraction result in session metadata
        await session_manager.update_session_metadata(
            session_id=session_id,
            metadata={
                f"document_{document_id}_extraction": {
                    "text": extraction_result.get("text", ""),
                    "page_count": extraction_result.get("metadata", {}).get("page_count", 1),
                    "word_count": extraction_result.get("word_count", 0),
                    "processing_time": extraction_result.get("processing_time", 0)
                }
            }
        )
        
        # Calculate process time
        process_time = extraction_result.get("processing_time", time.time() - start_time)
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_extraction",
            operation="extract",
            duration=process_time,
            input_size=len(content),
            output_size=len(result.original_text),
            success=True,
            metadata={
                "document_type": document_type,
                "filename": filename,
                "ocr_enabled": ocr_enabled,
                "document_id": document_id,
                "session_id": session_id
            }
        )
        
        # Create response with session info
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document text extraction completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time,
                document_id=document_id,
                session_id=session_id
            ),
            errors=None,
            pagination=None
        )
        
        # Set session cookie in response
        response_obj = Response(response.dict())
        response_obj.set_cookie(
            key="session_id",
            value=session_id,
            max_age=3600,  # 1 hour
            httponly=True,
            samesite="lax"
        )
        
        return response_obj
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Document extraction error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_extraction",
            operation="extract",
            duration=time.time() - start_time,
            input_size=len(await file.read()) if not file.closed else 0,
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "filename": file.filename,
                "session_id": session_id
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document extraction error: {str(e)}"
        )
    finally:
        # Make sure to close the file
        await file.close()


@router.post(
    "/process",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Process document",
    description="Process a document with translation, simplification, anonymization, etc."
)
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: Optional[str] = Form(None, description="Source language"),
    target_language: Optional[str] = Form(None, description="Target language"),
    translate: bool = Form(False, description="Whether to translate document"),
    simplify: bool = Form(False, description="Whether to simplify document text"),
    simplification_level: str = Form("medium", description="Simplification level (easy, medium, hard)"),
    anonymize: bool = Form(False, description="Whether to anonymize personal information"),
    ocr_enabled: bool = Form(True, description="Whether to use OCR for scanned documents"),
    preserve_formatting: bool = Form(True, description="Whether to preserve document formatting"),
    model_id: Optional[str] = Form(None, description="Model ID to use for processing"),
    generate_document: bool = Form(True, description="Whether to generate processed document"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Process a document with various operations.
    
    This endpoint handles document processing with options for translation,
    simplification, and anonymization, supporting PDF, DOCX, and image files.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Determine document type from file content type or extension
        content_type = file.content_type
        filename = file.filename
        
        # Map content type to document type
        if "pdf" in content_type.lower():
            document_type = "application/pdf"
        elif "word" in content_type.lower() or filename.lower().endswith((".docx", ".doc")):
            document_type = "application/docx"
        elif any(img_type in content_type.lower() for img_type in ["image", "png", "jpeg", "jpg"]):
            document_type = "image"
        else:
            document_type = content_type
            
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/document/process",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "filename": filename,
                "document_type": document_type,
                "translate": translate,
                "source_language": source_language,
                "target_language": target_language,
                "simplify": simplify,
                "simplification_level": simplification_level,
                "anonymize": anonymize,
                "ocr_enabled": ocr_enabled,
                "model_id": model_id
            }
        )
        
        # Read file content
        content = await file.read()
        
        # Prepare processing options
        options = {
            "source_language": source_language,
            "target_language": target_language if translate else None,
            "translate": translate,
            "simplify": simplify,
            "simplification_level": simplification_level,
            "anonymize": anonymize,
            "ocr_enabled": ocr_enabled,
            "preserve_formatting": preserve_formatting,
            "model_id": model_id,
            "generate_document": generate_document
        }
        
        # Process document
        processing_result = await processor.process_document(
            document_content=content,
            document_type=document_type,
            options=options,
            filename=filename,
            user_id=current_user["id"],
            request_id=request_id
        )
        
        # Check for errors
        if "error" in processing_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=processing_result["error"]
            )
        
        # Extract original and processed text
        original_text = processing_result.get("original_text", "")
        processed_text = processing_result.get("processed_text", "")
        
        # If translation was applied, extract languages
        languages = []
        if processing_result.get("translation_applied", False):
            languages = [
                processing_result.get("source_language", source_language or "auto"),
                processing_result.get("target_language", target_language)
            ]
        
        # Create result
        result = DocumentResult(
            original_text=original_text,
            processed_text=processed_text,
            document_type=DocumentType(document_type.split("/")[-1] if "/" in document_type else document_type),
            page_count=processing_result.get("document_metadata", {}).get("page_count", 1),
            word_count=len(processed_text.split()) if processed_text else 0,
            languages=languages if languages else None,
            processing_metrics={
                "translation_applied": processing_result.get("translation_applied", False),
                "simplification_applied": processing_result.get("simplification_applied", False),
                "anonymization_applied": processing_result.get("anonymization_applied", False),
                "processing_time": processing_result.get("processing_time", 0.0)
            }
        )
        
        # Calculate process time
        process_time = processing_result.get("processing_time", time.time() - start_time)
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_processing",
            operation="process",
            duration=process_time,
            input_size=len(content),
            output_size=len(processed_text),
            success=True,
            metadata={
                "document_type": document_type,
                "filename": filename,
                "translate": translate,
                "simplify": simplify,
                "anonymize": anonymize
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document processing completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_processing",
            operation="process",
            duration=time.time() - start_time,
            input_size=len(await file.read()) if not file.closed else 0,
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "filename": file.filename
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing error: {str(e)}"
        )
    finally:
        # Make sure to close the file
        await file.close()


@router.post(
    "/analyze",
    response_model=DocumentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze document",
    description="Analyze a document to extract metadata, entities, and insights."
)
async def analyze_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None, description="Language code"),
    detect_language: bool = Form(True, description="Whether to detect language"),
    ocr_enabled: bool = Form(True, description="Whether to use OCR for scanned documents"),
    extract_entities: bool = Form(False, description="Whether to extract entities"),
    extract_tables: bool = Form(False, description="Whether to extract tables"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Analyze a document to extract metadata and insights.
    
    This endpoint performs document analysis to extract metadata,
    detect languages, identify entities, and extract tables.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Determine document type from file content type or extension
        content_type = file.content_type
        filename = file.filename
        
        # Map content type to document type
        if "pdf" in content_type.lower():
            document_type = "application/pdf"
        elif "word" in content_type.lower() or filename.lower().endswith((".docx", ".doc")):
            document_type = "application/docx"
        elif any(img_type in content_type.lower() for img_type in ["image", "png", "jpeg", "jpg"]):
            document_type = "image"
        else:
            document_type = content_type
            
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/document/analyze",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "filename": filename,
                "document_type": document_type,
                "language": language,
                "detect_language": detect_language,
                "ocr_enabled": ocr_enabled,
                "extract_entities": extract_entities,
                "extract_tables": extract_tables
            }
        )
        
        # Read file content
        content = await file.read()
        
        # Prepare analysis options
        options = {
            "language": language,
            "detect_language": detect_language,
            "ocr_enabled": ocr_enabled,
            "extract_entities": extract_entities,
            "extract_tables": extract_tables
        }
        
        # Analyze document
        analysis_result = await processor.analyze_document(
            document_content=content,
            document_type=document_type,
            options=options,
            filename=filename,
            request_id=request_id
        )
        
        # Check for errors
        if "error" in analysis_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=analysis_result["error"]
            )
        
        # Create result model
        result = DocumentAnalysisResult(
            document_type=DocumentType(document_type.split("/")[-1] if "/" in document_type else document_type),
            page_count=analysis_result.get("page_count", 1),
            word_count=analysis_result.get("word_count", 0),
            character_count=analysis_result.get("character_count", 0),
            languages=[{
                "language": lang.get("language", "unknown"),
                "confidence": lang.get("confidence", 0.0)
            } for lang in analysis_result.get("languages", [])],
            tables=analysis_result.get("tables"),
            entities=analysis_result.get("entities"),
            sentiment=analysis_result.get("sentiment"),
            processing_time=analysis_result.get("processing_time", time.time() - start_time)
        )
        
        # Calculate process time
        process_time = analysis_result.get("processing_time", time.time() - start_time)
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_analysis",
            operation="analyze",
            duration=process_time,
            input_size=len(content),
            output_size=0,
            success=True,
            metadata={
                "document_type": document_type,
                "filename": filename,
                "word_count": result.word_count,
                "page_count": result.page_count
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document analysis completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="document_analysis",
            operation="analyze",
            duration=time.time() - start_time,
            input_size=len(await file.read()) if not file.closed else 0,
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "filename": file.filename
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis error: {str(e)}"
        )
    finally:
        # Make sure to close the file
        await file.close()