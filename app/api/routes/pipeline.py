"""
Pipeline API endpoints for CasaLingua.
These endpoints provide access to the main document and speech processing pipelines.
"""

import os
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
import asyncio

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, File, Form, UploadFile, status
from pydantic import BaseModel

from app.api.middleware.auth import get_current_user
from app.api.schemas.speech import STTResponse, STTRequest, STTResult, SupportedLanguagesResponse
from app.api.schemas.base import BaseResponse, StatusEnum, MetadataModel
from app.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    responses={404: {"description": "Not found"}},
)

# ----- Speech-to-Text Endpoint -----

@router.post(
    "/stt",
    response_model=STTResponse,
    status_code=status.HTTP_200_OK,
    summary="Convert speech to text",
    description="Converts audio speech to text transcription."
)
async def speech_to_text(
    request: Request,
    background_tasks: BackgroundTasks,
    language: Optional[str] = Form(None, description="Language code"),
    detect_language: bool = Form(False, description="Detect language automatically"),
    model_id: Optional[str] = Form(None, description="Model ID to use"),
    enhanced_results: bool = Form(False, description="Include enhanced results"),
    audio_file: UploadFile = File(..., description="Audio file"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Convert speech to text transcription.
    
    This endpoint processes audio files and returns transcribed text.
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
        
        # Read audio file content
        audio_content = await audio_file.read()
        audio_format = audio_file.filename.split('.')[-1].lower()
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/stt",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "audio_size": len(audio_content),
                "language": language,
                "detect_language": detect_language,
                "audio_format": audio_format,
                "model_id": model_id
            }
        )
        
        # Process transcription request
        options = {
            "enhanced_results": enhanced_results
        }
        
        # Check if processor has STT method
        if hasattr(processor, "transcribe_speech"):
            transcription_result = await processor.transcribe_speech(
                audio_content=audio_content,
                language=language,
                detect_language=detect_language,
                model_id=model_id,
                options=options,
                user_id=current_user["id"],
                request_id=request_id
            )
        else:
            # Fallback to process method with audio pipeline
            transcription_result = await processor.process(
                content=audio_content,
                options={
                    "source_language": language,
                    "detect_language": detect_language,
                    "model_id": model_id,
                    "enhanced_results": enhanced_results,
                    "audio_format": audio_format,
                    "operation": "speech_to_text",
                    "request_id": request_id,
                    "user_id": current_user["id"]
                }
            )
            
            # Extract result
            if "audio_transcribed" in transcription_result:
                text = transcription_result.get("original_audio_text", "")
                detected_language = transcription_result.get("detected_language", language)
            else:
                # Use speech_to_text model directly through model manager
                from app.services.models.wrapper import ModelWrapper
                model_wrapper = ModelWrapper(processor.model_manager, "speech_to_text")
                
                # Prepare input for speech-to-text model
                input_data = {
                    "audio_content": audio_content,
                    "source_language": language,
                    "parameters": {
                        "model_name": model_id,
                        "detect_language": detect_language
                    }
                }
                
                # Run speech-to-text model
                model_result = await model_wrapper.process(input_data)
                
                # Extract result
                if isinstance(model_result, dict) and "result" in model_result:
                    if isinstance(model_result["result"], str):
                        text = model_result["result"]
                    else:
                        text = str(model_result["result"])
                    detected_language = model_result.get("language", language)
                else:
                    text = str(model_result)
                    detected_language = language
                
                # Create transcription result
                transcription_result = {
                    "text": text,
                    "language": detected_language or language or "en",
                    "confidence": 0.7,
                    "model_used": "speech_to_text",
                    "processing_time": time.time() - start_time,
                    "audio_format": audio_format
                }
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="stt",
            operation="transcribe",
            duration=process_time,
            input_size=len(audio_content),
            output_size=len(transcription_result.get("text", "")),
            success=True,
            metadata={
                "language": transcription_result.get("language", language),
                "model_id": transcription_result.get("model_used", "stt"),
                "audio_format": audio_format,
                "audio_duration": transcription_result.get("duration", 0.0)
            }
        )
        
        # Create result model
        result = STTResult(
            text=transcription_result.get("text", ""),
            language=transcription_result.get("language", language or "en"),
            confidence=transcription_result.get("confidence", 0.7),
            segments=transcription_result.get("segments"),
            duration=transcription_result.get("duration"),
            model_used=transcription_result.get("model_used", "speech_to_text"),
            processing_time=process_time,
            audio_format=audio_format,
            fallback=transcription_result.get("fallback", False),
            performance_metrics=transcription_result.get("performance_metrics"),
            memory_usage=transcription_result.get("memory_usage"),
            operation_cost=transcription_result.get("operation_cost", 0.0)
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Speech transcription completed successfully",
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
        
    except Exception as e:
        logger.error(f"Speech transcription error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="stt",
            operation="transcribe",
            duration=time.time() - start_time,
            input_size=len(audio_content) if 'audio_content' in locals() else 0,
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": language
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech transcription error: {str(e)}"
        )

@router.get(
    "/stt/languages",
    response_model=SupportedLanguagesResponse,
    status_code=status.HTTP_200_OK,
    summary="Get supported STT languages",
    description="Returns a list of supported languages for speech recognition."
)
async def get_stt_languages(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get supported languages for speech recognition.
    
    This endpoint returns a list of available languages for STT.
    """
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        
        # Get available languages
        if hasattr(processor, "get_stt_languages"):
            languages_result = await processor.get_stt_languages()
        elif hasattr(processor, "stt_pipeline") and processor.stt_pipeline:
            languages_result = await processor.stt_pipeline.get_supported_languages()
        else:
            # Fallback to basic language list
            languages_result = {
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
                "default_language": "en"
            }
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Supported STT languages retrieved successfully",
            data=languages_result,
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0")
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting STT languages: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting STT languages: {str(e)}"
        )