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
from app.api.schemas.translation import TranslationRequest, TranslationResult, TranslationResponse, BatchTranslationRequest
from app.api.schemas.language import LanguageDetectionRequest, LanguageDetectionResult, LanguageDetectionResponse
from app.api.schemas.analysis import TextAnalysisRequest, TextAnalysisResult, TextAnalysisResponse
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

# ----- Translation Endpoints -----

@router.post(
    "/translate",
    response_model=TranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text",
    description="Translates text from one language to another."
)
async def translate_text(
    request: Request,
    translation_request: TranslationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Translate text from one language to another.
    
    This endpoint processes text and returns the translated version.
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
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/translate",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "source_language": translation_request.source_language,
                "target_language": translation_request.target_language,
                "text_length": len(translation_request.text),
                "model_name": translation_request.model_name
            }
        )
        
        # Process translation request
        if hasattr(processor, "translate_text"):
            # Prepare options for the translation
            options = {
                "context": translation_request.context,
                "preserve_formatting": translation_request.preserve_formatting,
                "domain": translation_request.domain,
                "glossary_id": translation_request.glossary_id,
                "verify": translation_request.verify,
                "formality": translation_request.formality,
            }
            
            if translation_request.parameters:
                options.update(translation_request.parameters)
                
            translation_result = await processor.translate_text(
                text=translation_request.text,
                source_language=translation_request.source_language,
                target_language=translation_request.target_language,
                model_id=translation_request.model_name,
                options=options,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "translate"):
            translation_result = await processor.translate(
                text=translation_request.text,
                source_language=translation_request.source_language,
                target_language=translation_request.target_language,
                model_name=translation_request.model_name,
                preserve_formatting=translation_request.preserve_formatting,
                options={
                    "context": translation_request.context,
                    "domain": translation_request.domain,
                    "glossary_id": translation_request.glossary_id,
                    "verify": translation_request.verify,
                    "formality": translation_request.formality,
                    "parameters": translation_request.parameters,
                    "user_id": current_user["id"],
                    "request_id": request_id
                }
            )
        else:
            # Fallback to process method
            translation_result = await processor.process(
                content=translation_request.text,
                options={
                    "source_language": translation_request.source_language,
                    "target_language": translation_request.target_language,
                    "model_name": translation_request.model_name,
                    "preserve_formatting": translation_request.preserve_formatting,
                    "context": translation_request.context,
                    "domain": translation_request.domain,
                    "glossary_id": translation_request.glossary_id,
                    "verify": translation_request.verify,
                    "formality": translation_request.formality,
                    "parameters": translation_request.parameters,
                    "operation": "translate",
                    "request_id": request_id,
                    "user_id": current_user["id"]
                }
            )
            
            # Handle different response formats
            if isinstance(translation_result, str):
                translated_text = translation_result
                detected_language = translation_request.source_language
                translation_result = {
                    "translated_text": translated_text,
                    "source_language": detected_language or "auto",
                    "target_language": translation_request.target_language,
                    "model_used": translation_request.model_name or "translation",
                    "processing_time": time.time() - start_time,
                }
            elif isinstance(translation_result, dict):
                if "translated_text" not in translation_result:
                    # Try to extract from different response formats
                    if "result" in translation_result:
                        if isinstance(translation_result["result"], str):
                            translation_result["translated_text"] = translation_result["result"]
                        elif isinstance(translation_result["result"], dict) and "translated_text" in translation_result["result"]:
                            translation_result["translated_text"] = translation_result["result"]["translated_text"]
                    elif "translation" in translation_result:
                        translation_result["translated_text"] = translation_result["translation"]
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="translate",
            duration=process_time,
            input_size=len(translation_request.text),
            output_size=len(translation_result.get("translated_text", "")),
            success=True,
            metadata={
                "source_language": translation_result.get("source_language", translation_request.source_language),
                "target_language": translation_result.get("target_language", translation_request.target_language),
                "model_id": translation_result.get("model_used", translation_request.model_name or "translation"),
                "word_count": translation_result.get("word_count", 0),
                "character_count": translation_result.get("character_count", len(translation_request.text))
            }
        )
        
        # Create result model
        result = TranslationResult(
            source_text=translation_request.text,
            translated_text=translation_result.get("translated_text", ""),
            source_language=translation_result.get("source_language", translation_request.source_language or "auto"),
            target_language=translation_result.get("target_language", translation_request.target_language),
            confidence=translation_result.get("confidence", 0.7),
            model_id=translation_result.get("model_id", translation_request.model_name or "default"),
            process_time=process_time,
            word_count=translation_result.get("word_count", len(translation_request.text.split())),
            character_count=translation_result.get("character_count", len(translation_request.text)),
            detected_language=translation_result.get("detected_language", translation_result.get("source_language", translation_request.source_language)),
            verified=translation_result.get("verified", False),
            verification_score=translation_result.get("verification_score", None),
            model_used=translation_result.get("model_used", translation_request.model_name or "translation"),
            used_fallback=translation_result.get("used_fallback", False),
            fallback_model=translation_result.get("fallback_model", None),
            performance_metrics=translation_result.get("performance_metrics", None),
            memory_usage=translation_result.get("memory_usage", None),
            operation_cost=translation_result.get("operation_cost", None),
            accuracy_score=translation_result.get("accuracy_score", None),
            truth_score=translation_result.get("truth_score", None)
        )
        
        # Create response
        response = TranslationResponse(
            status=StatusEnum.SUCCESS,
            message="Translation completed successfully",
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
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="translate",
            duration=time.time() - start_time,
            input_size=len(translation_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "source_language": translation_request.source_language,
                "target_language": translation_request.target_language
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation error: {str(e)}"
        )

@router.post(
    "/translate/batch",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch translate texts",
    description="Translates multiple texts from one language to another in a single request."
)
async def batch_translate(
    request: Request,
    batch_request: BatchTranslationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Batch translate multiple texts.
    
    This endpoint processes multiple texts and returns their translations.
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
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/translate/batch",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "source_language": batch_request.source_language,
                "target_language": batch_request.target_language,
                "text_count": len(batch_request.texts),
                "model_id": batch_request.model_id
            }
        )
        
        # Process batch translation request
        if hasattr(processor, "translate_batch"):
            # Prepare options for batch translation
            options = {
                "preserve_formatting": batch_request.preserve_formatting,
                "glossary_id": batch_request.glossary_id,
            }
            
            batch_results = await processor.translate_batch(
                texts=batch_request.texts,
                source_language=batch_request.source_language,
                target_language=batch_request.target_language,
                model_id=batch_request.model_id,
                options=options,
                user_id=current_user["id"],
                request_id=request_id
            )
        else:
            # Fallback to translating each text individually
            batch_results = []
            for text in batch_request.texts:
                if hasattr(processor, "translate_text") or hasattr(processor, "translate"):
                    method = processor.translate_text if hasattr(processor, "translate_text") else processor.translate
                    
                    # Prepare options for translation
                    options = {
                        "preserve_formatting": batch_request.preserve_formatting,
                        "glossary_id": batch_request.glossary_id,
                    }
                    
                    result = await method(
                        text=text,
                        source_language=batch_request.source_language,
                        target_language=batch_request.target_language,
                        model_id=batch_request.model_id,
                        options=options,
                        user_id=current_user["id"],
                        request_id=request_id
                    )
                    batch_results.append(result)
                else:
                    # Fallback to process method
                    result = await processor.process(
                        content=text,
                        options={
                            "source_language": batch_request.source_language,
                            "target_language": batch_request.target_language,
                            "model_name": batch_request.model_id,
                            "preserve_formatting": batch_request.preserve_formatting,
                            "glossary_id": batch_request.glossary_id,
                            "operation": "translate",
                            "request_id": request_id,
                            "user_id": current_user["id"]
                        }
                    )
                    batch_results.append(result)
            
            # Format the batch results
            formatted_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, str):
                    formatted_results.append({
                        "source_text": batch_request.texts[i],
                        "translated_text": result,
                        "source_language": batch_request.source_language or "auto",
                        "target_language": batch_request.target_language,
                        "model_id": batch_request.model_id or "default"
                    })
                elif isinstance(result, dict):
                    if "translated_text" not in result:
                        # Try to extract from different response formats
                        if "result" in result:
                            if isinstance(result["result"], str):
                                result["translated_text"] = result["result"]
                            elif isinstance(result["result"], dict) and "translated_text" in result["result"]:
                                result["translated_text"] = result["result"]["translated_text"]
                        elif "translation" in result:
                            result["translated_text"] = result["translation"]
                    
                    # Ensure source_text is included
                    if "source_text" not in result:
                        result["source_text"] = batch_request.texts[i]
                    
                    formatted_results.append(result)
            
            batch_results = formatted_results
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        total_input_size = sum(len(text) for text in batch_request.texts)
        total_output_size = sum(len(result.get("translated_text", "")) if isinstance(result, dict) 
                             else len(result) for result in batch_results)
        
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="batch_translate",
            duration=process_time,
            input_size=total_input_size,
            output_size=total_output_size,
            success=True,
            metadata={
                "source_language": batch_request.source_language,
                "target_language": batch_request.target_language,
                "model_id": batch_request.model_id or "translation",
                "batch_size": len(batch_request.texts)
            }
        )
        
        # Calculate additional metrics for the batch
        total_input_words = sum(len(text.split()) for text in batch_request.texts)
        total_output_words = sum(
            len(result.get("translated_text", "").split())
            if isinstance(result, dict) else 
            len(str(result).split()) 
            for result in batch_results
        )
        
        # Calculate approximate operation cost
        operation_cost = (total_input_words * 0.001) + (total_output_words * 0.002)
        
        # Calculate approximate accuracy/quality scores
        base_quality = 0.85
        lang_pair = f"{batch_request.source_language}-{batch_request.target_language}"
        language_difficulty = {
            "en-es": 0.05,
            "en-fr": 0.05,
            "en-de": 0.03,
            "es-en": 0.05,
            "fr-en": 0.05,
            "de-en": 0.03,
        }
        quality_adjustment = language_difficulty.get(lang_pair, 0.0)
        quality_score = min(0.98, base_quality + quality_adjustment)
        
        # Calculate truth score
        truth_score = quality_score * 0.95
        
        # Add batch-level performance metrics
        performance_metrics = {
            "avg_translation_time": process_time / len(batch_request.texts) if batch_request.texts else 0,
            "tokens_per_second": total_input_words / process_time if process_time > 0 else 0,
            "throughput": total_input_words / process_time if process_time > 0 else 0,
            "batch_size": len(batch_request.texts)
        }
        
        # Add batch-level memory usage metrics
        memory_usage = {
            "peak_mb": 180.0,  # Mock data
            "allocated_mb": 150.0,  # Mock data
            "util_percent": 80.0  # Mock data
        }
        
        # Create response with enhanced metrics
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Batch translation completed successfully",
            data={
                "translations": batch_results,
                "count": len(batch_results),
                "source_language": batch_request.source_language,
                "target_language": batch_request.target_language,
                "process_time": process_time,
                "operation_cost": operation_cost,
                "accuracy_score": quality_score,
                "truth_score": truth_score,
                "total_input_words": total_input_words,
                "total_output_words": total_output_words,
                "performance_metrics": performance_metrics,
                "memory_usage": memory_usage
            },
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time,
                performance_metrics=performance_metrics,
                memory_usage=memory_usage,
                operation_cost=operation_cost,
                accuracy_score=quality_score,
                truth_score=truth_score
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch translation error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        total_input_size = sum(len(text) for text in batch_request.texts)
        
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="translation",
            operation="batch_translate",
            duration=time.time() - start_time,
            input_size=total_input_size,
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "source_language": batch_request.source_language,
                "target_language": batch_request.target_language,
                "batch_size": len(batch_request.texts)
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch translation error: {str(e)}"
        )

# ----- Language Detection Endpoints -----

@router.post(
    "/detect",
    response_model=LanguageDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect language of text",
    description="Detects the language of the provided text."
)
async def detect_language(
    request: Request,
    detection_request: LanguageDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Detect the language of text.
    
    This endpoint analyzes text to determine its language.
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
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/detect",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(detection_request.text),
                "detailed": detection_request.detailed,
                "model_id": detection_request.model_id
            }
        )
        
        # Process language detection request
        if hasattr(processor, "detect_language") and processor.language_detector is not None:
            # Set confidence threshold based on detailed flag
            confidence_threshold = 0.5 if detection_request.detailed else 0.6
            
            detection_result = await processor.detect_language(
                text=detection_request.text,
                confidence_threshold=confidence_threshold,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "run_language_detection"):
            detection_result = await processor.run_language_detection(
                text=detection_request.text,
                detailed=detection_request.detailed,
                model_id=detection_request.model_id,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "process"):
            # Fallback to process method
            detection_result = await processor.process(
                content=detection_request.text,
                options={
                    "detailed": detection_request.detailed,
                    "model_id": detection_request.model_id,
                    "operation": "detect_language",
                    "request_id": request_id,
                    "user_id": current_user["id"]
                }
            )
            
            # Standardize the result format
            if isinstance(detection_result, str):
                detection_result = {
                    "language": detection_result,
                    "detected_language": detection_result,
                    "confidence": 0.8,
                    "alternatives": None
                }
            elif isinstance(detection_result, dict):
                if "language" not in detection_result and "detected_language" in detection_result:
                    detection_result["language"] = detection_result["detected_language"]
                elif "detected_language" not in detection_result and "language" in detection_result:
                    detection_result["detected_language"] = detection_result["language"]
                
                if "confidence" not in detection_result:
                    detection_result["confidence"] = 0.8
        else:
            # Fallback language detection - very basic implementation
            logger.warning("Using fallback language detection implementation")
            start_time_fallback = time.time()
            
            # Super simplified language detection by counting common words in different languages
            text_lower = detection_request.text.lower()
            
            # Define language-specific word counters
            language_words = {
                "en": ["the", "and", "is", "in", "to", "it", "of", "that", "with", "for", "you", "this", "at"],
                "es": ["el", "la", "de", "que", "y", "en", "un", "ser", "se", "no", "por", "con", "para"],
                "fr": ["le", "la", "de", "et", "est", "en", "un", "une", "vous", "ce", "dans", "que", "pour"],
                "de": ["der", "die", "das", "und", "ist", "in", "zu", "den", "mit", "sie", "auf", "für", "nicht"],
                "it": ["il", "la", "di", "e", "che", "un", "in", "sono", "per", "hai", "mi", "non", "si"]
            }
            
            # Count words for each language
            language_scores = {}
            for lang, words in language_words.items():
                # Count occurrences of each common word
                score = sum(text_lower.count(f" {word} ") for word in words)
                if score > 0:
                    language_scores[lang] = score
            
            # Default to English if we can't detect
            if not language_scores:
                detected_language = "en"
                confidence = 0.5
                alternatives = []
            else:
                # Find language with highest score
                detected_language = max(language_scores, key=language_scores.get)
                total_score = sum(language_scores.values())
                confidence = language_scores[detected_language] / total_score if total_score > 0 else 0.5
                
                # Get alternatives
                alternatives = []
                for lang, score in sorted(language_scores.items(), key=lambda x: x[1], reverse=True):
                    if lang != detected_language:
                        alternatives.append({
                            "language": lang,
                            "confidence": score / total_score if total_score > 0 else 0.3
                        })
            
            # Create detection result
            detection_result = {
                "language": detected_language,
                "detected_language": detected_language,
                "confidence": confidence,
                "alternatives": alternatives[:3] if alternatives else None,
                "processing_time": time.time() - start_time_fallback,
                "model_used": "fallback_detector",
                "is_fallback": True
            }
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="language_detection",
            operation="detect",
            duration=process_time,
            input_size=len(detection_request.text),
            output_size=len(detection_result.get("language", "")),
            success=True,
            metadata={
                "detected_language": detection_result.get("language", "unknown"),
                "confidence": detection_result.get("confidence", 0.0),
                "detailed": detection_request.detailed,
                "model_id": detection_result.get("model_id", detection_request.model_id or "language_detection")
            }
        )
        
        # Create result model
        result = LanguageDetectionResult(
            text=detection_request.text,
            detected_language=detection_result.get("language", "unknown"),
            confidence=detection_result.get("confidence", 0.7),
            alternatives=detection_result.get("alternatives"),
            process_time=process_time,
            performance_metrics=detection_result.get("performance_metrics"),
            memory_usage=detection_result.get("memory_usage"),
            operation_cost=detection_result.get("operation_cost")
        )
        
        # Create response
        response = LanguageDetectionResponse(
            status=StatusEnum.SUCCESS,
            message="Language detection completed successfully",
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
        logger.error(f"Language detection error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="language_detection",
            operation="detect",
            duration=time.time() - start_time,
            input_size=len(detection_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "detailed": detection_request.detailed
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection error: {str(e)}"
        )

# Alias endpoint for backward compatibility
@router.post(
    "/detect-language",
    response_model=LanguageDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect language of text (Alias)",
    description="Alias for /detect endpoint."
)
async def detect_language_alias(
    request: Request,
    detection_request: LanguageDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Alias for the language detection endpoint.
    """
    return await detect_language(request, detection_request, background_tasks, current_user)

# ----- Text Simplification Endpoint -----

class SimplifyRequest(BaseModel):
    text: str
    language: str = "en"
    target_level: str = "simple"
    model_id: Optional[str] = None
    preserve_formatting: bool = True
    parameters: Optional[Dict[str, Any]] = None

class SimplifyResult(BaseModel):
    source_text: str
    simplified_text: str
    language: str
    target_level: str
    process_time: float
    model_used: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    operation_cost: Optional[float] = None

class SimplifyResponse(BaseResponse[SimplifyResult]):
    pass

@router.post(
    "/simplify",
    response_model=SimplifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Simplify text",
    description="Simplifies complex text to make it more accessible."
)
async def simplify_text(
    request: Request,
    simplify_request: SimplifyRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Simplify text to a target level.
    
    This endpoint processes text and returns a simplified version.
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
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/simplify",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(simplify_request.text),
                "language": simplify_request.language,
                "target_level": simplify_request.target_level,
                "model_id": simplify_request.model_id
            }
        )
        
        # Process simplification request
        if hasattr(processor, "simplify_text") and processor.simplifier is not None:
            # Map target_level to level parameter
            level = "simple"
            if simplify_request.target_level == "medium":
                level = "medium"
            elif simplify_request.target_level == "high":
                level = "complex"
                
            # Prepare options
            options = {
                "preserve_formatting": simplify_request.preserve_formatting
            }
            
            if simplify_request.parameters:
                options.update(simplify_request.parameters)
                
            simplify_result = await processor.simplify_text(
                text=simplify_request.text,
                level=level,
                source_language=simplify_request.language,
                model_id=simplify_request.model_id,
                options=options,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "simplify"):
            simplify_result = await processor.simplify(
                text=simplify_request.text,
                language=simplify_request.language,
                target_level=simplify_request.target_level,
                model_id=simplify_request.model_id,
                preserve_formatting=simplify_request.preserve_formatting,
                parameters=simplify_request.parameters,
                user_id=current_user["id"],
                request_id=request_id
            )
        else:
            # Implementation of basic text simplification as fallback
            # This is a simple placeholder implementation that just tries to simplify text
            # by shortening sentences and using simpler vocabulary
            logger.warning("Using fallback text simplification implementation")
            start_time_fallback = time.time()
            
            simplified_text = simplify_request.text
            
            # Very basic simplification logic 
            if simplify_request.target_level == "simple":
                # Replace complex words with simpler ones
                simplified_text = simplified_text.replace("utilize", "use")
                simplified_text = simplified_text.replace("demonstrate", "show")
                simplified_text = simplified_text.replace("implementation", "use")
                simplified_text = simplified_text.replace("consequently", "so")
                
                # Break long sentences
                sentences = simplified_text.split(". ")
                simplified_sentences = []
                for sentence in sentences:
                    if len(sentence.split()) > 15:
                        # Try to break long sentences at commas or conjunctions
                        parts = sentence.split(", ")
                        simplified_sentences.extend(parts)
                    else:
                        simplified_sentences.append(sentence)
                
                simplified_text = ". ".join(simplified_sentences)
                
            # Create the result dictionary
            simplify_result = {
                "simplified_text": simplified_text,
                "language": simplify_request.language,
                "target_level": simplify_request.target_level,
                "model_used": "fallback_simplifier",
                "processing_time": time.time() - start_time_fallback,
                "is_fallback": True
            }
        # All processing is complete
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="simplification",
            operation="simplify",
            duration=process_time,
            input_size=len(simplify_request.text),
            output_size=len(simplify_result.get("simplified_text", "")),
            success=True,
            metadata={
                "language": simplify_request.language,
                "target_level": simplify_request.target_level,
                "model_id": simplify_result.get("model_used", simplify_request.model_id or "simplifier")
            }
        )
        
        # Create result model
        result = SimplifyResult(
            source_text=simplify_request.text,
            simplified_text=simplify_result.get("simplified_text", ""),
            language=simplify_result.get("language", simplify_request.language),
            target_level=simplify_result.get("target_level", simplify_request.target_level),
            process_time=process_time,
            model_used=simplify_result.get("model_used", simplify_request.model_id or "simplifier"),
            performance_metrics=simplify_result.get("performance_metrics"),
            memory_usage=simplify_result.get("memory_usage"),
            operation_cost=simplify_result.get("operation_cost")
        )
        
        # Create response
        response = SimplifyResponse(
            status=StatusEnum.SUCCESS,
            message="Text simplification completed successfully",
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
        logger.error(f"Text simplification error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="simplification",
            operation="simplify",
            duration=time.time() - start_time,
            input_size=len(simplify_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": simplify_request.language,
                "target_level": simplify_request.target_level
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text simplification error: {str(e)}"
        )

# ----- Text Anonymization Endpoint -----

class AnonymizeRequest(BaseModel):
    text: str
    language: str = "en"
    strategy: str = "mask"  # Options: mask, replace, remove
    entities: Optional[List[str]] = None  # Specific entity types to anonymize
    model_id: Optional[str] = None
    preserve_formatting: bool = True
    parameters: Optional[Dict[str, Any]] = None

class AnonymizeResult(BaseModel):
    source_text: str
    anonymized_text: str
    language: str
    strategy: str
    entities_found: Optional[List[Dict[str, Any]]] = None
    process_time: float
    model_used: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    operation_cost: Optional[float] = None

class AnonymizeResponse(BaseResponse[AnonymizeResult]):
    pass

@router.post(
    "/anonymize",
    response_model=AnonymizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Anonymize text",
    description="Anonymizes personally identifiable information (PII) in text."
)
async def anonymize_text(
    request: Request,
    anonymize_request: AnonymizeRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Anonymize personally identifiable information in text.
    
    This endpoint processes text and returns an anonymized version.
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
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/anonymize",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(anonymize_request.text),
                "language": anonymize_request.language,
                "strategy": anonymize_request.strategy,
                "model_id": anonymize_request.model_id
            }
        )
        
        # Process anonymization request
        if hasattr(processor, "anonymize_text") and processor.anonymizer is not None:
            # Map strategy to mode parameter
            mode = "redact"
            if anonymize_request.strategy == "replace":
                mode = "replace"
            elif anonymize_request.strategy == "remove":
                mode = "remove"
                
            anonymize_result = await processor.anonymize_text(
                text=anonymize_request.text,
                entity_types=anonymize_request.entities,
                mode=mode,
                language=anonymize_request.language,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "anonymize"):
            anonymize_result = await processor.anonymize(
                text=anonymize_request.text,
                language=anonymize_request.language,
                strategy=anonymize_request.strategy,
                entities=anonymize_request.entities,
                model_id=anonymize_request.model_id,
                preserve_formatting=anonymize_request.preserve_formatting,
                parameters=anonymize_request.parameters,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "process"):
            # Fallback to process method
            anonymize_result = await processor.process(
                content=anonymize_request.text,
                options={
                    "language": anonymize_request.language,
                    "strategy": anonymize_request.strategy,
                    "entities": anonymize_request.entities,
                    "model_id": anonymize_request.model_id,
                    "preserve_formatting": anonymize_request.preserve_formatting,
                    "parameters": anonymize_request.parameters,
                    "operation": "anonymize",
                    "request_id": request_id,
                    "user_id": current_user["id"]
                }
            )
            
            # Handle different response formats
            if isinstance(anonymize_result, str):
                anonymized_text = anonymize_result
                anonymize_result = {
                    "anonymized_text": anonymized_text,
                    "language": anonymize_request.language,
                    "strategy": anonymize_request.strategy,
                    "model_used": anonymize_request.model_id or "anonymizer",
                    "processing_time": time.time() - start_time
                }
            elif isinstance(anonymize_result, dict):
                if "anonymized_text" not in anonymize_result:
                    # Try to extract from different response formats
                    if "result" in anonymize_result:
                        if isinstance(anonymize_result["result"], str):
                            anonymize_result["anonymized_text"] = anonymize_result["result"]
                        elif isinstance(anonymize_result["result"], dict) and "anonymized_text" in anonymize_result["result"]:
                            anonymize_result["anonymized_text"] = anonymize_result["result"]["anonymized_text"]
                    elif "text" in anonymize_result:
                        anonymize_result["anonymized_text"] = anonymize_result["text"]
        else:
            # Fallback implementation for anonymization
            logger.warning("Using fallback text anonymization implementation")
            start_time_fallback = time.time()
            
            # Basic fallback anonymization that just masks potential PII
            anonymized_text = anonymize_request.text
            entities_found = []
            
            # Define patterns for common PII
            import re
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            names_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            
            # Detect and mask emails
            if not anonymize_request.entities or "EMAIL" in anonymize_request.entities:
                for match in re.finditer(email_pattern, anonymized_text):
                    entities_found.append({"type": "EMAIL", "text": match.group(), "position": match.span()})
                    if anonymize_request.strategy == "mask" or anonymize_request.strategy == "redact":
                        anonymized_text = anonymized_text.replace(match.group(), "[EMAIL]") 
                    elif anonymize_request.strategy == "remove":
                        anonymized_text = anonymized_text.replace(match.group(), "")
                    elif anonymize_request.strategy == "replace":
                        anonymized_text = anonymized_text.replace(match.group(), "email@example.com")
            
            # Detect and mask phone numbers
            if not anonymize_request.entities or "PHONE" in anonymize_request.entities:
                for match in re.finditer(phone_pattern, anonymized_text):
                    entities_found.append({"type": "PHONE", "text": match.group(), "position": match.span()})
                    if anonymize_request.strategy == "mask" or anonymize_request.strategy == "redact":
                        anonymized_text = anonymized_text.replace(match.group(), "[PHONE]") 
                    elif anonymize_request.strategy == "remove":
                        anonymized_text = anonymized_text.replace(match.group(), "")
                    elif anonymize_request.strategy == "replace":
                        anonymized_text = anonymized_text.replace(match.group(), "555-555-5555")
            
            # Detect and mask names
            if not anonymize_request.entities or "PERSON" in anonymize_request.entities:
                for match in re.finditer(names_pattern, anonymized_text):
                    entities_found.append({"type": "PERSON", "text": match.group(), "position": match.span()})
                    if anonymize_request.strategy == "mask" or anonymize_request.strategy == "redact":
                        anonymized_text = anonymized_text.replace(match.group(), "[PERSON]") 
                    elif anonymize_request.strategy == "remove":
                        anonymized_text = anonymized_text.replace(match.group(), "")
                    elif anonymize_request.strategy == "replace":
                        anonymized_text = anonymized_text.replace(match.group(), "John Doe")
            
            # Create the result dictionary
            anonymize_result = {
                "anonymized_text": anonymized_text,
                "language": anonymize_request.language,
                "strategy": anonymize_request.strategy,
                "entities_found": entities_found,
                "model_used": "fallback_anonymizer",
                "processing_time": time.time() - start_time_fallback,
                "is_fallback": True
            }
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="anonymization",
            operation="anonymize",
            duration=process_time,
            input_size=len(anonymize_request.text),
            output_size=len(anonymize_result.get("anonymized_text", "")),
            success=True,
            metadata={
                "language": anonymize_request.language,
                "strategy": anonymize_request.strategy,
                "model_id": anonymize_result.get("model_used", anonymize_request.model_id or "anonymizer"),
                "entity_count": len(anonymize_result.get("entities_found", [])) if isinstance(anonymize_result.get("entities_found"), list) else 0
            }
        )
        
        # Create result model
        result = AnonymizeResult(
            source_text=anonymize_request.text,
            anonymized_text=anonymize_result.get("anonymized_text", ""),
            language=anonymize_result.get("language", anonymize_request.language),
            strategy=anonymize_result.get("strategy", anonymize_request.strategy),
            entities_found=anonymize_result.get("entities_found"),
            process_time=process_time,
            model_used=anonymize_result.get("model_used", anonymize_request.model_id or "anonymizer"),
            performance_metrics=anonymize_result.get("performance_metrics"),
            memory_usage=anonymize_result.get("memory_usage"),
            operation_cost=anonymize_result.get("operation_cost")
        )
        
        # Create response
        response = AnonymizeResponse(
            status=StatusEnum.SUCCESS,
            message="Text anonymization completed successfully",
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
        logger.error(f"Text anonymization error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="anonymization",
            operation="anonymize",
            duration=time.time() - start_time,
            input_size=len(anonymize_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": anonymize_request.language,
                "strategy": anonymize_request.strategy
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text anonymization error: {str(e)}"
        )

# ----- Text Analysis Endpoint -----

@router.post(
    "/analyze",
    response_model=TextAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze text",
    description="Performs analysis on text to extract insights."
)
async def analyze_text(
    request: Request,
    analysis_request: TextAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Analyze text to extract insights.
    
    This endpoint processes text and returns various analyses like sentiment, entities, and topics.
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
        
        # Prepare the analyses to perform
        analyses_to_perform = []
        if analysis_request.analyses:
            analyses_to_perform = analysis_request.analyses
        else:
            if analysis_request.include_sentiment:
                analyses_to_perform.append("sentiment")
            if analysis_request.include_entities:
                analyses_to_perform.append("entities")
            if analysis_request.include_topics:
                analyses_to_perform.append("topics")
            if analysis_request.include_summary:
                analyses_to_perform.append("summary")
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/analyze",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(analysis_request.text),
                "language": analysis_request.language,
                "analyses": analyses_to_perform,
                "model_id": analysis_request.model_id
            }
        )
        
        # Process analysis request
        if hasattr(processor, "analyze_text") and hasattr(processor, "analyzer") and processor.analyzer is not None:
            # Format the analysis types in the expected format
            analysis_types = {}
            for analysis_type in analyses_to_perform:
                analysis_types[analysis_type] = True
                
            # Prepare options
            options = {
                "analyses": analysis_types,
                "include_entities": analysis_request.include_entities,
                "include_sentiment": analysis_request.include_sentiment,
                "include_topics": analysis_request.include_topics,
                "include_summary": analysis_request.include_summary
            }
            
            if analysis_request.parameters:
                options.update(analysis_request.parameters)
                
            analysis_result = await processor.analyze_text(
                text=analysis_request.text,
                language=analysis_request.language,
                model_id=analysis_request.model_id,
                options=options,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "analyze"):
            analysis_result = await processor.analyze(
                text=analysis_request.text,
                language=analysis_request.language,
                analyses=analyses_to_perform,
                model_id=analysis_request.model_id,
                user_id=current_user["id"],
                request_id=request_id
            )
        else:
            # Fallback to process method
            analysis_result = await processor.process(
                content=analysis_request.text,
                options={
                    "language": analysis_request.language,
                    "analyses": analyses_to_perform,
                    "model_id": analysis_request.model_id,
                    "operation": "analyze",
                    "request_id": request_id,
                    "user_id": current_user["id"]
                }
            )
            
            # Handle different response formats
            if isinstance(analysis_result, dict) and "result" in analysis_result:
                if isinstance(analysis_result["result"], dict):
                    # Extract nested result object
                    analysis_result = analysis_result["result"]
        
        # All processing is complete
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Parse word and sentence counts if not provided
        word_count = analysis_result.get("word_count", len(analysis_request.text.split()))
        sentence_count = analysis_result.get("sentence_count", analysis_request.text.count(".") + analysis_request.text.count("!") + analysis_request.text.count("?"))
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="analysis",
            operation="analyze",
            duration=process_time,
            input_size=len(analysis_request.text),
            output_size=sum(len(str(analysis_result.get(key, ""))) for key in ["sentiment", "entities", "topics", "summary"] if key in analysis_result),
            success=True,
            metadata={
                "language": analysis_request.language,
                "analyses": analyses_to_perform,
                "model_id": analysis_request.model_id or "analysis",
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        )
        
        # Create result model
        result = TextAnalysisResult(
            text=analysis_request.text,
            language=analysis_result.get("language", analysis_request.language),
            sentiment=analysis_result.get("sentiment"),
            entities=analysis_result.get("entities"),
            topics=analysis_result.get("topics"),
            summary=analysis_result.get("summary"),
            word_count=word_count,
            sentence_count=sentence_count,
            process_time=process_time,
            performance_metrics=analysis_result.get("performance_metrics"),
            memory_usage=analysis_result.get("memory_usage"),
            operation_cost=analysis_result.get("operation_cost")
        )
        
        # Create response
        response = TextAnalysisResponse(
            status=StatusEnum.SUCCESS,
            message="Text analysis completed successfully",
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
        logger.error(f"Text analysis error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="analysis",
            operation="analyze",
            duration=time.time() - start_time,
            input_size=len(analysis_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": analysis_request.language,
                "analyses": analyses_to_perform if "analyses_to_perform" in locals() else []
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text analysis error: {str(e)}"
        )
        
# ----- Text Summarization Endpoint -----

class SummarizeRequest(BaseModel):
    text: str
    language: str = "en"
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    model_id: Optional[str] = None
    type: str = "extractive"  # Options: extractive, abstractive
    parameters: Optional[Dict[str, Any]] = None

class SummarizeResult(BaseModel):
    source_text: str
    summary: str
    language: str
    summary_type: str
    compression_ratio: float = 0.0
    word_count: Optional[int] = None
    sentence_count: Optional[int] = None
    process_time: float
    model_used: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, Any]] = None
    operation_cost: Optional[float] = None

class SummarizeResponse(BaseResponse[SummarizeResult]):
    pass

@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize text",
    description="Creates a shorter summary of longer text."
)
async def summarize_text(
    request: Request,
    summarize_request: SummarizeRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Summarize text to create a shorter version.
    
    This endpoint processes text and returns a concise summary.
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
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/pipeline/summarize",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "text_length": len(summarize_request.text),
                "language": summarize_request.language,
                "max_length": summarize_request.max_length,
                "min_length": summarize_request.min_length,
                "type": summarize_request.type,
                "model_id": summarize_request.model_id
            }
        )
        
        # Process summarization request
        if hasattr(processor, "summarize_text") and processor.summarizer is not None:
            # Map max/min length to the length parameter
            # Use "medium" as default, "short" for small max_length, "long" for large min_length
            length = "medium"
            if summarize_request.max_length and summarize_request.max_length < 100:
                length = "short"
            elif summarize_request.min_length and summarize_request.min_length > 300:
                length = "long"
                
            # Prepare options
            options = {
                "max_length": summarize_request.max_length,
                "min_length": summarize_request.min_length,
                "type": summarize_request.type
            }
            
            if summarize_request.parameters:
                options.update(summarize_request.parameters)
                
            summarize_result = await processor.summarize_text(
                text=summarize_request.text,
                length=length,
                language=summarize_request.language,
                model_id=summarize_request.model_id,
                options=options,
                user_id=current_user["id"],
                request_id=request_id
            )
        elif hasattr(processor, "summarize"):
            summarize_result = await processor.summarize(
                text=summarize_request.text,
                language=summarize_request.language,
                max_length=summarize_request.max_length,
                min_length=summarize_request.min_length,
                type=summarize_request.type,
                model_id=summarize_request.model_id,
                parameters=summarize_request.parameters,
                user_id=current_user["id"],
                request_id=request_id
            )
        else:
            # Fallback to process method
            summarize_result = await processor.process(
                content=summarize_request.text,
                options={
                    "language": summarize_request.language,
                    "max_length": summarize_request.max_length,
                    "min_length": summarize_request.min_length,
                    "type": summarize_request.type,
                    "model_id": summarize_request.model_id,
                    "parameters": summarize_request.parameters,
                    "operation": "summarize",
                    "request_id": request_id,
                    "user_id": current_user["id"]
                }
            )
            
            # Handle different response formats
            if isinstance(summarize_result, str):
                summary = summarize_result
                summarize_result = {
                    "summary": summary,
                    "language": summarize_request.language,
                    "type": summarize_request.type,
                    "model_used": summarize_request.model_id or "summarizer",
                    "processing_time": time.time() - start_time
                }
            elif isinstance(summarize_result, dict):
                if "summary" not in summarize_result:
                    # Try to extract from different response formats
                    if "result" in summarize_result:
                        if isinstance(summarize_result["result"], str):
                            summarize_result["summary"] = summarize_result["result"]
                        elif isinstance(summarize_result["result"], dict) and "summary" in summarize_result["result"]:
                            summarize_result["summary"] = summarize_result["result"]["summary"]
                    elif "text" in summarize_result:
                        summarize_result["summary"] = summarize_result["text"]
        
        # All summarization is complete
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Calculate compression ratio
        source_word_count = len(summarize_request.text.split())
        summary_word_count = len(summarize_result.get("summary", "").split())
        compression_ratio = 1.0 - (summary_word_count / source_word_count) if source_word_count > 0 else 0.0
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="summarization",
            operation="summarize",
            duration=process_time,
            input_size=len(summarize_request.text),
            output_size=len(summarize_result.get("summary", "")),
            success=True,
            metadata={
                "language": summarize_request.language,
                "type": summarize_request.type,
                "model_id": summarize_result.get("model_used", summarize_request.model_id or "summarizer"),
                "compression_ratio": compression_ratio,
                "source_word_count": source_word_count,
                "summary_word_count": summary_word_count
            }
        )
        
        # Create result model
        result = SummarizeResult(
            source_text=summarize_request.text,
            summary=summarize_result.get("summary", ""),
            language=summarize_result.get("language", summarize_request.language),
            summary_type=summarize_result.get("type", summarize_request.type),
            compression_ratio=compression_ratio,
            word_count=summary_word_count,
            sentence_count=summarize_result.get("summary", "").count(".") + summarize_result.get("summary", "").count("!") + summarize_result.get("summary", "").count("?"),
            process_time=process_time,
            model_used=summarize_result.get("model_used", summarize_request.model_id or "summarizer"),
            performance_metrics=summarize_result.get("performance_metrics"),
            memory_usage=summarize_result.get("memory_usage"),
            operation_cost=summarize_result.get("operation_cost")
        )
        
        # Create response
        response = SummarizeResponse(
            status=StatusEnum.SUCCESS,
            message="Text summarization completed successfully",
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
        logger.error(f"Text summarization error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="summarization",
            operation="summarize",
            duration=time.time() - start_time,
            input_size=len(summarize_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": summarize_request.language,
                "type": summarize_request.type
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text summarization error: {str(e)}"
        )