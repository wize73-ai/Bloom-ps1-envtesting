"""
Direct TTS endpoint using gTTS for guaranteed audible output.
This is a standalone implementation that bypasses the model manager.
"""

import os
import io
import uuid
import time
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response, BackgroundTasks, Header
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.api.schemas.base import BaseResponse
from app.utils.helpers import get_timestamp

# Import gTTS
from gtts import gTTS

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/direct_tts",
    tags=["direct_tts"],
    responses={404: {"description": "Not found"}},
)

# Create temp directory for audio files
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)

@router.post("/generate", response_class=StreamingResponse)
async def generate_speech(
    request: Request,
    background_tasks: BackgroundTasks,
    accept: Optional[str] = Header(None)
):
    """
    Generate speech from text using gTTS.
    
    This is a direct implementation that bypasses the model manager,
    guaranteeing audible output using Google Text-to-Speech.
    
    Returns:
        StreamingResponse: The audio file as a streaming response.
    """
    try:
        # Parse JSON request body
        request_data = await request.json()
        
        # Extract parameters
        text = request_data.get("text", "")
        language = request_data.get("language", "en")
        output_format = request_data.get("output_format", "mp3")
        
        # Validate parameters
        if not text:
            raise HTTPException(status_code=400, detail="Text parameter is required")
        
        # Create a unique ID for this request
        request_id = str(uuid.uuid4())
        output_file = temp_dir / f"direct_tts_{request_id}.mp3"
        
        # Log the request
        logger.info(f"Direct TTS request: text='{text[:50]}...', language={language}")
        
        # Create gTTS object
        start_time = time.time()
        
        # Make sure language is in correct format for gTTS
        gtts_lang = language
        if len(language) > 2 and '-' in language:
            gtts_lang = language.split('-')[0]
        
        # Create gTTS object and save to file
        try:
            tts = gTTS(text=text, lang=gtts_lang)
            tts.save(str(output_file))
            
            # Add background task to delete file after some time
            background_tasks.add_task(delete_file, output_file, delay=300)  # Delete after 5 minutes
            
            # Get file info
            file_size = output_file.stat().st_size
            processing_time = time.time() - start_time
            
            logger.info(f"Speech generated successfully: {output_file} ({file_size} bytes) in {processing_time:.2f}s")
            
            # Check if the expected format is JSON
            if accept and "application/json" in accept:
                # Return JSON response with file URL
                return JSONResponse({
                    "status": "success",
                    "audio_file": str(output_file),
                    "file_size": file_size,
                    "processing_time": processing_time,
                    "language": language,
                    "format": "mp3",
                    "timestamp": get_timestamp()
                })
            
            # Get file size
            file_size = output_file.stat().st_size
            
            # Check if file seems valid
            if file_size < 1000:
                logger.warning(f"Generated file is suspiciously small: {file_size} bytes")
            
            # Return the file as a streaming response
            return StreamingResponse(
                open(output_file, "rb"),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"attachment; filename=speech_{request_id}.mp3",
                    "Content-Length": str(file_size),
                    "X-Processing-Time": str(processing_time),
                    "Cache-Control": "max-age=3600"  # Cache for 1 hour
                }
            )
        except Exception as tts_error:
            logger.error(f"Error generating speech with gTTS: {str(tts_error)}")
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(tts_error)}")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error handling TTS request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error handling request: {str(e)}")

async def delete_file(file_path: Path, delay: int = 300):
    """Delete a file after a delay."""
    try:
        await asyncio.sleep(delay)
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
