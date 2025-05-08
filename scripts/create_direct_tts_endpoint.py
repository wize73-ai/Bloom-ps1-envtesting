#\!/usr/bin/env python3
"""
Create a direct TTS endpoint using gTTS.
This script creates a standalone TTS endpoint that bypasses the model manager.
"""

import os
import sys
import importlib
import time
import json
from pathlib import Path

def create_endpoint_file():
    """Create a direct TTS endpoint implementation."""
    # Define the path for our direct endpoint
    endpoint_path = "app/api/routes/direct_tts.py"
    
    # Define the content for the direct endpoint
    endpoint_content = '''"""
Direct TTS endpoint using gTTS for guaranteed audible output.
This is a standalone implementation that bypasses the model manager.
"""

import os
import io
import uuid
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response, BackgroundTasks, Header
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.api.schemas.base import ApiResponse
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
'''
    
    # Write the endpoint file
    try:
        with open(endpoint_path, "w") as f:
            f.write(endpoint_content)
        
        print(f"Created direct TTS endpoint at {endpoint_path}")
        return True
    except Exception as e:
        print(f"Error creating endpoint file: {str(e)}")
        return False

def update_main_router():
    """Update the main router to include our direct endpoint."""
    # Path to main.py
    main_path = "app/main.py"
    
    try:
        with open(main_path, "r") as f:
            content = f.read()
        
        # Check if we need to import our direct_tts module
        if "from app.api.routes import direct_tts" not in content:
            # Find the imports section
            import_pattern = "from app.api.routes import "
            imports_start = content.find(import_pattern)
            
            if imports_start >= 0:
                # Find the end of the imports section
                imports_end = content.find("\n\n", imports_start)
                if imports_end < 0:
                    imports_end = content.find("\n", imports_start)
                
                if imports_end > imports_start:
                    # Add our import at the end of the imports section
                    new_content = (
                        content[:imports_end] + 
                        "\nfrom app.api.routes import direct_tts  # Direct gTTS endpoint" + 
                        content[imports_end:]
                    )
                    
                    # Update the content
                    content = new_content
        
        # Check if we need to add our router
        if "app.include_router(direct_tts.router)" not in content:
            # Find where routers are added
            router_pattern = "app.include_router("
            router_pos = content.find(router_pattern)
            
            if router_pos >= 0:
                # Find a good location to add our router
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if router_pattern in line and i < len(lines) - 1:
                        # Add our router after an existing router
                        lines.insert(i + 1, "    app.include_router(direct_tts.router)  # Direct TTS endpoint")
                        content = "\n".join(lines)
                        break
        
        # Write the updated content
        with open(main_path, "w") as f:
            f.write(content)
        
        print(f"Updated {main_path} to include direct TTS endpoint")
        return True
    except Exception as e:
        print(f"Error updating main router: {str(e)}")
        return False

def create_test_script():
    """Create a test script for the direct TTS endpoint."""
    # Path to test script
    test_path = "scripts/test_direct_tts.py"
    
    # Define the test script content
    test_content = '''#\!/usr/bin/env python3
"""
Test script for direct TTS endpoint.
This script tests the direct TTS endpoint that uses gTTS.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

def test_direct_tts_endpoint():
    """Test the direct TTS endpoint with different parameters."""
    # Create output directory
    os.makedirs("temp/direct_tts_test", exist_ok=True)
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "name": "english_short",
            "text": "This is a test message for the direct text to speech endpoint.",
            "language": "en",
            "output_format": "mp3"
        },
        {
            "name": "spanish_short",
            "text": "Este es un mensaje de prueba para el punto final directo de texto a voz.",
            "language": "es",
            "output_format": "mp3"
        },
        {
            "name": "french_short",
            "text": "Ceci est un message de test pour le point de terminaison direct de synthèse vocale.",
            "language": "fr",
            "output_format": "mp3"
        }
    ]
    
    print("===== DIRECT TTS ENDPOINT TEST =====")
    
    # Test each case
    for case in test_cases:
        print(f"\\nTesting case: {case['name']}")
        
        # Create request data
        request_data = {
            "text": case["text"],
            "language": case["language"],
            "output_format": case["output_format"]
        }
        
        # Output file
        output_file = f"temp/direct_tts_test/direct_tts_{case['name']}.{case['output_format']}"
        
        # Make request
        print(f"Making request to /direct_tts/generate with text: '{request_data['text'][:30]}...'")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/direct_tts/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            request_time = time.time() - start_time
            
            # Check response
            if response.status_code == 200:
                # Save audio file
                with open(output_file, "wb") as f:
                    f.write(response.content)
                
                # Get file size
                file_size = os.path.getsize(output_file)
                
                print(f"✅ Success\! Output saved to {output_file}")
                print(f"   File size: {file_size} bytes")
                print(f"   Request time: {request_time:.2f} seconds")
                
                # Check if file is too small (potential silent file)
                if file_size < 10000:
                    print(f"⚠️  WARNING: File size is relatively small ({file_size} bytes)")
                    print("   This might be a low-quality audio file.")
                else:
                    print(f"   File size looks good ({file_size} bytes)")
                
                # Suggest command to play the file
                print(f"   To play: open {output_file}")
            else:
                print(f"❌ Error: Status code {response.status_code}")
                print(f"   Response: {response.text}")
        
        except Exception as e:
            print(f"❌ Error making request: {str(e)}")
    
    print("\\n===== TEST COMPLETE =====")
    print("Test files have been saved to temp/direct_tts_test directory.")
    print("You can play these files to verify they contain audible speech.")

if __name__ == "__main__":
    test_direct_tts_endpoint()
'''
    
    # Write the test script
    try:
        with open(test_path, "w") as f:
            f.write(test_content)
        
        # Make it executable
        os.chmod(test_path, 0o755)
        
        print(f"Created test script at {test_path}")
        return True
    except Exception as e:
        print(f"Error creating test script: {str(e)}")
        return False

def main():
    """Main function."""
    print("===== CREATING DIRECT TTS ENDPOINT =====")
    
    # Create endpoint file
    endpoint_ok = create_endpoint_file()
    if not endpoint_ok:
        print("Failed to create direct TTS endpoint")
        return False
    
    # Update main router
    router_ok = update_main_router()
    if not router_ok:
        print("Failed to update main router")
        return False
    
    # Create test script
    test_ok = create_test_script()
    if not test_ok:
        print("Failed to create test script")
        return False
    
    print("\n===== DIRECT TTS ENDPOINT CREATED =====")
    print("The direct TTS endpoint has been created and added to the main router.")
    print("Restart the server for changes to take effect.")
    print("\nOnce the server is restarted, you can test the endpoint with:")
    print("python scripts/test_direct_tts.py")

if __name__ == "__main__":
    main()
