#!/usr/bin/env python3
"""
Simple test script for the Speech-to-Text endpoint.
"""

import os
import uuid
import requests
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file."""
    temp_dir = Path("temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_file = temp_dir / f"test_stt_{uuid.uuid4()}.mp3"
    
    # Create a minimal MP3 file
    with open(audio_file, "wb") as f:
        # Simple MP3 file header + minimal data
        silence_mp3 = b'\xFF\xFB\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        f.write(silence_mp3 * 100)  # Repeat to make it longer
    
    print(f"Created test audio file: {audio_file}")
    return audio_file

def test_stt():
    """Test the STT endpoint with a simple audio file."""
    audio_file = create_test_audio()
    
    url = "http://localhost:8000/pipeline/stt"
    print(f"Testing STT endpoint: {url}")
    
    # Create form data
    form_data = {
        "language": "en",
        "detect_language": "false",
        "enhanced_results": "true"
    }
    
    # Create file to upload
    files = {
        "audio_file": (os.path.basename(audio_file), open(audio_file, "rb"), "audio/mpeg")
    }
    
    try:
        # Send request
        response = requests.post(url, data=form_data, files=files)
        print(f"Status code: {response.status_code}")
        
        # Process response
        if response.status_code == 200:
            response_json = response.json()
            print(f"Response: {response_json}")
            
            # Check for text in response
            if "data" in response_json and "text" in response_json["data"]:
                text = response_json["data"]["text"]
                print(f"Transcribed text: '{text}'")
                print("✅ TEST PASSED: Got transcription text")
                return True
            else:
                print("❌ TEST FAILED: No transcription text in response")
        else:
            print(f"❌ TEST FAILED: {response.text}")
        
        return False
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        return False
    finally:
        # Close file
        files["audio_file"][1].close()

if __name__ == "__main__":
    test_stt()