#!/usr/bin/env python3
"""
Enhanced test script for STT (Speech-to-Text) and TTS (Text-to-Speech) endpoints in CasaLingua.
"""

import os
import sys
import time
import json
import requests
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Define the base URL for API requests
BASE_URL = "http://localhost:8000"

# Set authentication headers (if needed)
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "dev-key"  # Replace with your actual API key if needed
}

def get_api_base_url() -> str:
    """Get the base URL for API requests."""
    custom_url = os.environ.get("CASALINGUA_API_URL")
    return custom_url if custom_url else BASE_URL

def get_http_headers() -> Dict[str, str]:
    """Get the HTTP headers for API requests."""
    headers = dict(HEADERS)
    api_key = os.environ.get("CASALINGUA_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers

def print_separator(title: str):
    """Print a separator with a title."""
    print("\n" + "="*5 + f" {title} " + "="*5)

def test_stt_languages() -> bool:
    """Test the STT languages endpoint."""
    print_separator("Testing STT Languages Endpoint")
    
    url = f"{get_api_base_url()}/pipeline/stt/languages"
    print(f"Getting supported languages from {url}...")
    
    try:
        response = requests.get(url, headers=get_http_headers())
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"Response:\n{json.dumps(response_json, indent=2)}")
            
            if "data" in response_json and "languages" in response_json["data"]:
                languages = response_json["data"].get("languages", [])
                print(f"Supported Languages: {len(languages)}")
                return True
            else:
                print("❌ Error: Missing 'languages' data in response")
        else:
            print(f"Error: {response.text}")
        
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_tts_voices(language: str = "en") -> bool:
    """Test the TTS voices endpoint."""
    print_separator("Testing TTS Voices Endpoint")
    
    url = f"{get_api_base_url()}/pipeline/tts/voices?language={language}"
    print(f"Getting available voices from {url}...")
    
    try:
        response = requests.get(url, headers=get_http_headers())
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"Response:\n{json.dumps(response_json, indent=2)}")
            
            if "data" in response_json and "voices" in response_json["data"]:
                voices = response_json["data"].get("voices", [])
                print(f"Available Voices: {len(voices)}")
                return True
            else:
                print("❌ Error: Missing 'voices' data in response")
        else:
            print(f"Error: {response.text}")
        
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_tts_endpoint(text: str, language: str = "en", voice: Optional[str] = None) -> bool:
    """Test the TTS endpoint."""
    print_separator("Testing TTS Endpoint")
    
    url = f"{get_api_base_url()}/pipeline/tts"
    print(f"Sending text to {url}...")
    print(f"Text: '{text}'")
    
    request_data = {
        "text": text,
        "language": language,
        "output_format": "mp3"
    }
    
    if voice:
        request_data["voice"] = voice
    
    try:
        start_time = time.time()
        response = requests.post(url, json=request_data, headers=get_http_headers())
        process_time = time.time() - start_time
        print(f"Request completed in {process_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            audio_url = response_json.get("data", {}).get("audio_url")
            
            if audio_url:
                print(f"Audio URL: {audio_url}")
                
                # Download the audio file
                full_audio_url = f"{get_api_base_url()}{audio_url}"
                print(f"Downloading audio from {full_audio_url}...")
                
                audio_response = requests.get(full_audio_url, headers=get_http_headers())
                
                if audio_response.status_code == 200:
                    # Save the audio to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                        temp_file.write(audio_response.content)
                        print(f"Audio saved to {temp_file.name}")
                        
                    print(f"Audio size: {len(audio_response.content)} bytes")
                    return True
                else:
                    print(f"❌ Error downloading audio: {audio_response.text}")
            else:
                print("❌ Error: No audio URL in response")
        else:
            print(f"Error: {response.text}")
        
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def create_sample_audio_file(text: str, language: str = "en") -> Optional[str]:
    """Create a sample audio file for testing STT."""
    print(f"Creating sample audio file with text: '{text}'...")
    
    # Use gTTS to create a sample audio file
    try:
        from gtts import gTTS
        
        audio_file = "/tmp/test_speech_sample.mp3"
        tts = gTTS(text=text, lang=language)
        tts.save(audio_file)
        
        print(f"Sample audio file created at {audio_file}")
        return audio_file
    except ImportError:
        print("gTTS not available. Please install it with: pip install gtts")
        return None
    except Exception as e:
        print(f"Error creating sample audio file: {str(e)}")
        return None

def test_stt_endpoint(audio_file_path: str, language: Optional[str] = None) -> bool:
    """Test the STT endpoint with an audio file."""
    print_separator("Testing STT Endpoint")
    
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        audio_file_path = create_sample_audio_file("Hello, this is a test message for speech recognition.")
        
        if not audio_file_path:
            print("Could not create sample audio file. Aborting STT test.")
            return False
    
    url = f"{get_api_base_url()}/pipeline/stt"
    print(f"Sending audio file to {url}...")
    
    try:
        # Prepare form data
        form_data = {}
        if language:
            form_data["language"] = language
        form_data["detect_language"] = "true" if not language else "false"
        
        # Prepare files
        with open(audio_file_path, "rb") as audio_file:
            files = {"audio_file": (os.path.basename(audio_file_path), audio_file, "audio/mpeg")}
            
            # Send request
            start_time = time.time()
            response = requests.post(url, data=form_data, files=files, headers=get_http_headers())
            process_time = time.time() - start_time
            
            print(f"Request completed in {process_time:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                print(f"Response:\n{json.dumps(response_json, indent=2)}")
                
                if "data" in response_json and "text" in response_json["data"]:
                    transcribed_text = response_json["data"]["text"]
                    print(f"Transcribed text: '{transcribed_text}'")
                    return True
                else:
                    print("❌ Error: Missing 'text' data in response")
            else:
                print(f"Error: {response.text}")
            
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main function to run the tests."""
    # Test results tracking
    results = {
        "stt_languages": False,
        "stt": False,
        "tts_voices": False,
        "tts": False
    }
    
    # Test STT languages endpoint
    results["stt_languages"] = test_stt_languages()
    
    # Test STT endpoint
    audio_file = "/tmp/test_speech_sample.mp3"
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        audio_file = create_sample_audio_file("Hello, this is a test message for speech recognition.")
    
    if audio_file:
        results["stt"] = test_stt_endpoint(audio_file)
    
    # Test TTS voices endpoint
    results["tts_voices"] = test_tts_voices()
    
    # Test TTS endpoint
    results["tts"] = test_tts_endpoint("Hello, this is a test message for speech processing.")
    
    # Print test summary
    print_separator("Test Summary")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    main()