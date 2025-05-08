#!/usr/bin/env python3
"""
Comprehensive test script for all STT (Speech-to-Text) and TTS (Text-to-Speech) endpoints.
Tests all speech-related functionality including language detection, voices, audio generation, and transcription.
"""

import os
import sys
import time
import json
import requests
import uuid
import argparse
from pathlib import Path

# Define base URL for API requests
BASE_URL = "http://localhost:8000"

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "="*10 + f" {title} " + "="*10)

def ensure_temp_directory():
    """Ensure temp directory exists for audio files."""
    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def create_test_audio_file(text="This is a test audio file for speech to text", language="en"):
    """Create a test audio file for STT testing."""
    temp_dir = ensure_temp_directory()
    audio_file = temp_dir / f"test_audio_{uuid.uuid4()}.mp3"
    
    try:
        # Try using gTTS to create an actual audio file
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=language)
            tts.save(str(audio_file))
            print(f"Created test audio file with gTTS: {audio_file}")
            return audio_file
        except ImportError:
            print("gTTS not available, creating fallback audio file")
        
        # Fallback: Create a simple audio file
        with open(audio_file, "wb") as f:
            # Simple MP3 file header + minimal data
            silence_mp3 = b'\xFF\xFB\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            f.write(silence_mp3 * 100)  # Repeat to make it longer
        
        print(f"Created fallback audio file: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"Error creating test audio file: {str(e)}")
        return None

def test_stt_languages():
    """Test the STT languages endpoint."""
    print_separator("Testing STT Languages")
    
    url = f"{BASE_URL}/pipeline/stt/languages"
    print(f"GET {url}")
    
    try:
        response = requests.get(url)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            
            # Verify languages are present
            languages = data.get("data", {}).get("languages", [])
            print(f"Found {len(languages)} languages")
            
            if languages:
                print("Languages found:")
                for lang in languages:
                    print(f"  - {lang.get('code')}: {lang.get('name')}")
                return True
            else:
                print("No languages found in response")
        else:
            print(f"Error: {response.text}")
        
        return False
    except Exception as e:
        print(f"Error accessing STT languages endpoint: {str(e)}")
        return False

def test_tts_voices():
    """Test the TTS voices endpoint."""
    print_separator("Testing TTS Voices")
    
    url = f"{BASE_URL}/pipeline/tts/voices"
    print(f"GET {url}")
    
    try:
        response = requests.get(url)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify voices are present
            voices = data.get("data", {}).get("voices", [])
            print(f"Found {len(voices)} voices")
            
            if voices:
                print("Voices found:")
                for voice in voices[:5]:  # Show first 5 voices
                    print(f"  - {voice.get('id')}: {voice.get('name')} ({voice.get('language')})")
                if len(voices) > 5:
                    print(f"  ... and {len(voices) - 5} more")
                return True
            else:
                print("No voices found in response")
        else:
            print(f"Error: {response.text}")
        
        return False
    except Exception as e:
        print(f"Error accessing TTS voices endpoint: {str(e)}")
        return False

def test_tts_endpoint(text="Hello, this is a test of the text-to-speech system.", language="en"):
    """Test the TTS endpoint."""
    print_separator("Testing TTS Endpoint")
    
    url = f"{BASE_URL}/pipeline/tts"
    print(f"POST {url}")
    print(f"Text: '{text}'")
    print(f"Language: {language}")
    
    request_data = {
        "text": text,
        "language": language,
        "output_format": "mp3"
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=request_data)
        elapsed_time = time.time() - start_time
        print(f"Response time: {elapsed_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for audio URL
            audio_url = data.get("data", {}).get("audio_url")
            if audio_url:
                print(f"Audio URL: {audio_url}")
                
                # Download the audio file
                audio_download_url = f"{BASE_URL}{audio_url}"
                print(f"Downloading audio from: {audio_download_url}")
                
                audio_response = requests.get(audio_download_url)
                if audio_response.status_code == 200:
                    # Save the audio file
                    audio_file = ensure_temp_directory() / f"downloaded_tts_{uuid.uuid4()}.mp3"
                    with open(audio_file, "wb") as f:
                        f.write(audio_response.content)
                    
                    print(f"Downloaded audio file: {audio_file} ({len(audio_response.content)} bytes)")
                    return True, audio_file
                else:
                    print(f"Error downloading audio: {audio_response.status_code}")
            else:
                print("No audio URL found in response")
        else:
            print(f"Error: {response.text}")
        
        return False, None
    except Exception as e:
        print(f"Error accessing TTS endpoint: {str(e)}")
        return False, None

def test_stt_endpoint(audio_file=None, language=None):
    """Test the STT endpoint."""
    print_separator("Testing STT Endpoint")
    
    # Create audio file if not provided
    if not audio_file or not os.path.exists(audio_file):
        audio_file = create_test_audio_file()
        if not audio_file:
            print("Failed to create test audio file, skipping STT test")
            return False
    
    url = f"{BASE_URL}/pipeline/stt"
    print(f"POST {url} (multipart/form-data)")
    print(f"Audio file: {audio_file}")
    print(f"Language: {language if language else 'auto-detect'}")
    
    # Prepare form data
    form_data = {}
    if language:
        form_data["language"] = language
    form_data["detect_language"] = "false" if language else "true"
    
    try:
        with open(audio_file, "rb") as f:
            files = {
                "audio_file": (os.path.basename(audio_file), f, "audio/mpeg")
            }
            
            start_time = time.time()
            response = requests.post(url, data=form_data, files=files)
            elapsed_time = time.time() - start_time
            print(f"Response time: {elapsed_time:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for transcribed text
                text = data.get("data", {}).get("text", "")
                language = data.get("data", {}).get("language", "")
                
                if text:
                    print(f"Transcribed text: '{text}'")
                    print(f"Detected language: {language}")
                    return True
                else:
                    print("No transcribed text found in response")
            else:
                print(f"Error: {response.text}")
            
            return False
    except Exception as e:
        print(f"Error accessing STT endpoint: {str(e)}")
        return False

def run_all_tests():
    """Run all speech endpoint tests."""
    print_separator("TESTING ALL SPEECH ENDPOINTS")
    
    # Dictionary to store test results
    results = {
        "stt_languages": False,
        "tts_voices": False,
        "tts_endpoint": False,
        "stt_endpoint": False
    }
    
    # Test STT languages endpoint
    results["stt_languages"] = test_stt_languages()
    
    # Test TTS voices endpoint
    results["tts_voices"] = test_tts_voices()
    
    # Test TTS endpoint
    results["tts_endpoint"], audio_file = test_tts_endpoint()
    
    # Test STT endpoint
    results["stt_endpoint"] = test_stt_endpoint()
    
    # Print summary
    print_separator("TEST SUMMARY")
    passed = 0
    failed = 0
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if failed > 0:
        print("\nSuggestions for fixing failed tests:")
        print("1. Check server logs for error messages")
        print("2. Verify all required packages are installed:")
        print("   python scripts/install_tts_requirements.py")
        print("3. Run the server with monitoring:")
        print("   python scripts/monitor_and_test_speech.py")
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test all speech endpoints")
    parser.add_argument("--tts-only", action="store_true", help="Test only TTS endpoints")
    parser.add_argument("--stt-only", action="store_true", help="Test only STT endpoints")
    parser.add_argument("--text", type=str, help="Text to synthesize for TTS test")
    parser.add_argument("--language", type=str, help="Language code for tests (e.g., en, es)")
    args = parser.parse_args()
    
    if args.tts_only:
        # Test only TTS endpoints
        test_tts_voices()
        test_tts_endpoint(text=args.text or "Hello, this is a TTS test.", language=args.language or "en")
    elif args.stt_only:
        # Test only STT endpoints
        test_stt_languages()
        test_stt_endpoint(language=args.language)
    else:
        # Test all endpoints
        run_all_tests()

if __name__ == "__main__":
    main()