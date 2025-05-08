#!/usr/bin/env python3
"""
Test script for STT (Speech-to-Text) and TTS (Text-to-Speech) endpoints in CasaLingua.
"""
import os
import sys
import time
import json
import requests
import argparse
from pathlib import Path
import base64

# Base URL for API
BASE_URL = "http://localhost:8000"

# Create sample audio file if needed
SAMPLE_TEXT = "Hello, this is a test message for speech processing."
SAMPLE_AUDIO_PATH = "/tmp/test_speech_sample.mp3"
OUTPUT_AUDIO_PATH = "/tmp/tts_output.mp3"

def create_sample_audio(text=SAMPLE_TEXT, language="en", path=SAMPLE_AUDIO_PATH):
    """
    Create a sample audio file for testing STT.
    Uses gTTS (Google Text-to-Speech) if available.
    """
    print(f"Creating sample audio file at {path}...")
    
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=language)
        tts.save(path)
        print(f"Created sample audio with text: '{text}'")
        return path
    except ImportError:
        print("gTTS not available. Please install it with: pip install gtts")
        return None
    except Exception as e:
        print(f"Error creating sample audio: {e}")
        return None

def test_stt_endpoint(audio_path=SAMPLE_AUDIO_PATH):
    """
    Test the Speech-to-Text endpoint.
    """
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        create_sample_audio()
        if not os.path.exists(audio_path):
            print("Could not create sample audio file. Aborting STT test.")
            return False
    
    print(f"\n===== Testing STT Endpoint =====")
    url = f"{BASE_URL}/pipeline/stt"
    
    # Prepare the file for upload
    files = {
        'audio_file': open(audio_path, 'rb')
    }
    
    # Additional form data
    data = {
        'language': 'en',
        'detect_language': 'false',
        'enhanced_results': 'true'
    }
    
    print(f"Sending audio file {audio_path} to {url}...")
    start_time = time.time()
    
    try:
        response = requests.post(url, files=files, data=data)
        
        # Close the file
        files['audio_file'].close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Request completed in {duration:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            
            if "data" in result and "text" in result["data"]:
                print(f"\nTranscribed Text: {result['data']['text']}")
                return True
            else:
                print("No transcription in response")
        else:
            print(f"Error: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"Error testing STT endpoint: {e}")
        return False

def test_tts_endpoint(text=SAMPLE_TEXT, language="en", output_path=OUTPUT_AUDIO_PATH):
    """
    Test the Text-to-Speech endpoint if available.
    """
    print(f"\n===== Testing TTS Endpoint =====")
    
    # Check if TTS endpoint exists
    url = f"{BASE_URL}/pipeline/tts"
    
    # Data for the request
    data = {
        "text": text,
        "language": language,
        "voice": None,  # Use default voice
        "speed": 1.0,
        "pitch": 1.0,
        "output_format": "mp3"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Sending text to {url}...")
    print(f"Text: '{text}'")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Request completed in {duration:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response (metadata):")
            
            if "data" in result and "audio_content" in result["data"]:
                # Extract base64 audio content and save to file
                audio_content = result["data"]["audio_content"]
                if isinstance(audio_content, str):
                    # Decode base64 if provided in that format
                    try:
                        audio_bytes = base64.b64decode(audio_content)
                    except:
                        print("Audio content isn't valid base64, treating as raw bytes")
                        audio_bytes = audio_content.encode('latin1')
                else:
                    audio_bytes = audio_content
                
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                
                print(f"Audio saved to {output_path}")
                print(f"Duration: {result['data'].get('duration', 'unknown')} seconds")
                print(f"Model used: {result['data'].get('model_used', 'unknown')}")
                return True
            else:
                print("No audio content in response")
                print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"Error testing TTS endpoint: {e}")
        return False

def test_stt_languages_endpoint():
    """
    Test the STT languages endpoint.
    """
    print(f"\n===== Testing STT Languages Endpoint =====")
    url = f"{BASE_URL}/pipeline/stt/languages"
    
    print(f"Getting supported languages from {url}...")
    
    try:
        response = requests.get(url)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            
            if "data" in result and "languages" in result["data"]:
                print(f"\nSupported Languages: {len(result['data']['languages'])}")
                return True
            else:
                print("No languages data in response")
        else:
            print(f"Error: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"Error testing STT languages endpoint: {e}")
        return False

def test_tts_voices_endpoint(language="en"):
    """
    Test the TTS voices endpoint if available.
    """
    print(f"\n===== Testing TTS Voices Endpoint =====")
    url = f"{BASE_URL}/pipeline/tts/voices?language={language}"
    
    print(f"Getting available voices from {url}...")
    
    try:
        response = requests.get(url)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            
            if "data" in result and "voices" in result["data"]:
                print(f"\nAvailable Voices: {len(result['data']['voices'])}")
                return True
            else:
                print("No voices data in response")
        else:
            print(f"Error: {response.text}")
        
        return False
        
    except Exception as e:
        print(f"Error testing TTS voices endpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test STT and TTS endpoints")
    parser.add_argument("--stt", action="store_true", help="Test STT endpoint")
    parser.add_argument("--tts", action="store_true", help="Test TTS endpoint")
    parser.add_argument("--all", action="store_true", help="Test all endpoints")
    parser.add_argument("--text", type=str, default=SAMPLE_TEXT, help="Text to synthesize")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--input", type=str, default=SAMPLE_AUDIO_PATH, help="Input audio file for STT")
    parser.add_argument("--output", type=str, default=OUTPUT_AUDIO_PATH, help="Output audio file for TTS")
    
    args = parser.parse_args()
    
    # If no specific tests are selected, test all
    if not (args.stt or args.tts or args.all):
        args.all = True
    
    results = {}
    
    # Test STT if requested
    if args.stt or args.all:
        # Test STT languages endpoint
        results["stt_languages"] = test_stt_languages_endpoint()
        
        # Test STT endpoint
        results["stt"] = test_stt_endpoint(args.input)
    
    # Test TTS if requested
    if args.tts or args.all:
        # Test TTS voices endpoint
        results["tts_voices"] = test_tts_voices_endpoint(args.language)
        
        # Test TTS endpoint
        results["tts"] = test_tts_endpoint(args.text, args.language, args.output)
    
    # Print summary
    print("\n===== Test Summary =====")
    for test, result in results.items():
        print(f"{test}: {'✅ PASS' if result else '❌ FAIL'}")
    
    # Return success if all tests passed
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())