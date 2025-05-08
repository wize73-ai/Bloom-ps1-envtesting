#!/usr/bin/env python3
"""
Test script for TTS endpoint after fixing syntax errors.
This script tests the emergency fallback functionality.
"""

import os
import json
import uuid
import requests
import time
from pathlib import Path
import argparse

# Define base URL for API
BASE_URL = "http://localhost:8000"

def print_separator(title):
    """Print a separator with title."""
    print("\n" + "="*5 + f" {title} " + "="*5)

def ensure_temp_directory():
    """Ensure that the temp directory exists."""
    temp_dir = Path("temp")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Ensured temp directory exists at {temp_dir}")
    return temp_dir

def create_emergency_audio(file_format="mp3"):
    """Create an emergency audio file."""
    temp_dir = ensure_temp_directory()
    audio_id = str(uuid.uuid4())
    audio_path = temp_dir / f"tts_emergency_{audio_id}.{file_format}"
    
    with open(audio_path, "wb") as f:
        if file_format == "mp3":
            # Simple MP3 file header + minimal data
            silence_mp3 = b'\xFF\xFB\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            f.write(silence_mp3 * 100)  # Repeat to make it longer
        elif file_format == "wav":
            # Simple WAV header + silence
            sample_rate = 44100
            bits_per_sample = 16
            channels = 1
            
            # Write WAV header
            f.write(b'RIFF')
            f.write((36).to_bytes(4, byteorder='little'))  # Chunk size
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write((16).to_bytes(4, byteorder='little'))  # Subchunk1 size
            f.write((1).to_bytes(2, byteorder='little'))   # Audio format (PCM)
            f.write(channels.to_bytes(2, byteorder='little'))
            f.write(sample_rate.to_bytes(4, byteorder='little'))
            byte_rate = sample_rate * channels * bits_per_sample // 8
            f.write(byte_rate.to_bytes(4, byteorder='little'))
            block_align = channels * bits_per_sample // 8
            f.write(block_align.to_bytes(2, byteorder='little'))
            f.write(bits_per_sample.to_bytes(2, byteorder='little'))
            f.write(b'data')
            f.write((0).to_bytes(4, byteorder='little'))  # Subchunk2 size (0 = empty)
        else:
            # For other formats, just write some bytes as placeholder
            f.write(b'\x00' * 1024)
    
    print(f"Created emergency audio file: {audio_path}")
    return audio_path

def test_tts_endpoint():
    """Test the TTS endpoint with a simple text."""
    print_separator("Testing TTS Endpoint")
    
    url = f"{BASE_URL}/pipeline/tts"
    text = "Hello, this is a test of the emergency TTS system."
    language = "en"
    
    print(f"Sending request to {url}")
    print(f"Text: '{text}'")
    
    request_data = {
        "text": text,
        "language": language,
        "output_format": "mp3"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=request_data, headers=headers)
        process_time = time.time() - start_time
        
        print(f"Request completed in {process_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print("Response data:")
            print(json.dumps(response_json, indent=2))
            
            # Get audio URL
            audio_url = response_json.get("data", {}).get("audio_url")
            if audio_url:
                print(f"Audio URL: {audio_url}")
                
                # Try to download the audio
                full_audio_url = f"{BASE_URL}{audio_url}"
                print(f"Downloading audio from {full_audio_url}")
                
                audio_response = requests.get(full_audio_url)
                if audio_response.status_code == 200:
                    # Save to a temp file
                    audio_path = f"temp/downloaded_tts_{uuid.uuid4()}.mp3"
                    with open(audio_path, "wb") as f:
                        f.write(audio_response.content)
                    
                    print(f"Successfully downloaded audio to {audio_path}")
                    print(f"Audio size: {len(audio_response.content)} bytes")
                    return True
                else:
                    print(f"Error downloading audio: {audio_response.status_code}")
            else:
                print("No audio URL found in response")
        else:
            print(f"Error: {response.text}")
        
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_tts_voices_endpoint():
    """Test the TTS voices endpoint."""
    print_separator("Testing TTS Voices Endpoint")
    
    url = f"{BASE_URL}/pipeline/tts/voices"
    print(f"Sending request to {url}")
    
    try:
        response = requests.get(url)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print("Response data:")
            print(json.dumps(response_json, indent=2))
            
            voices = response_json.get("data", {}).get("voices", [])
            print(f"Found {len(voices)} voices")
            return True
        else:
            print(f"Error: {response.text}")
        
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Test TTS endpoints after fix')
    parser.add_argument('--create-audio', action='store_true', help='Create emergency audio files')
    args = parser.parse_args()
    
    # Create emergency audio files if requested
    if args.create_audio:
        create_emergency_audio("mp3")
        create_emergency_audio("wav")
    
    # Test TTS voices endpoint
    voices_result = test_tts_voices_endpoint()
    
    # Test TTS endpoint
    tts_result = test_tts_endpoint()
    
    # Print summary
    print_separator("Test Summary")
    print(f"TTS Voices Endpoint: {'✅ PASS' if voices_result else '❌ FAIL'}")
    print(f"TTS Endpoint: {'✅ PASS' if tts_result else '❌ FAIL'}")
    
    if not voices_result or not tts_result:
        print("\nSuggestions for debugging:")
        print("1. Check server logs for detailed error messages")
        print("2. Run 'python scripts/install_tts_requirements.py' to install needed packages")
        print("3. Create emergency audio files with '--create-audio' flag")
        print("4. Try running 'python scripts/monitor_and_test_speech.py' to watch logs while testing")

if __name__ == "__main__":
    main()