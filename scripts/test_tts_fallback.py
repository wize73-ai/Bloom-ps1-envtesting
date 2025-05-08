#!/usr/bin/env python3
"""
Test script to verify TTS fallback functionality.
This script tests the direct TTS fallback mechanism that should work even if the primary model fails.
"""

import os
import sys
import requests
import json
import tempfile
from pathlib import Path

# Define base URL
BASE_URL = "http://localhost:8000"

def test_tts_direct_endpoint():
    """Test the direct TTS endpoint using gTTS fallback."""
    print("Testing direct TTS endpoint...")
    
    # Endpoint URL
    url = f"{BASE_URL}/api/v1/tts"
    
    # Test data
    test_data = {
        "text": "This is a test of the text to speech fallback system.",
        "language": "en"
    }
    
    # Send request
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            # Save audio to file
            temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            audio_file = os.path.join(temp_dir, "test_tts_fallback.mp3")
            
            with open(audio_file, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Success! TTS endpoint returned {len(response.content)} bytes of audio")
            print(f"Audio saved to: {audio_file}")
            print(f"To play: open {audio_file}")
            
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error making request: {e}")
        return False

def test_tts_pipeline_endpoint():
    """Test the pipeline TTS endpoint."""
    print("\nTesting pipeline TTS endpoint...")
    
    # Endpoint URL
    url = f"{BASE_URL}/api/v1/pipeline/tts"
    
    # Test data
    test_data = {
        "text": "This is a test of the pipeline text to speech system.",
        "language": "en",
        "output_format": "mp3"
    }
    
    # Send request
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Get audio URL from response
                if "data" in result and "audio_url" in result["data"]:
                    audio_url = result["data"]["audio_url"]
                    print(f"Audio URL: {audio_url}")
                    
                    # Download the audio
                    audio_response = requests.get(f"{BASE_URL}{audio_url}")
                    
                    if audio_response.status_code == 200:
                        # Save to file
                        temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp")
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        audio_file = os.path.join(temp_dir, "test_tts_pipeline_fallback.mp3")
                        
                        with open(audio_file, "wb") as f:
                            f.write(audio_response.content)
                        
                        print(f"✅ Success! Pipeline TTS endpoint returned audio")
                        print(f"Audio saved to: {audio_file}")
                        print(f"To play: open {audio_file}")
                        
                        return True
                    else:
                        print(f"❌ Error downloading audio: {audio_response.status_code}")
                        print(audio_response.text)
                        return False
                else:
                    print(f"❌ No audio URL in response: {result}")
                    return False
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response: {response.text[:100]}...")
                return False
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error making request: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing TTS Fallback Mechanisms ===\n")
    
    # Test both endpoints
    direct_success = test_tts_direct_endpoint()
    pipeline_success = test_tts_pipeline_endpoint()
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Direct TTS endpoint: {'✅ Success' if direct_success else '❌ Failed'}")
    print(f"Pipeline TTS endpoint: {'✅ Success' if pipeline_success else '❌ Failed'}")
    
    # Return overall success
    return direct_success and pipeline_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)