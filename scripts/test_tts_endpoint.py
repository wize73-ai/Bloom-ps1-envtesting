#\!/usr/bin/env python3
"""
Test script for TTS endpoint.
This script tests the TTS endpoint to make sure it's generating audible audio.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

def test_tts_endpoint():
    """Test the TTS endpoint with different parameters."""
    # Create output directory
    os.makedirs("temp/test_results", exist_ok=True)
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "name": "english_short",
            "text": "This is a test message for the text to speech system.",
            "language": "en",
            "output_format": "mp3"
        },
        {
            "name": "spanish_short",
            "text": "Este es un mensaje de prueba para el sistema de texto a voz.",
            "language": "es",
            "output_format": "mp3"
        },
        {
            "name": "english_with_voice",
            "text": "This is a test with a specific voice parameter.",
            "language": "en",
            "voice": "en-us-1",
            "output_format": "mp3"
        }
    ]
    
    print("===== TTS ENDPOINT TEST =====")
    
    # Test each case
    for case in test_cases:
        print(f"\nTesting case: {case['name']}")
        
        # Create request data
        request_data = {
            "text": case["text"],
            "language": case["language"],
            "output_format": case["output_format"]
        }
        
        # Add optional parameters
        if "voice" in case:
            request_data["voice"] = case["voice"]
        
        # Output file
        output_file = f"temp/test_results/tts_{case['name']}.{case['output_format']}"
        
        # Make request
        print(f"Making request to /pipeline/tts with text: '{request_data['text'][:30]}...'")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/pipeline/tts",
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
                if file_size < 2000:
                    print(f"⚠️  WARNING: File size is very small ({file_size} bytes)")
                    print("   This might be a silent or corrupted audio file.")
                
                # Suggest command to play the file
                print(f"   To play: open {output_file}")
            else:
                print(f"❌ Error: Status code {response.status_code}")
                print(f"   Response: {response.text}")
        
        except Exception as e:
            print(f"❌ Error making request: {str(e)}")
    
    print("\n===== TEST COMPLETE =====")
    print("Test files have been saved to temp/test_results directory.")
    print("You can play these files to verify they contain audible speech.")
    print("If you hear audio in these files, the fix was successful\!")

if __name__ == "__main__":
    test_tts_endpoint()
