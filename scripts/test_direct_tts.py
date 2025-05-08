#\!/usr/bin/env python3
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
        print(f"\nTesting case: {case['name']}")
        
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
    
    print("\n===== TEST COMPLETE =====")
    print("Test files have been saved to temp/direct_tts_test directory.")
    print("You can play these files to verify they contain audible speech.")

if __name__ == "__main__":
    test_direct_tts_endpoint()
