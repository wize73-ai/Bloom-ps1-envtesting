#!/usr/bin/env python3
"""
Test script for all TTS endpoints.
This script tests both the regular pipeline TTS and direct TTS endpoints
to ensure they're generating audible audio.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

def test_pipeline_tts_endpoint():
    """Test the pipeline TTS endpoint."""
    # Create output directory
    os.makedirs("temp/tts_test_results", exist_ok=True)
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "name": "pipeline_english",
            "text": "This is a test message for the pipeline text to speech system.",
            "language": "en",
            "output_format": "mp3"
        },
        {
            "name": "pipeline_spanish",
            "text": "Este es un mensaje de prueba para el sistema de texto a voz de pipeline.",
            "language": "es",
            "output_format": "mp3"
        }
    ]
    
    print("\n===== PIPELINE TTS ENDPOINT TEST =====")
    
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
        output_file = f"temp/tts_test_results/{case['name']}.{case['output_format']}"
        
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
                
                print(f"✅ Success! Output saved to {output_file}")
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

def test_direct_tts_endpoint():
    """Test the direct TTS endpoint."""
    # Create output directory
    os.makedirs("temp/tts_test_results", exist_ok=True)
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test cases
    test_cases = [
        {
            "name": "direct_english",
            "text": "This is a test message for the direct text to speech system.",
            "language": "en",
            "output_format": "mp3"
        },
        {
            "name": "direct_spanish",
            "text": "Este es un mensaje de prueba para el sistema directo de texto a voz.",
            "language": "es",
            "output_format": "mp3"
        },
        {
            "name": "direct_french",
            "text": "Ceci est un test pour le système de synthèse vocale.",
            "language": "fr",
            "output_format": "mp3"
        }
    ]
    
    print("\n===== DIRECT TTS ENDPOINT TEST =====")
    
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
        output_file = f"temp/tts_test_results/{case['name']}.{case['output_format']}"
        
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
                
                print(f"✅ Success! Output saved to {output_file}")
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

def main():
    """Run all TTS endpoint tests."""
    print("🎤 Testing all TTS endpoints")
    
    # Test pipeline TTS endpoint
    test_pipeline_tts_endpoint()
    
    # Test direct TTS endpoint
    test_direct_tts_endpoint()
    
    print("\n===== ALL TTS TESTS COMPLETE =====")
    print("Test files have been saved to temp/tts_test_results directory.")
    print("You can play these files to verify they contain audible speech.")
    print("If you hear audio in these files, TTS functionality is working correctly!")

if __name__ == "__main__":
    main()