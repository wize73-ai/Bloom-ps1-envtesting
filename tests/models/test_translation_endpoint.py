#!/usr/bin/env python3
"""
Test script for the translation endpoint in CasaLingua
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Configuration
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

def test_translation_endpoint():
    """
    Test the translation endpoint with various language pairs
    """
    print("Testing translation endpoint...")
    
    # Test cases with different language pairs
    test_cases = [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "source_language": "en",
            "target_language": "es",
            "description": "English to Spanish"
        },
        {
            "text": "Hello world, this is a test of the translation system.",
            "source_language": "en",
            "target_language": "fr",
            "description": "English to French"
        },
        {
            "text": "Machine learning models are improving every day.",
            "source_language": "en",
            "target_language": "de",
            "description": "English to German"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case['description']}")
        print(f"Text: {test_case['text']}")
        
        # Call the translation endpoint
        response = call_translation_api(
            text=test_case['text'],
            source_language=test_case['source_language'],
            target_language=test_case['target_language']
        )
        
        # Print results
        if "status" in response and response["status"] == "success" and "data" in response:
            print(f"✅ Success: {response['data']['translated_text']}")
            print(f"Model used: {response['data'].get('model_used', 'unknown')}")
            print(f"Time taken: {response['data'].get('process_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
            print(f"Details: {json.dumps(response, indent=2)}")
    
    print("\nTranslation testing complete.")

def call_translation_api(text: str, source_language: str, target_language: str) -> Dict[str, Any]:
    """
    Call the translation API endpoint
    
    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language code
        
    Returns:
        API response as dictionary
    """
    endpoint = f"{API_URL}/pipeline/translate"
    
    payload = {
        "text": text,
        "source_language": source_language,
        "target_language": target_language
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        start_time = time.time()
        response = requests.post(endpoint, json=payload, headers=headers)
        response_time = time.time() - start_time
        
        # Check if response is successful
        if response.status_code == 200:
            result = response.json()
            # Add processing time to metadata
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["processing_time"] = response_time
            return result
        else:
            return {
                "error": f"API Error: {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "error": f"Request Error: {str(e)}"
        }

if __name__ == "__main__":
    test_translation_endpoint()