#!/usr/bin/env python3
"""
Simple test for the translation endpoint
"""

import requests
import json

API_URL = "http://localhost:8000"

def main():
    # Test parameters
    text = "Hello world, this is a test."
    source_language = "en"
    target_language = "es"
    
    # API request
    endpoint = f"{API_URL}/pipeline/translate"
    payload = {
        "text": text,
        "source_language": source_language,
        "target_language": target_language
    }
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending request to {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Print key information
            if "data" in result:
                print("\nTranslation Result:")
                print(f"Source text: {result['data']['source_text']}")
                print(f"Translated text: {result['data']['translated_text']}")
                print(f"Model used: {result['data']['model_used']}")
                
                # Check if actual translation happened
                if result['data']['source_text'] == result['data']['translated_text']:
                    print("\nWARNING: Source and translated text are identical!")
            else:
                print("No data in response")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    main()