#!/usr/bin/env python3
"""
CasaLingua Single Test

This is the simplest possible test for the CasaLingua server.
It just makes a single translation request and reports the result.
"""

import sys
import requests
import json

API_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Response: Success")
            # Try to decode json
            try:
                data = response.json()
                print(f"API status: {data.get('status', 'Unknown')}")
                return True
            except:
                print("Could not parse JSON response")
                print(f"Raw response: {response.text[:100]}...")
        else:
            print(f"Failed with status {response.status_code}")
            print(f"Response: {response.text[:100]}...")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_single_translation():
    """Make a single translation request"""
    print("\nTesting translation endpoint...")
    
    payload = {
        "text": "The quick brown fox jumps over the lazy dog.",
        "source_language": "en",
        "target_language": "es"
    }
    
    try:
        print(f"Request data: {json.dumps(payload)}")
        response = requests.post(
            f"{API_URL}/pipeline/translate",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        print(f"Status code: {response.status_code}")
        
        try:
            result = response.json()
            print("Response JSON parsed successfully")
            
            # Print out the full response for debugging
            print("\nFull response:")
            print(json.dumps(result, indent=2))
            
            if response.status_code == 200:
                data = result.get("data", {})
                if "translated_text" in data:
                    print(f"\nTranslation successful:")
                    print(f"Source: {payload['text']}")
                    print(f"Translation: {data['translated_text']}")
                    print(f"Model used: {data.get('model_used', 'Unknown')}")
                    return True
                else:
                    print("No translation in response")
            return False
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw response: {response.text[:500]}...")
            return False
    except Exception as e:
        print(f"Request error: {str(e)}")
        return False

def main():
    """Main function"""
    print("==== CasaLingua Single Test ====")
    
    # Test health first
    health_ok = test_health()
    if not health_ok:
        print("\nWARNING: Health check failed, but continuing anyway")
    
    # Test a single translation
    translation_ok = test_single_translation()
    
    # Print summary
    print("\n==== Test Summary ====")
    print(f"Health check: {'OK' if health_ok else 'FAILED'}")
    print(f"Translation test: {'OK' if translation_ok else 'FAILED'}")
    
    if health_ok and translation_ok:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED.")
        return 1

if __name__ == "__main__":
    sys.exit(main())