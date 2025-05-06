#!/usr/bin/env python3
import requests
import json
import time
import sys
from datetime import datetime

# Configure test parameters
API_BASE_URL = "http://localhost:8000"

def test_translate_endpoint():
    """Test the translation endpoint."""
    endpoint = f"{API_BASE_URL}/pipeline/translate"
    
    # Test data
    data = {
        "text": "Hello, how are you today?",
        "source_language": "en",
        "target_language": "es"
    }
    
    # Make the request
    print(f"Testing {endpoint} with data: {data}")
    try:
        response = requests.post(endpoint, json=data)
        
        # Print response details
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        # Parse and validate response
        if response.status_code == 200:
            result = response.json()
            if "data" in result and "translated_text" in result["data"]:
                translated_text = result["data"]["translated_text"]
                print(f"✅ Translation successful: '{translated_text}'")
                return True
            else:
                print("❌ Response did not contain expected translation data format")
                return False
        else:
            print(f"❌ Request failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during request: {str(e)}")
        return False

# Main test runner
if __name__ == "__main__":
    print("Testing translation endpoint...")
    
    # Check if server is running
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code == 200:
            print(f"✅ Connected to server at {API_BASE_URL}")
        else:
            print(f"❌ Server at {API_BASE_URL} returned status {health_response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to connect to server at {API_BASE_URL}: {str(e)}")
        print("Please make sure the server is running.")
        sys.exit(1)
    
    # Run the test
    success = test_translate_endpoint()
    
    # Summarize results
    if success:
        print("\n=== Summary ===")
        print("✅ Translation endpoint test passed!")
        sys.exit(0)
    else:
        print("\n=== Summary ===")
        print("❌ Translation endpoint test failed!")
        sys.exit(1)