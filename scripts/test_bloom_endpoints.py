#!/usr/bin/env python3
"""
Test script for Bloom Housing endpoints

These endpoints use the prefix /bloom-housing/ instead of /bloom/ which was previously tested.
"""
import requests
import json
import time
import uuid
from typing import Dict, Any

# Base URL
BASE_URL = "http://localhost:8000"

def test_endpoint(method: str, path: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test a specific endpoint and return the result."""
    full_url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    request_id = str(uuid.uuid4())
    
    print(f"Testing {method} {path}...")
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(full_url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(full_url, json=data, headers=headers, timeout=5)
        else:
            return {
                "endpoint": f"{method} {path}",
                "status": "ERROR",
                "message": f"Unsupported method: {method}",
                "time_ms": 0
            }
        
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000, 2)
        
        # Check if successful
        status = "PASS" if 200 <= response.status_code < 300 else "FAIL"
        
        try:
            response_body = response.json()
        except json.JSONDecodeError:
            response_body = response.text[:200] + "..." if len(response.text) > 200 else response.text
        
        return {
            "endpoint": f"{method} {path}",
            "status": status,
            "status_code": response.status_code,
            "time_ms": time_ms,
            "response": response_body
        }
    except Exception as e:
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000, 2)
        
        return {
            "endpoint": f"{method} {path}",
            "status": "ERROR",
            "message": str(e),
            "time_ms": time_ms
        }

def main():
    """Main function to test Bloom housing endpoints."""
    # Test data
    translate_data = {
        "text": "Hello, how are you?",
        "sourceLanguage": "en",
        "targetLanguage": "es",
        "preserveFormatting": True
    }
    
    language_detection_data = {
        "text": "Hello, how are you?",
        "detailed": True
    }
    
    analysis_data = {
        "text": "This text needs to be analyzed for sentiment and complexity.",
        "language": "en",
        "analyses": ["sentiment", "complexity"]
    }
    
    # Test endpoints
    results = []
    
    # /bloom-housing/translate endpoint
    results.append(test_endpoint("POST", "/bloom-housing/translate", translate_data))
    
    # /bloom-housing/detect-language endpoint
    results.append(test_endpoint("POST", "/bloom-housing/detect-language", language_detection_data))
    
    # /bloom-housing/analyze endpoint
    results.append(test_endpoint("POST", "/bloom-housing/analyze", analysis_data))
    
    # Print results
    print("\n----- RESULTS -----")
    for result in results:
        status_symbol = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "!"
        print(f"{status_symbol} {result['endpoint']} - {result['status']} ({result.get('status_code', 'N/A')}) in {result['time_ms']}ms")
    
    # Print summary
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    print("\n----- SUMMARY -----")
    print(f"Total endpoints tested: {len(results)}")
    print(f"PASS: {pass_count} ({pass_count/len(results)*100:.1f}%)")
    print(f"FAIL: {fail_count} ({fail_count/len(results)*100:.1f}%)")
    print(f"ERROR: {error_count} ({error_count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()