#!/usr/bin/env python3
"""
Test specific endpoints to verify they're working properly after commenting out problematic ones.
"""
import sys
import json
import time
import requests
from typing import Dict, Any, Optional

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Endpoints to test
ENDPOINTS = [
    # Healthy endpoints
    {"method": "POST", "url": "/pipeline/anonymize", "data": {"text": "John Smith lives at 123 Main St."}},
    {"method": "POST", "url": "/pipeline/summarize", "data": {"text": "This is a test paragraph that needs to be summarized."}},
    
    # Pipeline endpoints 
    {"method": "POST", "url": "/pipeline/translate", "data": {"text": "Hello world", "source_language": "en", "target_language": "es"}},
    {"method": "POST", "url": "/pipeline/detect", "data": {"text": "Hello world"}},
    {"method": "POST", "url": "/pipeline/simplify", "data": {"text": "The patient presents with cardiomyopathy.", "level": "medium"}},
    
    # Health endpoints
    {"method": "GET", "url": "/health"},
    {"method": "GET", "url": "/health/detailed"},
    {"method": "GET", "url": "/readiness"},
    {"method": "GET", "url": "/liveness"},
]

def test_endpoint(method: str, url: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test a single endpoint and return the results."""
    full_url = f"{BASE_URL}{url}"
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(full_url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(full_url, json=data, headers=headers, timeout=5)
        else:
            return {
                "endpoint": f"{method} {url}",
                "status": "ERROR",
                "message": f"Unsupported method: {method}",
                "status_code": 0,
                "time_ms": 0
            }
        
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000, 2)
        
        # Check if the request was successful (status code 2xx)
        status = "PASS" if 200 <= response.status_code < 300 else "FAIL"
        
        # Try to extract the response body (might not be JSON)
        try:
            response_body = response.json()
        except json.JSONDecodeError:
            response_body = response.text[:200] + "..." if len(response.text) > 200 else response.text
        
        return {
            "endpoint": f"{method} {url}",
            "status": status,
            "status_code": response.status_code,
            "time_ms": time_ms,
            "response_sample": response_body
        }
    except Exception as e:
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000, 2)
        
        return {
            "endpoint": f"{method} {url}",
            "status": "ERROR",
            "message": str(e),
            "status_code": 0,
            "time_ms": time_ms
        }

def main():
    """Main function to test endpoints."""
    print(f"Testing {len(ENDPOINTS)} endpoints against {BASE_URL}")
    print("Please make sure the server is running.\n")
    
    results = []
    
    for endpoint in ENDPOINTS:
        result = test_endpoint(endpoint["method"], endpoint["url"], endpoint.get("data"))
        results.append(result)
        
        # Print result
        status_symbol = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "!"
        print(f"{status_symbol} {endpoint['method']} {endpoint['url']} - {result['status']} ({result['status_code']}) in {result['time_ms']}ms")
    
    # Count results by status
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    # Print summary
    print("\n----- SUMMARY -----")
    print(f"Total endpoints tested: {len(results)}")
    print(f"PASS: {pass_count} ({pass_count/len(results)*100:.1f}%)")
    print(f"FAIL: {fail_count} ({fail_count/len(results)*100:.1f}%)")
    print(f"ERROR: {error_count} ({error_count/len(results)*100:.1f}%)")
    
    return 0 if error_count + fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())