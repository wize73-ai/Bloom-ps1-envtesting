#!/usr/bin/env python3
"""
Focused test for pipeline endpoints which are working well.
"""
import sys
import json
import time
import requests
from typing import Dict, Any, Optional

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test data for different endpoint types
TEST_DATA = {
    "translate": {
        "text": "Hello, how are you?",
        "source_language": "en",
        "target_language": "es"
    },
    "detect": {
        "text": "Hello, how are you?"
    },
    "simplify": {
        "text": "The patient presents with cardiomyopathy and requires immediate medical intervention.",
        "level": "medium"
    },
    "summarize": {
        "text": "This is a long text that needs to be summarized. It contains multiple sentences and paragraphs. The summary should capture the main points.",
        "max_length": 50
    },
    "anonymize": {
        "text": "John Smith lives at 123 Main St. His phone number is 555-123-4567."
    }
}

# Pipeline endpoints only
ENDPOINTS = [
    # Pipeline endpoints - Core functionality
    {"method": "POST", "url": "/pipeline/translate", "data": "translate", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/detect", "data": "detect", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/detect-language", "data": "detect", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/simplify", "data": "simplify", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/summarize", "data": "summarize", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/anonymize", "data": "anonymize", "category": "Pipeline"}
]

def test_endpoint(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single endpoint and return the results."""
    method = endpoint["method"]
    url = endpoint["url"]
    full_url = f"{BASE_URL}{url}"
    headers = {"Content-Type": "application/json"}
    
    # Get test data if specified
    data = None
    if "data" in endpoint and endpoint["data"] in TEST_DATA:
        data = TEST_DATA[endpoint["data"]]
    
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(full_url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(full_url, json=data, headers=headers, timeout=5)
        else:
            return {
                "endpoint": f"{method} {url}",
                "category": endpoint.get("category", "Unknown"),
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
            "category": endpoint.get("category", "Unknown"),
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
            "category": endpoint.get("category", "Unknown"),
            "status": "ERROR",
            "message": str(e),
            "status_code": 0,
            "time_ms": time_ms
        }

def main():
    """Main function to test pipeline endpoints."""
    print(f"Testing {len(ENDPOINTS)} pipeline endpoints against {BASE_URL}")
    print("Please make sure the server is running.\n")
    
    results = []
    
    # Test each endpoint sequentially
    for endpoint in ENDPOINTS:
        # Print what we're about to test
        print(f"Testing {endpoint['method']} {endpoint['url']}...")
        
        # Test the endpoint
        result = test_endpoint(endpoint)
        results.append(result)
        
        # Print result
        status_symbol = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "!"
        print(f"{status_symbol} {result['endpoint']} - {result['status']} ({result['status_code']}) in {result['time_ms']}ms")
        
        # Add a small delay between requests
        time.sleep(0.5)
    
    # Count results by status
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    total_count = len(results)
    
    # Print summary
    print("\n----- SUMMARY -----")
    print(f"Total endpoints: {total_count}")
    print(f"PASS: {pass_count} ({pass_count/total_count*100:.1f}%)")
    print(f"FAIL: {fail_count} ({fail_count/total_count*100:.1f}%)")
    print(f"ERROR: {error_count} ({error_count/total_count*100:.1f}%)")
    
    # Print category summary
    categories = {}
    for result in results:
        category = result.get("category", "Unknown")
        if category not in categories:
            categories[category] = {"total": 0, "pass": 0}
        
        categories[category]["total"] += 1
        if result["status"] == "PASS":
            categories[category]["pass"] += 1
    
    print("\n----- CATEGORY SUMMARY -----")
    for category, stats in categories.items():
        pass_pct = round((stats["pass"] / stats["total"] * 100), 1)
        print(f"{category}: {stats['pass']}/{stats['total']} passed ({pass_pct}%)")
    
    return 0 if fail_count + error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())