#!/usr/bin/env python3
"""
Endpoint testing script that checks all API endpoints and reports their status.
"""
import os
import sys
import json
import time
import requests
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Sample test data for different endpoint types
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
    },
    "chat": {
        "message": "What can you tell me about language translation?",
        "conversation_id": "test-conv-123"
    },
    "document_upload": {
        # This would require multipart/form-data handling for actual file upload
        "test_only": True
    }
}

# Endpoints to test
# Format: (method, path, test_data_key or None, expected_status_code)
ENDPOINTS: List[Tuple[str, str, Optional[str], int]] = [
    # Health endpoints
    ("GET", "/health", None, 200),
    ("GET", "/health/detailed", None, 200),
    ("GET", "/health/models", None, 200),
    ("GET", "/health/database", None, 200),
    ("GET", "/readiness", None, 200),
    ("GET", "/liveness", None, 200),
    
    # Pipeline endpoints
    ("POST", "/pipeline/translate", "translate", 200),
    ("POST", "/pipeline/detect", "detect", 200),
    ("POST", "/pipeline/detect-language", "detect", 200),
    ("POST", "/pipeline/simplify", "simplify", 200),
    ("POST", "/pipeline/summarize", "summarize", 200),
    ("POST", "/pipeline/anonymize", "anonymize", 200),
    
    # RAG endpoints
    ("POST", "/translate", "translate", 200),
    ("POST", "/query", "chat", 200),
    ("POST", "/chat", "chat", 200),
    
    # Metrics endpoints
    ("GET", "/metrics", None, 200),
    
    # Admin endpoints
    ("GET", "/system/info", None, 200),
    ("GET", "/models", None, 200),
    ("GET", "/languages", None, 200),
]

def test_endpoint(method: str, path: str, data_key: Optional[str], expected_status: int) -> Dict[str, Any]:
    """Test a single endpoint and return the results."""
    url = f"{BASE_URL}{path}"
    data = None if data_key is None else TEST_DATA.get(data_key, {})
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        else:
            return {
                "endpoint": f"{method} {path}",
                "status": "ERROR",
                "message": f"Unsupported method: {method}",
                "status_code": 0,
                "time_ms": 0
            }
        
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000, 2)
        
        # Check if the status code matches the expected code
        status = "PASS" if response.status_code == expected_status else "FAIL"
        
        # Try to extract the response body (might not be JSON)
        try:
            response_body = response.json()
        except json.JSONDecodeError:
            response_body = response.text[:200] + "..." if len(response.text) > 200 else response.text
        
        return {
            "endpoint": f"{method} {path}",
            "status": status,
            "status_code": response.status_code,
            "expected_status_code": expected_status,
            "time_ms": time_ms,
            "response_sample": response_body
        }
    except Exception as e:
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000, 2)
        
        return {
            "endpoint": f"{method} {path}",
            "status": "ERROR",
            "message": str(e),
            "status_code": 0,
            "time_ms": time_ms
        }

def main():
    """Main function to test all endpoints and generate a report."""
    print(f"Testing {len(ENDPOINTS)} endpoints against {BASE_URL}...\n")
    
    results = []
    
    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_endpoint = {
            executor.submit(test_endpoint, method, path, data_key, expected_status): (method, path)
            for method, path, data_key, expected_status in ENDPOINTS
        }
        
        for future in as_completed(future_to_endpoint):
            method, path = future_to_endpoint[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print result as we go
                status_symbol = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "!"
                print(f"{status_symbol} {method} {path} - {result['status']} ({result['status_code']}) in {result['time_ms']}ms")
            except Exception as e:
                print(f"! {method} {path} - Error: {str(e)}")
    
    # Sort results by status (PASS, FAIL, ERROR)
    results.sort(key=lambda x: 0 if x["status"] == "PASS" else 1 if x["status"] == "FAIL" else 2)
    
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
    
    # Write results to a file
    with open("endpoint_test_results.json", "w") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "pass": pass_count,
                "fail": fail_count,
                "error": error_count,
                "pass_percentage": pass_count/len(results)*100
            },
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results written to endpoint_test_results.json")
    
    # Generate a file with commented out endpoints that failed
    generate_endpoint_fixes(results)
    
    return 0 if error_count + fail_count == 0 else 1

def generate_endpoint_fixes(results):
    """Generate a file with endpoints that need fixing."""
    failing_endpoints = [r for r in results if r["status"] in ["FAIL", "ERROR"]]
    
    if not failing_endpoints:
        print("All endpoints are working correctly!")
        return
    
    with open("endpoint_fixes.py", "w") as f:
        f.write("# Endpoints that need to be fixed or commented out\n\n")
        f.write("# Add this code to your route files or use it as a reference\n\n")
        
        for result in failing_endpoints:
            endpoint = result["endpoint"]
            parts = endpoint.split(" ")
            method = parts[0].lower()
            path = parts[1]
            
            f.write(f"# FAILING ENDPOINT: {endpoint}\n")
            f.write(f"# Status: {result['status']}, Code: {result['status_code']}, Expected: {result.get('expected_status_code', 'N/A')}\n")
            f.write(f"# Error: {result.get('message', 'No specific error message')}\n")
            f.write(f"'''\n")
            f.write(f"@router.{method}(\"{path}\")\n")
            f.write(f"async def endpoint_function(request_data: SomeRequestModel):\n")
            f.write(f"    # Implementation here\n")
            f.write(f"    pass\n")
            f.write(f"'''\n\n")
    
    print(f"\nGenerated endpoint_fixes.py with {len(failing_endpoints)} endpoints that need fixing")

if __name__ == "__main__":
    sys.exit(main())