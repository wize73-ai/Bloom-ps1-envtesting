#!/usr/bin/env python3
"""
Comprehensive test for all documented API endpoints in CasaLingua.
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

# Comprehensive list of all documented endpoints
ENDPOINTS = [
    # Health endpoints
    {"method": "GET", "url": "/health", "category": "Health"},
    {"method": "GET", "url": "/health/detailed", "category": "Health"},
    {"method": "GET", "url": "/health/models", "category": "Health"},
    {"method": "GET", "url": "/health/database", "category": "Health"},
    {"method": "GET", "url": "/readiness", "category": "Health"},
    {"method": "GET", "url": "/liveness", "category": "Health"},
    
    # Pipeline endpoints - Core functionality
    {"method": "POST", "url": "/pipeline/translate", "data": "translate", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/detect", "data": "detect", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/detect-language", "data": "detect", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/simplify", "data": "simplify", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/summarize", "data": "summarize", "category": "Pipeline"},
    {"method": "POST", "url": "/pipeline/anonymize", "data": "anonymize", "category": "Pipeline"},
    
    # Document processing endpoints
    {"method": "POST", "url": "/document/process", "category": "Document", "skip": True},  # Needs file upload
    {"method": "POST", "url": "/document/extract", "category": "Document", "skip": True},  # Needs file upload
    
    # Bloom housing specific endpoints
    {"method": "POST", "url": "/bloom/translate", "data": "translate", "category": "Bloom"},
    {"method": "POST", "url": "/bloom/simplify", "data": "simplify", "category": "Bloom"},
]

def test_endpoint(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single endpoint and return the results."""
    method = endpoint["method"]
    url = endpoint["url"]
    full_url = f"{BASE_URL}{url}"
    headers = {"Content-Type": "application/json"}
    
    # Skip endpoints marked for skipping
    if endpoint.get("skip", False):
        return {
            "endpoint": f"{method} {url}",
            "category": endpoint.get("category", "Unknown"),
            "status": "SKIPPED",
            "message": "Endpoint skipped as requested",
            "status_code": 0,
            "time_ms": 0
        }
    
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
        elif method == "PUT":
            response = requests.put(full_url, json=data, headers=headers, timeout=5)
        elif method == "DELETE":
            response = requests.delete(full_url, headers=headers, timeout=5)
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

def save_results_to_file(results: List[Dict[str, Any]], filename: str = "endpoint_test_results.json"):
    """Save test results to a JSON file."""
    # Count results by status
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    skipped_count = sum(1 for r in results if r["status"] == "SKIPPED")
    
    total_count = len(results)
    tested_count = total_count - skipped_count
    
    # Create summary by category
    categories = {}
    for result in results:
        category = result.get("category", "Unknown")
        if category not in categories:
            categories[category] = {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "error": 0,
                "skipped": 0
            }
        
        categories[category]["total"] += 1
        
        if result["status"] == "PASS":
            categories[category]["pass"] += 1
        elif result["status"] == "FAIL":
            categories[category]["fail"] += 1
        elif result["status"] == "ERROR":
            categories[category]["error"] += 1
        elif result["status"] == "SKIPPED":
            categories[category]["skipped"] += 1
    
    # Create report data
    report = {
        "summary": {
            "total": total_count,
            "tested": tested_count,
            "pass": pass_count,
            "fail": fail_count,
            "error": error_count,
            "skipped": skipped_count,
            "pass_percentage": round((pass_count / tested_count * 100), 1) if tested_count > 0 else 0
        },
        "categories": categories,
        "results": results,
        "timestamp": time.time(),
        "server": BASE_URL
    }
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Main function to test all endpoints."""
    print(f"Testing {len(ENDPOINTS)} endpoints against {BASE_URL}")
    print("Please make sure the server is running.\n")
    
    results = []
    
    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_endpoint = {executor.submit(test_endpoint, endpoint): endpoint for endpoint in ENDPOINTS}
        
        for future in as_completed(future_to_endpoint):
            endpoint = future_to_endpoint[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print result as we go
                status_symbol = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "!" if result["status"] == "ERROR" else "○"
                print(f"{status_symbol} {endpoint['method']} {endpoint['url']} - {result['status']} ({result['status_code']}) in {result['time_ms']}ms")
            except Exception as e:
                print(f"! {endpoint['method']} {endpoint['url']} - Error: {str(e)}")
    
    # Count results by status
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    skipped_count = sum(1 for r in results if r["status"] == "SKIPPED")
    
    total_count = len(results)
    tested_count = total_count - skipped_count
    
    # Print summary
    print("\n----- SUMMARY -----")
    print(f"Total endpoints: {total_count}")
    print(f"Tested endpoints: {tested_count}")
    print(f"PASS: {pass_count} ({pass_count/tested_count*100:.1f}%)")
    print(f"FAIL: {fail_count} ({fail_count/tested_count*100:.1f}%)")
    print(f"ERROR: {error_count} ({error_count/tested_count*100:.1f}%)")
    print(f"SKIPPED: {skipped_count}")
    
    # Save results to file
    report = save_results_to_file(results)
    print(f"\nDetailed results written to endpoint_test_results.json")
    
    # Print category summary
    print("\n----- CATEGORY SUMMARY -----")
    for category, stats in report["categories"].items():
        tested = stats["total"] - stats["skipped"]
        pass_pct = round((stats["pass"] / tested * 100), 1) if tested > 0 else 0
        print(f"{category}: {stats['pass']}/{tested} passed ({pass_pct}%)")
    
    return 0 if fail_count + error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())