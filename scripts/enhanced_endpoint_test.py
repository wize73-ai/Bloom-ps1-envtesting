#!/usr/bin/env python3
"""
Enhanced endpoint testing script with veracity and metrics monitoring.
"""
import os
import sys
import json
import time
import uuid
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Log file path
LOG_FILE = "/Users/jameswilson/Desktop/PRODUCTION/test/casMay4/logs/server.log"
METRICS_LOG = "/Users/jameswilson/Desktop/PRODUCTION/test/casMay4/logs/metrics.log"
VERACITY_LOG = "/Users/jameswilson/Desktop/PRODUCTION/test/casMay4/logs/veracity.log"

# Test data for different endpoint types
TEST_DATA = {
    "translate": {
        "text": "Hello, how are you? My name is John and I live at 123 Main St. I'm 42 years old and have $5000 in my account.",
        "source_language": "en",
        "target_language": "es"
    },
    "translate_with_numbers": {
        "text": "The contract was signed on January 15, 2025. The total amount is $4,250.75 for 3 services.",
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
        "text": "This is a long text that needs to be summarized. It contains multiple sentences and paragraphs. The summary should capture the main points about climate change and its effects on global agriculture. Farmers around the world are experiencing challenges due to changing weather patterns.",
        "max_length": 50
    },
    "anonymize": {
        "text": "John Smith lives at 123 Main St. His phone number is 555-123-4567 and his email is john.smith@example.com. His social security number is 123-45-6789."
    }
}

# Comprehensive list of all documented endpoints
ENDPOINTS = [
    # Health endpoints
    {"method": "GET", "url": "/health", "category": "Health", "check_metrics": True},
    {"method": "GET", "url": "/health/detailed", "category": "Health", "check_metrics": True},
    {"method": "GET", "url": "/health/models", "category": "Health", "check_metrics": True},
    {"method": "GET", "url": "/readiness", "category": "Health", "check_metrics": True},
    {"method": "GET", "url": "/liveness", "category": "Health", "check_metrics": True},
    
    # Pipeline endpoints - Core functionality
    {"method": "POST", "url": "/pipeline/translate", "data": "translate", "category": "Pipeline", "check_veracity": True, "check_metrics": True},
    {"method": "POST", "url": "/pipeline/translate", "data": "translate_with_numbers", "category": "Pipeline", "name": "translate_with_numbers", "check_veracity": True, "check_metrics": True},
    {"method": "POST", "url": "/pipeline/detect", "data": "detect", "category": "Pipeline", "check_metrics": True},
    {"method": "POST", "url": "/pipeline/detect-language", "data": "detect", "category": "Pipeline", "check_metrics": True},
    {"method": "POST", "url": "/pipeline/simplify", "data": "simplify", "category": "Pipeline", "check_veracity": True, "check_metrics": True},
    {"method": "POST", "url": "/pipeline/summarize", "data": "summarize", "category": "Pipeline", "check_veracity": True, "check_metrics": True},
    {"method": "POST", "url": "/pipeline/anonymize", "data": "anonymize", "category": "Pipeline", "check_metrics": True},
    
    # Bloom housing specific endpoints
    {"method": "POST", "url": "/bloom/translate", "data": "translate", "category": "Bloom", "check_veracity": True, "check_metrics": True},
    {"method": "POST", "url": "/bloom/simplify", "data": "simplify", "category": "Bloom", "check_veracity": True, "check_metrics": True},
]

def ensure_log_directories():
    """Ensure log directories exist."""
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def tail_log(log_file, num_lines=10):
    """Tail the log file to see recent entries."""
    if not os.path.exists(log_file):
        return "Log file does not exist"
        
    try:
        result = subprocess.run(['tail', '-n', str(num_lines), log_file], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except Exception as e:
        return f"Error tailing log: {str(e)}"

def generate_request_id():
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())

def check_logs_for_request(request_id, log_file=LOG_FILE):
    """Check logs for entries related to the request ID."""
    if not os.path.exists(log_file):
        return False, "Log file does not exist"
        
    try:
        # Use grep to find the request ID in the log file
        result = subprocess.run(['grep', request_id, log_file], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        
        entries = result.stdout.decode('utf-8').strip().split('\n')
        entries = [e for e in entries if e]  # Remove empty entries
        
        if entries:
            return True, entries
        else:
            return False, "No log entries found for the request ID"
    except Exception as e:
        return False, f"Error checking logs: {str(e)}"

def check_metrics_for_endpoint(endpoint_path, minutes=5):
    """Check if metrics are being recorded for an endpoint."""
    # Replace / with _ in endpoint path for metrics naming
    endpoint_key = endpoint_path.replace('/', '_').lstrip('_')
    
    # Use grep to search for metrics related to this endpoint
    try:
        since = int(time.time() - (minutes * 60))
        
        result = subprocess.run(['grep', f'"endpoint":"{endpoint_path}"', METRICS_LOG], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        
        entries = result.stdout.decode('utf-8').strip().split('\n')
        entries = [e for e in entries if e]  # Remove empty entries
        
        if entries:
            # Try to parse the entries to check for metrics data
            metrics_data = []
            for entry in entries:
                try:
                    data = json.loads(entry)
                    if 'timestamp' in data and data['timestamp'] > since:
                        metrics_data.append(data)
                except:
                    continue
            
            if metrics_data:
                return True, metrics_data
            else:
                return False, "No recent metrics data found"
        else:
            return False, "No metrics entries found for the endpoint"
    except Exception as e:
        return False, f"Error checking metrics: {str(e)}"

def check_veracity_for_request(request_id):
    """Check if veracity checks were performed for a request."""
    try:
        result = subprocess.run(['grep', request_id, VERACITY_LOG], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        
        entries = result.stdout.decode('utf-8').strip().split('\n')
        entries = [e for e in entries if e]  # Remove empty entries
        
        if entries:
            # Try to parse the entries to check for veracity data
            veracity_data = []
            for entry in entries:
                try:
                    data = json.loads(entry)
                    veracity_data.append(data)
                except:
                    continue
            
            if veracity_data:
                return True, veracity_data
            else:
                return False, "No veracity data found"
        else:
            return False, "No veracity entries found for the request ID"
    except Exception as e:
        return False, f"Error checking veracity: {str(e)}"

def test_endpoint(endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single endpoint and return the results with monitoring data."""
    method = endpoint["method"]
    url = endpoint["url"]
    endpoint_name = endpoint.get("name", url.replace("/", "_").lstrip("_"))
    full_url = f"{BASE_URL}{url}"
    
    # Generate a unique request ID for tracing
    request_id = generate_request_id()
    
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request_id
    }
    
    # Get test data if specified
    data = None
    if "data" in endpoint and endpoint["data"] in TEST_DATA:
        data = TEST_DATA[endpoint["data"]]
    
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(full_url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(full_url, json=data, headers=headers, timeout=10)
        elif method == "PUT":
            response = requests.put(full_url, json=data, headers=headers, timeout=10)
        elif method == "DELETE":
            response = requests.delete(full_url, headers=headers, timeout=10)
        else:
            return {
                "endpoint": f"{method} {url}",
                "endpoint_name": endpoint_name,
                "category": endpoint.get("category", "Unknown"),
                "status": "ERROR",
                "message": f"Unsupported method: {method}",
                "status_code": 0,
                "request_id": request_id,
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
            
        # Basic result
        result = {
            "endpoint": f"{method} {url}",
            "endpoint_name": endpoint_name,
            "category": endpoint.get("category", "Unknown"),
            "status": status,
            "status_code": response.status_code,
            "request_id": request_id,
            "time_ms": time_ms,
            "response_sample": response_body
        }
        
        # Check logs if requested
        time.sleep(0.5)  # Give some time for logs to be written
        logs_found, log_entries = check_logs_for_request(request_id)
        result["logs_found"] = logs_found
        if logs_found:
            result["log_entries_count"] = len(log_entries) if isinstance(log_entries, list) else 0
        else:
            result["log_entries_error"] = log_entries
            
        # Check metrics if requested
        if endpoint.get("check_metrics", False):
            metrics_found, metrics_data = check_metrics_for_endpoint(url)
            result["metrics_found"] = metrics_found
            if metrics_found:
                result["metrics_entries_count"] = len(metrics_data) if isinstance(metrics_data, list) else 0
            else:
                result["metrics_error"] = metrics_data
                
        # Check veracity if requested
        if endpoint.get("check_veracity", False):
            veracity_found, veracity_data = check_veracity_for_request(request_id)
            result["veracity_found"] = veracity_found
            if veracity_found:
                result["veracity_entries_count"] = len(veracity_data) if isinstance(veracity_data, list) else 0
                # Extract veracity scores if available
                if isinstance(veracity_data, list) and veracity_data:
                    try:
                        result["veracity_scores"] = {
                            "accuracy_score": veracity_data[0].get("accuracy_score"),
                            "content_integrity_score": veracity_data[0].get("content_integrity_score"),
                            "semantic_score": veracity_data[0].get("semantic_score")
                        }
                    except:
                        pass
            else:
                result["veracity_error"] = veracity_data
        
        return result
    except Exception as e:
        end_time = time.time()
        time_ms = round((end_time - start_time) * 1000, 2)
        
        return {
            "endpoint": f"{method} {url}",
            "endpoint_name": endpoint_name,
            "category": endpoint.get("category", "Unknown"),
            "status": "ERROR",
            "message": str(e),
            "status_code": 0,
            "request_id": request_id,
            "time_ms": time_ms
        }

def save_results_to_file(results: List[Dict[str, Any]], filename: str = "enhanced_endpoint_results.json"):
    """Save test results to a JSON file."""
    # Count results by status
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    # Count logging, metrics, and veracity success
    logs_found_count = sum(1 for r in results if r.get("logs_found", False))
    metrics_found_count = sum(1 for r in results if r.get("metrics_found", False))
    veracity_found_count = sum(1 for r in results if r.get("veracity_found", False))
    
    metrics_endpoints = sum(1 for r in results if "metrics_found" in r)
    veracity_endpoints = sum(1 for r in results if "veracity_found" in r)
    
    total_count = len(results)
    
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
                "logs_found": 0,
                "metrics_found": 0,
                "veracity_found": 0
            }
        
        categories[category]["total"] += 1
        
        if result["status"] == "PASS":
            categories[category]["pass"] += 1
        elif result["status"] == "FAIL":
            categories[category]["fail"] += 1
        elif result["status"] == "ERROR":
            categories[category]["error"] += 1
            
        if result.get("logs_found", False):
            categories[category]["logs_found"] += 1
            
        if result.get("metrics_found", False):
            categories[category]["metrics_found"] += 1
            
        if result.get("veracity_found", False):
            categories[category]["veracity_found"] += 1
    
    # Create report data
    report = {
        "summary": {
            "total": total_count,
            "pass": pass_count,
            "fail": fail_count,
            "error": error_count,
            "pass_percentage": round((pass_count / total_count * 100), 1) if total_count > 0 else 0,
            "logs_found": logs_found_count,
            "logs_percentage": round((logs_found_count / total_count * 100), 1) if total_count > 0 else 0,
            "metrics_found": metrics_found_count,
            "metrics_percentage": round((metrics_found_count / metrics_endpoints * 100), 1) if metrics_endpoints > 0 else 0,
            "veracity_found": veracity_found_count,
            "veracity_percentage": round((veracity_found_count / veracity_endpoints * 100), 1) if veracity_endpoints > 0 else 0
        },
        "categories": categories,
        "results": results,
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "server": BASE_URL
    }
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Main function to test all endpoints."""
    ensure_log_directories()
    
    print(f"Testing {len(ENDPOINTS)} endpoints against {BASE_URL}")
    print("Please make sure the server is running.\n")
    
    results = []
    
    # Test each endpoint sequentially to better track logs
    for endpoint in ENDPOINTS:
        endpoint_name = endpoint.get("name", endpoint["url"].replace("/", "_").lstrip("_"))
        print(f"Testing {endpoint['method']} {endpoint['url']} ({endpoint_name})...")
        
        result = test_endpoint(endpoint)
        results.append(result)
        
        # Print result
        status_symbol = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "!"
        log_symbol = "📋" if result.get("logs_found", False) else "❌"
        metrics_symbol = "📊" if result.get("metrics_found", False) else "❌" if "metrics_found" in result else "➖"
        veracity_symbol = "🔍" if result.get("veracity_found", False) else "❌" if "veracity_found" in result else "➖"
        
        print(f"{status_symbol} {endpoint['method']} {endpoint['url']} - {result['status']} ({result['status_code']}) in {result['time_ms']}ms | Logs: {log_symbol} | Metrics: {metrics_symbol} | Veracity: {veracity_symbol}")
        
        # Short pause between requests to allow logs to be written
        time.sleep(1)
    
    # Count results by status
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    # Count logging, metrics, and veracity success
    logs_found_count = sum(1 for r in results if r.get("logs_found", False))
    metrics_found_count = sum(1 for r in results if r.get("metrics_found", False))
    veracity_found_count = sum(1 for r in results if r.get("veracity_found", False))
    
    metrics_endpoints = sum(1 for r in results if "metrics_found" in r)
    veracity_endpoints = sum(1 for r in results if "veracity_found" in r)
    
    total_count = len(results)
    
    # Print summary
    print("\n----- SUMMARY -----")
    print(f"Total endpoints: {total_count}")
    print(f"PASS: {pass_count} ({pass_count/total_count*100:.1f}%)")
    print(f"FAIL: {fail_count} ({fail_count/total_count*100:.1f}%)")
    print(f"ERROR: {error_count} ({error_count/total_count*100:.1f}%)")
    print(f"Logs found: {logs_found_count}/{total_count} ({logs_found_count/total_count*100:.1f}%)")
    
    if metrics_endpoints > 0:
        print(f"Metrics found: {metrics_found_count}/{metrics_endpoints} ({metrics_found_count/metrics_endpoints*100:.1f}%)")
    else:
        print(f"Metrics found: {metrics_found_count}/0 (0.0%)")
        
    if veracity_endpoints > 0:  
        print(f"Veracity found: {veracity_found_count}/{veracity_endpoints} ({veracity_found_count/veracity_endpoints*100:.1f}%)")
    else:
        print(f"Veracity found: {veracity_found_count}/0 (0.0%)")
    
    # Save results to file
    filename = f"enhanced_endpoint_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = save_results_to_file(results, filename)
    print(f"\nDetailed results written to {filename}")
    
    # Print category summary
    print("\n----- CATEGORY SUMMARY -----")
    for category, stats in report["categories"].items():
        pass_pct = round((stats["pass"] / stats["total"] * 100), 1) if stats["total"] > 0 else 0
        print(f"{category}: {stats['pass']}/{stats['total']} passed ({pass_pct}%)")
    
    return 0 if fail_count + error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())