#!/usr/bin/env python3
"""
CasaLingua Model Verification Tool

This script verifies each model in the system step by step,
checking for proper loading and operation without triggering
reload cascades.

It makes careful, controlled requests to ensure server stability.
"""

import os
import sys
import time
import json
import requests
import argparse

API_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Test data for various models
TEST_DATA = {
    "translation": {
        "text": "The quick brown fox jumps over the lazy dog.",
        "source_language": "en",
        "target_language": "es"
    },
    "language_detection": {
        "text": "Hello world, this is a test message."
    },
    "simplifier": {
        "text": "The aforementioned contractual obligations shall be considered null and void if the party of the first part fails to remit payment within the specified timeframe.",
        "target_grade_level": "5"
    },
    "anonymizer": {
        "text": "John Smith lives at 123 Main St, New York and his phone number is 555-123-4567.",
        "strategy": "redact"
    }
}

def print_header(message):
    """Print a header message"""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)

def print_success(message):
    """Print a success message"""
    print(f"✅ {message}")

def print_warning(message):
    """Print a warning message"""
    print(f"⚠️ {message}")

def print_error(message):
    """Print an error message"""
    print(f"❌ {message}")

def check_health():
    """Check basic server health"""
    print_header("Checking Basic Server Health")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        
        if response.status_code == 200:
            print_success(f"Server health check passed (Status: {response.status_code})")
            return True
        else:
            print_error(f"Server health check failed (Status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_error(f"Error during health check: {str(e)}")
        return False

def check_detailed_health():
    """Check detailed server health and get loaded models"""
    print_header("Checking Detailed Server Health")
    try:
        response = requests.get(f"{API_URL}/health/detailed", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for system information
            if "system" in data:
                sys_info = data["system"]
                print_success(f"System memory: {sys_info.get('memory_available', 'Unknown')}")
                print_success(f"System load: {sys_info.get('cpu_usage', 'Unknown')}%")
            
            # Check for loaded models
            loaded_models = []
            if "models" in data:
                models = data["models"]
                print(f"Found {len(models)} models in the detailed health data")
                
                for model in models:
                    model_name = model.get("name", "Unknown")
                    model_status = model.get("status", "Unknown")
                    
                    if model_status == "loaded":
                        print_success(f"Model '{model_name}' is loaded")
                        loaded_models.append(model_name)
                    else:
                        print_warning(f"Model '{model_name}' status: {model_status}")
            
            print(f"\nLoaded models: {', '.join(loaded_models) if loaded_models else 'None'}")
            return loaded_models
        else:
            print_error(f"Detailed health check failed (Status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return []
    except Exception as e:
        print_error(f"Error during detailed health check: {str(e)}")
        return []

def check_model_registry():
    """Check the model registry configuration"""
    print_header("Checking Model Registry")
    try:
        response = requests.get(f"{API_URL}/admin/models", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if "data" in data and "models" in data["data"]:
                models = data["data"]["models"]
                print_success(f"Model registry contains {len(models)} models")
                
                for model_id, model_info in models.items():
                    model_name = model_info.get("model_name", "Unknown")
                    model_type = model_info.get("type", "Unknown")
                    print(f"- {model_id}: {model_name} ({model_type})")
                
                return models
            else:
                print_warning("Model registry response has unexpected format")
                print(json.dumps(data, indent=2))
                return {}
        else:
            print_error(f"Model registry check failed (Status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return {}
    except Exception as e:
        print_error(f"Error checking model registry: {str(e)}")
        return {}

def test_translation(verbose=False, wait_time=5):
    """Test the translation model"""
    print_header("Testing Translation Model")
    
    payload = TEST_DATA["translation"]
    if verbose:
        print(f"Request data: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        print(f"Making translation request (en → es)...")
        response = requests.post(
            f"{API_URL}/pipeline/translate",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", {})
            
            if "translated_text" in data:
                translated_text = data["translated_text"]
                model_used = data.get("model_used", "Unknown")
                
                print_success(f"Translation successful in {elapsed:.2f} seconds")
                print(f"Source: \"{payload['text']}\"")
                print(f"Translation: \"{translated_text}\"")
                print(f"Model used: {model_used}")
                
                if verbose:
                    print("\nFull response:")
                    print(json.dumps(result, indent=2))
                
                # Wait before continuing to next test
                if wait_time > 0:
                    print(f"Waiting {wait_time} seconds before next test...")
                    time.sleep(wait_time)
                
                return True, model_used
            else:
                print_error("No translation in response")
                if verbose:
                    print("\nFull response:")
                    print(json.dumps(result, indent=2))
                return False, None
        else:
            print_error(f"Translation request failed (Status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return False, None
    except Exception as e:
        print_error(f"Error during translation test: {str(e)}")
        return False, None

def test_language_detection(verbose=False, wait_time=5):
    """Test the language detection model"""
    print_header("Testing Language Detection Model")
    
    payload = TEST_DATA["language_detection"]
    if verbose:
        print(f"Request data: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        print("Making language detection request...")
        response = requests.post(
            f"{API_URL}/pipeline/detect",
            headers=HEADERS,
            json=payload,
            timeout=15
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", {})
            
            if "detected_language" in data:
                detected_language = data["detected_language"]
                confidence = data.get("confidence", 0)
                
                print_success(f"Language detection successful in {elapsed:.2f} seconds")
                print(f"Text: \"{payload['text']}\"")
                print(f"Detected language: {detected_language}")
                print(f"Confidence: {confidence:.2f}")
                
                if verbose:
                    print("\nFull response:")
                    print(json.dumps(result, indent=2))
                
                # Wait before continuing to next test
                if wait_time > 0:
                    print(f"Waiting {wait_time} seconds before next test...")
                    time.sleep(wait_time)
                
                return True
            else:
                print_error("No language detection in response")
                if verbose:
                    print("\nFull response:")
                    print(json.dumps(result, indent=2))
                return False
        else:
            print_error(f"Language detection request failed (Status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_error(f"Error during language detection test: {str(e)}")
        return False

def test_simplification(verbose=False, wait_time=5):
    """Test the text simplification model"""
    print_header("Testing Text Simplification Model")
    
    payload = TEST_DATA["simplifier"]
    if verbose:
        print(f"Request data: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        print("Making text simplification request...")
        response = requests.post(
            f"{API_URL}/pipeline/simplify",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", {})
            
            if "simplified_text" in data:
                simplified_text = data["simplified_text"]
                
                print_success(f"Text simplification successful in {elapsed:.2f} seconds")
                print(f"Original: \"{payload['text'][:50]}...\"")
                print(f"Simplified: \"{simplified_text}\"")
                
                if verbose:
                    print("\nFull response:")
                    print(json.dumps(result, indent=2))
                
                # Wait before continuing to next test
                if wait_time > 0:
                    print(f"Waiting {wait_time} seconds before next test...")
                    time.sleep(wait_time)
                
                return True
            else:
                print_error("No simplified text in response")
                if verbose:
                    print("\nFull response:")
                    print(json.dumps(result, indent=2))
                return False
        else:
            print_error(f"Text simplification request failed (Status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_error(f"Error during text simplification test: {str(e)}")
        return False

def run_complete_test(verbose=False, wait_time=10):
    """Run a complete test of all models"""
    # First check the server health
    if not check_health():
        print_error("Basic health check failed, cannot continue with tests")
        return False
    
    # Get loaded models
    loaded_models = check_detailed_health()
    
    # Check model registry
    registry_models = check_model_registry()
    
    # Test individual models with delays between tests
    models_tested = 0
    models_passed = 0
    
    # Test translation
    print("\nStarting individual model tests...")
    success, model_used = test_translation(verbose, wait_time)
    models_tested += 1
    if success:
        models_passed += 1
    
    # Test language detection
    success = test_language_detection(verbose, wait_time)
    models_tested += 1
    if success:
        models_passed += 1
    
    # Test simplification
    success = test_simplification(verbose, wait_time)
    models_tested += 1
    if success:
        models_passed += 1
    
    # Print summary
    print_header("Test Summary")
    print(f"Tests run: {models_tested}")
    print(f"Tests passed: {models_passed}")
    print(f"Success rate: {(models_passed / models_tested) * 100:.0f}%")
    
    if models_passed == models_tested:
        print_success("All model tests passed!")
        return True
    else:
        print_warning(f"{models_tested - models_passed} tests failed")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="CasaLingua Model Verification Tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("-w", "--wait", type=int, default=10, help="Wait time between tests in seconds (default: 10)")
    parser.add_argument("-u", "--url", type=str, default="http://localhost:8000", help="API URL (default: http://localhost:8000)")
    args = parser.parse_args()
    
    global API_URL
    API_URL = args.url
    
    print("CasaLingua Model Verification Tool")
    print(f"API URL: {API_URL}")
    print(f"Verbose mode: {'On' if args.verbose else 'Off'}")
    print(f"Wait time: {args.wait} seconds")
    
    success = run_complete_test(args.verbose, args.wait)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())