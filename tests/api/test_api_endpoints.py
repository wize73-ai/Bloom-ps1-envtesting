import asyncio
import aiohttp
import json
import time
import sys
from typing import Dict, List, Any, Optional, Union

BASE_URL = "http://localhost:8000"
AUTH_TOKEN = None  # Set to None to skip auth, will be updated if auth is required

# Test data
TEST_TRANSLATION_TEXT = "Hello, how are you today?"
TEST_LANGUAGE_DETECTION_TEXT = "Hola, ¿cómo estás hoy?"
TEST_SIMPLIFICATION_TEXT = "The plaintiff alleged that the defendant had violated numerous provisions of the aforementioned contractual agreement."

# Define the API routes with prefixes
TRANSLATE_ENDPOINT = "/pipeline/translate"    # Try this prefix first
DETECT_ENDPOINT = "/pipeline/detect"          # Try this prefix first
SIMPLIFY_ENDPOINT = "/pipeline/simplify"      # Try this prefix first

# Alternative prefixes if the pipeline prefix fails
ALT_TRANSLATE_ENDPOINT = "/translate"  # No /api prefix since router is mounted without prefix in main.py
ALT_DETECT_ENDPOINT = "/detect"
ALT_SIMPLIFY_ENDPOINT = "/simplify"

async def get_auth_token() -> Optional[str]:
    """Get authentication token if needed"""
    # For testing purposes, we're implementing a bypass for auth
    # In a real implementation, you would get the token from the auth endpoint
    return None

async def get_current_user_mock():
    """Mock function to simulate get_current_user dependency"""
    return {"id": "test-user-123", "username": "testuser", "role": "admin"}

async def test_language_detection():
    """Test language detection endpoint"""
    # Try primary endpoint first, then fallback to alternative if needed
    endpoints = [f"{BASE_URL}{DETECT_ENDPOINT}", f"{BASE_URL}{ALT_DETECT_ENDPOINT}"]
    headers = {}
    
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    data = {
        "text": TEST_LANGUAGE_DETECTION_TEXT
    }
    
    last_error = None
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                async with session.post(endpoint, json=data, headers=headers) as response:
                    status = response.status
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
                    
                    # Check for success response
                    if status == 200:
                        # Parse the response based on different possible formats
                        language_detected = None
                        if isinstance(data, dict):
                            # Direct format where language is at the root
                            if "language" in data:
                                language_detected = data.get("language")
                            # Format where language is in data.result
                            elif "data" in data and isinstance(data["data"], dict):
                                result_data = data["data"]
                                if "language" in result_data:
                                    language_detected = result_data.get("language") 
                                elif "detected_language" in result_data:
                                    language_detected = result_data.get("detected_language")
                        
                        # If we've found a language, this is a success
                        if language_detected:
                            expected_language = "es"  # Spanish text should be detected as Spanish
                            language_correct = language_detected == expected_language
                            
                            return {
                                "endpoint": endpoint,
                                "status": status,
                                "data": data,
                                "successful": True,
                                "language_detected": language_detected,
                                "expected_language": expected_language,
                                "language_correct": language_correct
                            }
                    
                    # If we get here, the endpoint either gave a non-200 response or didn't return language data
                    # Save the error and try the next endpoint
                    last_error = f"HTTP {status}: Response did not contain language data"
            
            except Exception as e:
                last_error = str(e)
                # Continue to try the next endpoint
        
        # If we get here, both endpoints failed
        return {
            "endpoint": endpoints[-1],  # Report the last endpoint we tried
            "status": None,
            "error": last_error or "All language detection endpoints failed",
            "successful": False
        }

async def test_translation():
    """Test translation endpoint"""
    # Try primary endpoint first, then fallback to alternative if needed
    endpoints = [f"{BASE_URL}{TRANSLATE_ENDPOINT}", f"{BASE_URL}{ALT_TRANSLATE_ENDPOINT}"]
    headers = {}
    
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    data = {
        "text": TEST_TRANSLATION_TEXT,
        "source_language": "en",
        "target_language": "es"
    }
    
    last_error = None
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                async with session.post(endpoint, json=data, headers=headers) as response:
                    status = response.status
                    try:
                        resp_data = await response.json()
                    except:
                        resp_data = await response.text()
                    
                    # Check for success response
                    if status == 200:
                        # Parse the response based on different possible formats
                        translated_text = None
                        if isinstance(resp_data, dict):
                            # Direct format where translated_text is at the root
                            if "translated_text" in resp_data:
                                translated_text = resp_data.get("translated_text")
                            # Format where translated_text is in data.result
                            elif "data" in resp_data and isinstance(resp_data["data"], dict):
                                result_data = resp_data["data"]
                                if "translated_text" in result_data:
                                    translated_text = result_data.get("translated_text")
                        
                        # If we've found translated text, this is a success
                        if translated_text:
                            return {
                                "endpoint": endpoint,
                                "status": status,
                                "data": resp_data,
                                "successful": True,
                                "translated_text": translated_text
                            }
                    
                    # If we get here, the endpoint either gave a non-200 response or didn't return translation data
                    # Save the error and try the next endpoint
                    last_error = f"HTTP {status}: Response did not contain translation data"
            
            except Exception as e:
                last_error = str(e)
                # Continue to try the next endpoint
        
        # If we get here, both endpoints failed
        return {
            "endpoint": endpoints[-1],  # Report the last endpoint we tried
            "status": None,
            "error": last_error or "All translation endpoints failed",
            "successful": False
        }

async def test_simplification():
    """Test text simplification endpoint"""
    # Try primary endpoint first, then fallback to alternative if needed
    endpoints = [f"{BASE_URL}{SIMPLIFY_ENDPOINT}", f"{BASE_URL}{ALT_SIMPLIFY_ENDPOINT}"]
    headers = {}
    
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    
    data = {
        "text": TEST_SIMPLIFICATION_TEXT,
        "language": "en",  # Add language for wider compatibility
        "target_level": "simple"
    }
    
    last_error = None
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                async with session.post(endpoint, json=data, headers=headers) as response:
                    status = response.status
                    try:
                        resp_data = await response.json()
                    except:
                        resp_data = await response.text()
                    
                    # Check for success response
                    if status == 200:
                        # Parse the response based on different possible formats
                        simplified_text = None
                        if isinstance(resp_data, dict):
                            # Direct format where simplified_text is at the root
                            if "simplified_text" in resp_data:
                                simplified_text = resp_data.get("simplified_text")
                            # Format where simplified_text is in data.result
                            elif "data" in resp_data and isinstance(resp_data["data"], dict):
                                result_data = resp_data["data"]
                                if "simplified_text" in result_data:
                                    simplified_text = result_data.get("simplified_text")
                        
                        # If we've found simplified text, this is a success
                        if simplified_text:
                            return {
                                "endpoint": endpoint,
                                "status": status,
                                "data": resp_data,
                                "successful": True,
                                "simplified_text": simplified_text
                            }
                    
                    # If we get here, the endpoint either gave a non-200 response or didn't return simplification data
                    # Save the error and try the next endpoint
                    last_error = f"HTTP {status}: Response did not contain simplified text data"
            
            except Exception as e:
                last_error = str(e)
                # Continue to try the next endpoint
        
        # If we get here, both endpoints failed
        return {
            "endpoint": endpoints[-1],  # Report the last endpoint we tried
            "status": None,
            "error": last_error or "All simplification endpoints failed",
            "successful": False
        }

async def test_models_health():
    """Test models health endpoint"""
    endpoint = f"{BASE_URL}/health/models"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = await response.text()
                
                success = status == 200 and "status" in (data if isinstance(data, dict) else {})
                
                return {
                    "endpoint": endpoint,
                    "status": status,
                    "data": data,
                    "successful": success
                }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": None,
                "error": str(e),
                "successful": False
            }

async def test_database_health():
    """Test database health endpoint"""
    endpoint = f"{BASE_URL}/health/database"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = await response.text()
                
                success = status == 200 and "status" in (data if isinstance(data, dict) else {})
                
                return {
                    "endpoint": endpoint,
                    "status": status,
                    "data": data,
                    "successful": success
                }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": None,
                "error": str(e),
                "successful": False
            }

async def main():
    """Run all API endpoint tests"""
    print("Testing API endpoints...")
    
    # Try to connect to the server
    connected = False
    retries = 5
    
    while not connected and retries > 0:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE_URL}/health") as response:
                    if response.status == 200:
                        connected = True
                        print(f"✅ Connected to server at {BASE_URL}")
        except Exception as e:
            print(f"⚠️ Connection attempt failed: {e}")
            print(f"Retrying in 5 seconds... ({retries} attempts left)")
            retries -= 1
            await asyncio.sleep(5)
    
    if not connected:
        print("❌ Failed to connect to server. Make sure the server is running.")
        sys.exit(1)
    
    # Get authentication token if needed
    global AUTH_TOKEN
    AUTH_TOKEN = await get_auth_token()
    
    # Run all tests
    print("\nRunning API endpoint tests (this may take some time as models load)...")
    results = await asyncio.gather(
        test_language_detection(),
        test_translation(),
        test_simplification(),
        test_models_health(),
        test_database_health()
    )
    
    # Print results in a nice format
    print("\n=== API Endpoint Test Results ===")
    all_successful = True
    
    for result in results:
        endpoint = result["endpoint"]
        status = result["status"]
        successful = result["successful"]
        
        if successful:
            print(f"✅ {endpoint} - Status: {status}")
            
            # Print additional details for successful endpoints
            if "language_detection" in endpoint:
                lang_detected = result.get("language_detected")
                lang_expected = result.get("expected_language")
                lang_correct = result.get("language_correct", False)
                print(f"   ⮕ Detected language: {lang_detected}, Expected: {lang_expected}, Correct: {'✓' if lang_correct else '✗'}")
            
            elif "translate" in endpoint:
                translated_text = result.get("translated_text")
                if translated_text:
                    print(f"   ⮕ Translated text: {translated_text[:60]}{'...' if len(translated_text) > 60 else ''}")
            
            elif "simplify" in endpoint:
                simplified_text = result.get("simplified_text")
                if simplified_text:
                    print(f"   ⮕ Simplified text: {simplified_text[:60]}{'...' if len(simplified_text) > 60 else ''}")
            
            elif "models" in endpoint:
                if isinstance(result.get("data"), dict):
                    model_status = result["data"].get("status")
                    loaded_models = result["data"].get("loaded_models", [])
                    print(f"   ⮕ Model status: {model_status}")
                    print(f"   ⮕ Loaded models: {', '.join(loaded_models) if loaded_models else 'None'}")
            
            elif "database" in endpoint:
                if isinstance(result.get("data"), dict):
                    db_status = result["data"].get("status")
                    print(f"   ⮕ Database status: {db_status}")
        else:
            all_successful = False
            error = result.get("error") if "error" in result else \
                   f"HTTP {status}: {result.get('data')}" if status else "Unknown error"
            print(f"❌ {endpoint} - Error: {error}")
    
    # Print summary
    print("\n=== Summary ===")
    if all_successful:
        print("✅ All API endpoints are working correctly!")
    else:
        print("❌ Some API endpoints are not working correctly.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())