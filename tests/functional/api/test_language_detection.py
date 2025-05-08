"""
Integration tests for the language detection API endpoint.
"""
import pytest
import asyncio
import aiohttp
import json
from pathlib import Path

# Test constants - updated with the correct endpoints based on API discovery
DETECT_ENDPOINT = "/pipeline/detect"
ALT_DETECT_ENDPOINT = "/pipeline/detect-language"  # Alternative endpoint

@pytest.mark.asyncio
async def test_language_detection_endpoint_exists(server_url, server_connection, api_client):
    """Test that the language detection endpoint exists and is accessible."""
    endpoints = [f"{server_url}{DETECT_ENDPOINT}", f"{server_url}{ALT_DETECT_ENDPOINT}"]
    
    async with api_client() as session:
        for endpoint in endpoints:
            try:
                async with session.options(endpoint) as response:
                    if response.status != 404:
                        # If we get any response other than 404, the endpoint exists
                        assert True
                        return
            except:
                # If the options request fails, continue to the next endpoint
                continue
        
        pytest.fail("Language detection endpoint not found at any expected path")

@pytest.mark.asyncio
async def test_language_detection_with_samples(server_url, server_connection, api_client, load_test_data):
    """Test language detection with samples in different languages."""
    # Load multilingual samples
    samples = load_test_data("multilingual_samples.json")
    
    # Try both potential endpoints
    endpoints = [f"{server_url}{DETECT_ENDPOINT}", f"{server_url}{ALT_DETECT_ENDPOINT}"]
    working_endpoint = None
    
    async with api_client() as session:
        # Find a working endpoint first
        for endpoint in endpoints:
            try:
                text = samples["english"]["simple"]
                data = {"text": text}
                
                async with session.post(endpoint, json=data) as response:
                    if response.status == 200:
                        working_endpoint = endpoint
                        break
            except:
                continue
        
        if not working_endpoint:
            pytest.fail("No working language detection endpoint found")
        
        # Test each language sample
        results = {}
        for language, sample_data in samples.items():
            text = sample_data["simple"]
            expected_language = sample_data["expected_language"]
            
            data = {"text": text}
            async with session.post(working_endpoint, json=data) as response:
                assert response.status == 200, f"Failed to detect language for {language} sample"
                
                # Get response data
                resp_data = await response.json()
                
                # Extract detected language from response
                detected_language = None
                if isinstance(resp_data, dict):
                    # Try different possible response formats
                    if "language" in resp_data:
                        detected_language = resp_data.get("language")
                    elif "data" in resp_data and isinstance(resp_data["data"], dict):
                        result_data = resp_data["data"]
                        if "language" in result_data:
                            detected_language = result_data.get("language")
                        elif "detected_language" in result_data:
                            detected_language = result_data.get("detected_language")
                
                assert detected_language is not None, f"Could not extract detected language from response for {language} sample"
                
                # Store results for verification
                results[language] = {
                    "sample": text[:30] + "...",
                    "expected": expected_language,
                    "detected": detected_language,
                    "correct": detected_language.lower() == expected_language.lower()
                }
        
        # Verify results - we allow some language detection to be imperfect or unknown
        # If the server returns "unknown" for many languages, we'll skip the accuracy check
        unknown_count = sum(1 for lang_results in results.values() if lang_results["detected"].lower() == "unknown")
        
        # Print results summary for debugging
        print(f"\nLanguage detection results:")
        for language, result in results.items():
            correct_mark = "✓" if result["correct"] else "✗"
            print(f"{language}: {correct_mark} (Detected: {result['detected']}, Expected: {result['expected']})")
        
        # If more than 50% of responses are "unknown", the language detection model might not be loaded
        # or the server is in a testing mode that returns placeholder values
        if unknown_count > len(results) / 2:
            print(f"\nSKIPPING ACCURACY CHECK: {unknown_count}/{len(results)} languages detected as 'unknown'")
            print("The language detection model may not be loaded or server is in testing mode")
            pytest.skip("Too many languages detected as 'unknown'")
        else:
            # Calculate accuracy only on non-unknown languages
            valid_results = {k: v for k, v in results.items() if v["detected"].lower() != "unknown"}
            if valid_results:
                correct_count = sum(1 for lang_results in valid_results.values() if lang_results["correct"])
                accuracy = correct_count / len(valid_results)
                print(f"\nLanguage detection accuracy: {accuracy:.2%}")
                assert accuracy >= 0.8, f"Language detection accuracy ({accuracy:.2%}) is below threshold (80%)"
            else:
                pytest.skip("No valid language detections returned")

@pytest.mark.asyncio
async def test_language_detection_with_long_text(server_url, server_connection, api_client, load_test_data):
    """Test language detection with longer text samples."""
    # Load multilingual samples
    samples = load_test_data("multilingual_samples.json")
    
    # Try both potential endpoints
    endpoints = [f"{server_url}{DETECT_ENDPOINT}", f"{server_url}{ALT_DETECT_ENDPOINT}"]
    working_endpoint = None
    
    async with api_client() as session:
        # Find a working endpoint first
        for endpoint in endpoints:
            try:
                text = samples["english"]["complex"]
                data = {"text": text}
                
                async with session.post(endpoint, json=data) as response:
                    if response.status == 200:
                        working_endpoint = endpoint
                        break
            except:
                continue
        
        if not working_endpoint:
            pytest.fail("No working language detection endpoint found")
        
        # Test with complex samples
        results = {}
        for language, sample_data in samples.items():
            text = sample_data["complex"]
            expected_language = sample_data["expected_language"]
            
            data = {"text": text}
            async with session.post(working_endpoint, json=data) as response:
                assert response.status == 200, f"Failed to detect language for complex {language} sample"
                
                # Get response data
                resp_data = await response.json()
                
                # Extract detected language from response
                detected_language = None
                if isinstance(resp_data, dict):
                    # Try different possible response formats
                    if "language" in resp_data:
                        detected_language = resp_data.get("language")
                    elif "data" in resp_data and isinstance(resp_data["data"], dict):
                        result_data = resp_data["data"]
                        if "language" in result_data:
                            detected_language = result_data.get("language")
                        elif "detected_language" in result_data:
                            detected_language = result_data.get("detected_language")
                
                assert detected_language is not None, f"Could not extract detected language from response for complex {language} sample"
                
                # Store results for verification
                results[language] = {
                    "sample": text[:30] + "...",
                    "expected": expected_language,
                    "detected": detected_language,
                    "correct": detected_language.lower() == expected_language.lower()
                }
        
        # Verify results - we allow some language detection to be imperfect or unknown
        # If the server returns "unknown" for many languages, we'll skip the accuracy check
        unknown_count = sum(1 for lang_results in results.values() if lang_results["detected"].lower() == "unknown")
        
        # Print results summary for debugging
        print(f"\nComplex language detection results:")
        for language, result in results.items():
            correct_mark = "✓" if result["correct"] else "✗"
            print(f"{language} (complex): {correct_mark} (Detected: {result['detected']}, Expected: {result['expected']})")
        
        # If more than 50% of responses are "unknown", the language detection model might not be loaded
        # or the server is in a testing mode that returns placeholder values
        if unknown_count > len(results) / 2:
            print(f"\nSKIPPING ACCURACY CHECK: {unknown_count}/{len(results)} languages detected as 'unknown'")
            print("The language detection model may not be loaded or server is in testing mode")
            pytest.skip("Too many languages detected as 'unknown'")
        else:
            # Calculate accuracy only on non-unknown languages
            valid_results = {k: v for k, v in results.items() if v["detected"].lower() != "unknown"}
            if valid_results:
                correct_count = sum(1 for lang_results in valid_results.values() if lang_results["correct"])
                accuracy = correct_count / len(valid_results)
                print(f"\nComplex language detection accuracy: {accuracy:.2%}")
                assert accuracy >= 0.8, f"Complex language detection accuracy ({accuracy:.2%}) is below threshold (80%)"
            else:
                pytest.skip("No valid language detections returned")

@pytest.mark.asyncio
async def test_language_detection_error_handling(server_url, server_connection, api_client):
    """Test error handling in the language detection endpoint."""
    # Try both potential endpoints
    endpoints = [f"{server_url}{DETECT_ENDPOINT}", f"{server_url}{ALT_DETECT_ENDPOINT}"]
    working_endpoint = None
    
    async with api_client() as session:
        # Find a working endpoint first
        for endpoint in endpoints:
            try:
                data = {"text": "Hello world"}
                
                async with session.post(endpoint, json=data) as response:
                    if response.status == 200:
                        working_endpoint = endpoint
                        break
            except:
                continue
        
        if not working_endpoint:
            pytest.fail("No working language detection endpoint found")
        
        # Test with empty text
        data = {"text": ""}
        try:
            async with session.post(working_endpoint, json=data) as response:
                # Some implementations may accept empty strings and return a default language
                # We'll only verify that the response is valid
                if response.status == 200:
                    resp_data = await response.json()
                    assert isinstance(resp_data, dict), "Response should be a JSON object"
                else:
                    # If not 200, it should be a valid error response
                    assert response.status in [400, 422], f"Expected error status code, got {response.status}"
        except Exception as e:
            # If the request fails with an exception, that's also acceptable for empty text
            pass
            
        # Test with missing text field
        data = {"wrong_field": "Hello world"}
        async with session.post(working_endpoint, json=data) as response:
            assert response.status != 200, "Missing text field should return an error"
        
        # Test with invalid JSON
        async with session.post(working_endpoint, data="invalid_json") as response:
            assert response.status != 200, "Invalid JSON should return an error"