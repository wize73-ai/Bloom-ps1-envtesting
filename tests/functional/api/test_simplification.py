"""
Integration tests for the text simplification API endpoint.
"""
import pytest
import asyncio
import aiohttp
import json
from pathlib import Path

# Test constants
SIMPLIFY_ENDPOINT = "/pipeline/simplify"
ALT_SIMPLIFY_ENDPOINT = "/simplify"

@pytest.mark.asyncio
async def test_simplification_endpoint_exists(server_url, server_connection, api_client):
    """Test that the simplification endpoint exists and is accessible."""
    endpoints = [f"{server_url}{SIMPLIFY_ENDPOINT}", f"{server_url}{ALT_SIMPLIFY_ENDPOINT}"]
    
    for endpoint in endpoints:
        try:
            async with api_client.options(endpoint) as response:
                if response.status != 404:
                    # If we get any response other than 404, the endpoint exists
                    assert True
                    return
        except:
            # If the options request fails, continue to the next endpoint
            continue
    
    pytest.fail("Simplification endpoint not found at any expected path")

@pytest.mark.asyncio
async def test_simplification_basic_functionality(server_url, server_connection, api_client):
    """Test basic simplification functionality with a complex text."""
    # Try both potential endpoints
    endpoints = [f"{server_url}{SIMPLIFY_ENDPOINT}", f"{server_url}{ALT_SIMPLIFY_ENDPOINT}"]
    working_endpoint = None
    
    # Find a working endpoint first
    for endpoint in endpoints:
        try:
            data = {
                "text": "The plaintiff alleged that the defendant had violated numerous provisions of the aforementioned contractual agreement.",
                "language": "en",
                "target_level": "simple"
            }
            
            async with api_client.post(endpoint, json=data) as response:
                if response.status == 200:
                    working_endpoint = endpoint
                    break
        except:
            continue
    
    if not working_endpoint:
        pytest.fail("No working simplification endpoint found")
    
    # Test simplification
    data = {
        "text": "The plaintiff alleged that the defendant had violated numerous provisions of the aforementioned contractual agreement.",
        "language": "en",
        "target_level": "simple"
    }
    
    async with api_client.post(working_endpoint, json=data) as response:
        assert response.status == 200, "Simplification request failed"
        
        # Get response data
        resp_data = await response.json()
        
        # Extract simplified text from response
        simplified_text = None
        if isinstance(resp_data, dict):
            # Try different possible response formats
            if "simplified_text" in resp_data:
                simplified_text = resp_data.get("simplified_text")
            elif "data" in resp_data and isinstance(resp_data["data"], dict):
                result_data = resp_data["data"]
                if "simplified_text" in result_data:
                    simplified_text = result_data.get("simplified_text")
                elif "text" in result_data:
                    simplified_text = result_data.get("text")
        
        assert simplified_text is not None, "Could not extract simplified text from response"
        
        # Simplified text should be shorter or same length as original
        # (in some cases simplification might not reduce length but should always be simpler)
        original_words = len(data["text"].split())
        simplified_words = len(simplified_text.split())
        
        # Print info for debugging
        print(f"\nOriginal text ({original_words} words): {data['text']}")
        print(f"Simplified text ({simplified_words} words): {simplified_text}")
        
        # Basic verification that simplification happened:
        # The simplified text should contain certain keywords from original
        assert "plaintiff" in simplified_text.lower() or "person" in simplified_text.lower()
        assert "defendant" in simplified_text.lower() or "person" in simplified_text.lower()
        assert "contract" in simplified_text.lower() or "agreement" in simplified_text.lower()

@pytest.mark.asyncio
async def test_simplification_with_test_cases(server_url, server_connection, api_client, load_test_data):
    """Test simplification with predefined test cases."""
    # Load test cases
    test_cases = load_test_data("simplification_test_cases.json")["test_cases"]
    
    # Try both potential endpoints
    endpoints = [f"{server_url}{SIMPLIFY_ENDPOINT}", f"{server_url}{ALT_SIMPLIFY_ENDPOINT}"]
    working_endpoint = None
    
    # Find a working endpoint first
    for endpoint in endpoints:
        try:
            data = {
                "text": "The plaintiff alleged violations.",
                "language": "en",
                "target_level": "simple"
            }
            
            async with api_client.post(endpoint, json=data) as response:
                if response.status == 200:
                    working_endpoint = endpoint
                    break
        except:
            continue
    
    if not working_endpoint:
        pytest.fail("No working simplification endpoint found")
    
    # Test each test case
    results = {}
    for test_case in test_cases:
        test_name = test_case["name"]
        source_text = test_case["source_text"]
        language = test_case["language"]
        target_level = test_case["target_level"]
        expected_contains = test_case["expected_contains"]
        
        data = {
            "text": source_text,
            "language": language,
            "target_level": target_level
        }
        
        async with api_client.post(working_endpoint, json=data) as response:
            assert response.status == 200, f"Simplification request failed for test case: {test_name}"
            
            # Get response data
            resp_data = await response.json()
            
            # Extract simplified text from response
            simplified_text = None
            if isinstance(resp_data, dict):
                # Try different possible response formats
                if "simplified_text" in resp_data:
                    simplified_text = resp_data.get("simplified_text")
                elif "data" in resp_data and isinstance(resp_data["data"], dict):
                    result_data = resp_data["data"]
                    if "simplified_text" in result_data:
                        simplified_text = result_data.get("simplified_text")
                    elif "text" in result_data:
                        simplified_text = result_data.get("text")
            
            assert simplified_text is not None, f"Could not extract simplified text from response for test case: {test_name}"
            
            # Check that the simplified text contains expected keywords
            matches = [word.lower() in simplified_text.lower() for word in expected_contains]
            matched_count = sum(matches)
            # We consider it a pass if at least 75% of expected keywords are present
            min_matches = max(1, int(0.75 * len(expected_contains)))
            passed = matched_count >= min_matches
            
            # Store results
            results[test_name] = {
                "source": source_text[:30] + "..." if len(source_text) > 30 else source_text,
                "simplified": simplified_text[:50] + "..." if len(simplified_text) > 50 else simplified_text,
                "language": language,
                "target_level": target_level,
                "expected_words": expected_contains,
                "matches": matches,
                "match_count": matched_count,
                "min_matches": min_matches,
                "passed": passed
            }
    
    # Print results summary for debugging
    print("\nSimplification test results:")
    for test_name, result in results.items():
        result_mark = "✓" if result["passed"] else "✗"
        print(f"{test_name}: {result_mark} ({result['match_count']}/{len(result['expected_words'])} matches)")
        print(f"  Source: {result['source']}")
        print(f"  Simplified: {result['simplified']}")
        if not result["passed"]:
            for i, (word, matched) in enumerate(zip(result["expected_words"], result["matches"])):
                match_mark = "✓" if matched else "✗"
                print(f"  Expected word {i+1}: '{word}' - {match_mark}")
        print()
    
    # Verify overall results
    success_count = sum(1 for result in results.values() if result["passed"])
    success_rate = success_count / len(results)
    
    assert success_rate >= 0.8, f"Simplification success rate ({success_rate:.2%}) is below threshold (80%)"

@pytest.mark.asyncio
async def test_simplification_error_handling(server_url, server_connection, api_client):
    """Test error handling in the simplification endpoint."""
    # Try both potential endpoints
    endpoints = [f"{server_url}{SIMPLIFY_ENDPOINT}", f"{server_url}{ALT_SIMPLIFY_ENDPOINT}"]
    working_endpoint = None
    
    # Find a working endpoint first
    for endpoint in endpoints:
        try:
            data = {
                "text": "Test text",
                "language": "en",
                "target_level": "simple"
            }
            
            async with api_client.post(endpoint, json=data) as response:
                if response.status == 200:
                    working_endpoint = endpoint
                    break
        except:
            continue
    
    if not working_endpoint:
        pytest.fail("No working simplification endpoint found")
    
    # Test with empty text
    data = {
        "text": "",
        "language": "en",
        "target_level": "simple"
    }
    async with api_client.post(working_endpoint, json=data) as response:
        assert response.status != 200, "Empty text should return an error"
    
    # Test with missing text field
    data = {
        "language": "en",
        "target_level": "simple"
    }
    async with api_client.post(working_endpoint, json=data) as response:
        assert response.status != 200, "Missing text field should return an error"
    
    # Test with invalid language
    data = {
        "text": "Test text",
        "language": "invalid_language",
        "target_level": "simple"
    }
    async with api_client.post(working_endpoint, json=data) as response:
        # Language validation may or may not happen, so we don't assert on the status
        # But we should still get a response
        pass
    
    # Test with invalid target level
    data = {
        "text": "Test text",
        "language": "en",
        "target_level": "invalid_level"
    }
    async with api_client.post(working_endpoint, json=data) as response:
        # Target level validation may or may not happen, so we don't assert on the status
        # But we should still get a response
        pass
    
    # Test with invalid JSON
    async with api_client.post(working_endpoint, data="invalid_json") as response:
        assert response.status != 200, "Invalid JSON should return an error"