"""
End-to-end tests for translation workflows.

These tests simulate complete user workflows, testing multiple API endpoints in sequence
to verify that the entire translation pipeline works correctly.
"""
import pytest
import asyncio
import aiohttp
import json
from pathlib import Path
import time

# Test constants
DETECT_ENDPOINT = "/pipeline/detect"
TRANSLATE_ENDPOINT = "/pipeline/translate"
ALT_DETECT_ENDPOINT = "/detect"
ALT_TRANSLATE_ENDPOINT = "/translate"

@pytest.mark.asyncio
async def test_detect_then_translate_workflow(server_url, server_connection, api_client, load_test_data):
    """Test a workflow that first detects language and then translates the text."""
    # Load multilingual samples
    samples = load_test_data("multilingual_samples.json")
    
    # Try both potential endpoint combinations
    endpoint_pairs = [
        (f"{server_url}{DETECT_ENDPOINT}", f"{server_url}{TRANSLATE_ENDPOINT}"),
        (f"{server_url}{ALT_DETECT_ENDPOINT}", f"{server_url}{ALT_TRANSLATE_ENDPOINT}"),
        (f"{server_url}{DETECT_ENDPOINT}", f"{server_url}{ALT_TRANSLATE_ENDPOINT}"),
        (f"{server_url}{ALT_DETECT_ENDPOINT}", f"{server_url}{TRANSLATE_ENDPOINT}")
    ]
    
    working_endpoints = None
    
    # Find working endpoints first
    for detect_endpoint, translate_endpoint in endpoint_pairs:
        try:
            # Test detection
            detect_data = {"text": "Hello world"}
            async with api_client.post(detect_endpoint, json=detect_data) as detection_response:
                if detection_response.status != 200:
                    continue
                
                # Test translation with a known language pair
                translate_data = {
                    "text": "Hello world",
                    "source_language": "en",
                    "target_language": "es"
                }
                async with api_client.post(translate_endpoint, json=translate_data) as translation_response:
                    if translation_response.status == 200:
                        working_endpoints = (detect_endpoint, translate_endpoint)
                        break
        except:
            continue
    
    if not working_endpoints:
        pytest.fail("No working detection and translation endpoints found")
    
    detect_endpoint, translate_endpoint = working_endpoints
    
    # Test the workflow with multiple language samples
    languages_to_test = ["spanish", "french", "german", "italian"]
    target_language = "en"  # We'll translate everything to English
    
    results = {}
    for language in languages_to_test:
        # 1. Get the sample text
        sample_text = samples[language]["simple"]
        expected_language = samples[language]["expected_language"]
        
        print(f"\nTesting {language} to English workflow:")
        print(f"Sample text: {sample_text}")
        
        # 2. Detect the language
        detect_data = {"text": sample_text}
        async with api_client.post(detect_endpoint, json=detect_data) as detection_response:
            assert detection_response.status == 200, f"Language detection failed for {language} sample"
            
            detection_result = await detection_response.json()
            
            # Extract detected language from response
            detected_language = None
            if isinstance(detection_result, dict):
                # Try different possible response formats
                if "language" in detection_result:
                    detected_language = detection_result.get("language")
                elif "data" in detection_result and isinstance(detection_result["data"], dict):
                    result_data = detection_result["data"]
                    if "language" in result_data:
                        detected_language = result_data.get("language")
                    elif "detected_language" in result_data:
                        detected_language = result_data.get("detected_language")
            
            assert detected_language is not None, f"Could not extract detected language from response for {language} sample"
            
            print(f"Detected language: {detected_language} (Expected: {expected_language})")
            
            # 3. Translate using the detected language
            translate_data = {
                "text": sample_text,
                "source_language": detected_language,
                "target_language": target_language
            }
            
            async with api_client.post(translate_endpoint, json=translate_data) as translation_response:
                assert translation_response.status == 200, f"Translation failed for {language} sample"
                
                translation_result = await translation_response.json()
                
                # Extract translated text from response
                translated_text = None
                if isinstance(translation_result, dict):
                    # Try different possible response formats
                    if "translated_text" in translation_result:
                        translated_text = translation_result.get("translated_text")
                    elif "data" in translation_result and isinstance(translation_result["data"], dict):
                        result_data = translation_result["data"]
                        if "translated_text" in result_data:
                            translated_text = result_data.get("translated_text")
                
                assert translated_text is not None, f"Could not extract translated text from response for {language} sample"
                
                print(f"Translated text: {translated_text}")
                
                # Store results
                results[language] = {
                    "original": sample_text,
                    "expected_language": expected_language,
                    "detected_language": detected_language,
                    "language_detected_correctly": detected_language == expected_language,
                    "translated_text": translated_text,
                    "target_language": target_language
                }
    
    # Verify results
    correct_detections = sum(1 for result in results.values() if result["language_detected_correctly"])
    detection_accuracy = correct_detections / len(results)
    
    print("\nWorkflow test results:")
    print(f"Language detection accuracy: {detection_accuracy:.2%}")
    
    for language, result in results.items():
        detection_mark = "✓" if result["language_detected_correctly"] else "✗"
        print(f"\n{language}:")
        print(f"  Original: {result['original']}")
        print(f"  Detected language: {result['detected_language']} (Expected: {result['expected_language']}) {detection_mark}")
        print(f"  Translation to {result['target_language']}: {result['translated_text']}")
    
    assert detection_accuracy >= 0.8, f"Language detection accuracy in workflow ({detection_accuracy:.2%}) is below threshold (80%)"
    
    # All samples should have translations
    assert all("translated_text" in result and result["translated_text"] for result in results.values()), "Not all samples were translated"

@pytest.mark.asyncio
async def test_multi_stage_translation_workflow(server_url, server_connection, api_client, load_test_data):
    """Test a multi-stage translation workflow: English -> Spanish -> French -> English."""
    # Find working translation endpoint
    endpoints = [f"{server_url}{TRANSLATE_ENDPOINT}", f"{server_url}{ALT_TRANSLATE_ENDPOINT}"]
    working_endpoint = None
    
    for endpoint in endpoints:
        try:
            data = {
                "text": "Hello",
                "source_language": "en",
                "target_language": "es"
            }
            
            async with api_client.post(endpoint, json=data) as response:
                if response.status == 200:
                    working_endpoint = endpoint
                    break
        except:
            continue
    
    if not working_endpoint:
        pytest.fail("No working translation endpoint found")
    
    # Original English text
    original_text = "The quick brown fox jumps over the lazy dog."
    
    # Stage 1: English -> Spanish
    print("\nMulti-stage translation workflow:")
    print(f"Stage 1: English -> Spanish")
    print(f"Original (EN): {original_text}")
    
    stage1_data = {
        "text": original_text,
        "source_language": "en",
        "target_language": "es"
    }
    
    async with api_client.post(working_endpoint, json=stage1_data) as response:
        assert response.status == 200, "Translation from English to Spanish failed"
        
        result = await response.json()
        
        # Extract Spanish text
        spanish_text = None
        if isinstance(result, dict):
            if "translated_text" in result:
                spanish_text = result.get("translated_text")
            elif "data" in result and isinstance(result["data"], dict):
                result_data = result["data"]
                if "translated_text" in result_data:
                    spanish_text = result_data.get("translated_text")
        
        assert spanish_text is not None, "Could not extract Spanish translation"
        print(f"Spanish (ES): {spanish_text}")
        
        # Add a small delay to avoid rate limiting
        await asyncio.sleep(1)
        
        # Stage 2: Spanish -> French
        print(f"Stage 2: Spanish -> French")
        
        stage2_data = {
            "text": spanish_text,
            "source_language": "es",
            "target_language": "fr"
        }
        
        async with api_client.post(working_endpoint, json=stage2_data) as response:
            assert response.status == 200, "Translation from Spanish to French failed"
            
            result = await response.json()
            
            # Extract French text
            french_text = None
            if isinstance(result, dict):
                if "translated_text" in result:
                    french_text = result.get("translated_text")
                elif "data" in result and isinstance(result["data"], dict):
                    result_data = result["data"]
                    if "translated_text" in result_data:
                        french_text = result_data.get("translated_text")
            
            assert french_text is not None, "Could not extract French translation"
            print(f"French (FR): {french_text}")
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(1)
            
            # Stage 3: French -> English
            print(f"Stage 3: French -> English")
            
            stage3_data = {
                "text": french_text,
                "source_language": "fr",
                "target_language": "en"
            }
            
            async with api_client.post(working_endpoint, json=stage3_data) as response:
                assert response.status == 200, "Translation from French to English failed"
                
                result = await response.json()
                
                # Extract final English text
                final_english_text = None
                if isinstance(result, dict):
                    if "translated_text" in result:
                        final_english_text = result.get("translated_text")
                    elif "data" in result and isinstance(result["data"], dict):
                        result_data = result["data"]
                        if "translated_text" in result_data:
                            final_english_text = result_data.get("translated_text")
                
                assert final_english_text is not None, "Could not extract final English translation"
                print(f"Back to English (EN): {final_english_text}")
                
                # Verify that the final English text contains some of the original keywords
                original_keywords = ["quick", "fox", "jumps", "lazy", "dog"]
                final_keywords = [word.lower() for word in final_english_text.split()]
                
                matches = [keyword in ' '.join(final_keywords) for keyword in original_keywords]
                match_count = sum(matches)
                
                # We expect some meaning preservation, but allow for translation variations
                # Require at least 2 of the 5 keywords to be present
                assert match_count >= 2, f"Final translation lost too much meaning, only {match_count}/5 keywords preserved"
                
                # Print keyword matching details
                print("\nKeyword preservation:")
                for keyword, matched in zip(original_keywords, matches):
                    match_mark = "✓" if matched else "✗"
                    print(f"  '{keyword}': {match_mark}")
                print(f"Overall: {match_count}/5 keywords preserved")

@pytest.mark.asyncio
async def test_bulk_translation_workflow(server_url, server_connection, api_client, load_test_data):
    """Test bulk translation of multiple texts from different languages to a single target language."""
    # Find working translation endpoint
    endpoints = [f"{server_url}{TRANSLATE_ENDPOINT}", f"{server_url}{ALT_TRANSLATE_ENDPOINT}"]
    working_endpoint = None
    
    for endpoint in endpoints:
        try:
            data = {
                "text": "Hello",
                "source_language": "en",
                "target_language": "es"
            }
            
            async with api_client.post(endpoint, json=data) as response:
                if response.status == 200:
                    working_endpoint = endpoint
                    break
        except:
            continue
    
    if not working_endpoint:
        pytest.fail("No working translation endpoint found")
    
    # Load multilingual samples
    samples = load_test_data("multilingual_samples.json")
    
    # Select various languages to translate
    languages_to_translate = ["english", "spanish", "french", "german", "italian"]
    target_language = "en"  # Translate everything to English
    
    # Create batch of translation tasks
    translation_tasks = []
    for language in languages_to_translate:
        source_text = samples[language]["simple"]
        source_language = samples[language]["expected_language"]
        
        # Skip English to English translation
        if source_language == target_language:
            continue
        
        task = {
            "language": language,
            "source_text": source_text,
            "source_language": source_language,
            "target_language": target_language
        }
        translation_tasks.append(task)
    
    # Execute all translation tasks
    print("\nBulk translation workflow:")
    print(f"Translating {len(translation_tasks)} texts to {target_language}")
    
    results = {}
    start_time = time.time()
    
    for i, task in enumerate(translation_tasks):
        language = task["language"]
        source_text = task["source_text"]
        source_language = task["source_language"]
        
        print(f"\nTranslating {language} ({source_language}) to {target_language}:")
        print(f"Source: {source_text}")
        
        data = {
            "text": source_text,
            "source_language": source_language,
            "target_language": target_language
        }
        
        async with api_client.post(working_endpoint, json=data) as response:
            assert response.status == 200, f"Translation failed for {language} sample"
            
            result = await response.json()
            
            # Extract translated text
            translated_text = None
            if isinstance(result, dict):
                if "translated_text" in result:
                    translated_text = result.get("translated_text")
                elif "data" in result and isinstance(result["data"], dict):
                    result_data = result["data"]
                    if "translated_text" in result_data:
                        translated_text = result_data.get("translated_text")
            
            assert translated_text is not None, f"Could not extract translation for {language} sample"
            print(f"Translated: {translated_text}")
            
            # Store result
            results[language] = {
                "source": source_text,
                "source_language": source_language,
                "translated": translated_text,
                "target_language": target_language
            }
        
        # Add a small delay to avoid rate limiting
        if i < len(translation_tasks) - 1:
            await asyncio.sleep(0.5)
    
    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_translation = total_time / len(translation_tasks)
    
    # Print timing information
    print("\nBulk translation performance:")
    print(f"Total time: {total_time:.2f} seconds for {len(translation_tasks)} translations")
    print(f"Average time per translation: {avg_time_per_translation:.2f} seconds")
    
    # Verify all translations were completed
    assert len(results) == len(translation_tasks), "Not all translations were completed"
    
    # Verify all translations returned non-empty results
    assert all(result["translated"] for result in results.values()), "Some translations returned empty results"