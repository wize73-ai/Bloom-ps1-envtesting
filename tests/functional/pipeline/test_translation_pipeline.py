"""
Integration tests for the translation pipeline.

These tests focus on testing the translation pipeline components directly,
rather than through the API endpoints.
"""
import os
import sys
import pytest
import asyncio
from pathlib import Path
import gc
import time
from unittest import mock

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Set test mode environment variable
os.environ["CASALINGUA_ENV"] = "test"

@pytest.fixture(scope="module")
def load_translator():
    """Load a translator model for testing."""
    try:
        # Import the necessary components
        from app.core.pipeline.translator import Translator
        from app.services.models.loader import ModelLoader
        from app.services.models.manager import EnhancedModelManager
        from app.utils.config import load_config
        
        # Load the configuration
        config = load_config()
        
        # Initialize the model loader and manager
        model_loader = ModelLoader(config=config)
        model_manager = EnhancedModelManager(model_loader, {}, config)
        
        # Create a translator instance
        translator = Translator(model_manager, config)
        
        # Return the initialized translator
        return translator
    except (ImportError, Exception) as e:
        pytest.skip(f"Failed to load translator components: {str(e)}")
        return None

@pytest.fixture(scope="module")
def load_processor():
    """Load the unified processor for testing."""
    try:
        # Import the necessary components
        from app.core.pipeline.processor import UnifiedProcessor
        from app.services.models.loader import ModelLoader
        from app.services.models.manager import EnhancedModelManager
        from app.utils.config import load_config
        
        # Load the configuration
        config = load_config()
        
        # Initialize the model loader and manager
        model_loader = ModelLoader(config=config)
        model_manager = EnhancedModelManager(model_loader, {}, config)
        
        # Create a processor instance
        processor = UnifiedProcessor(model_manager, config=config)
        
        # Return the initialized processor
        return processor
    except (ImportError, Exception) as e:
        pytest.skip(f"Failed to load processor components: {str(e)}")
        return None

@pytest.mark.asyncio
async def test_translator_initialization(load_translator):
    """Test that the translator can be initialized."""
    translator = load_translator
    
    if translator is None:
        pytest.skip("Translator initialization failed, skipping test")
    
    # Basic verification that the translator is initialized
    assert hasattr(translator, "translate"), "Translator missing translate method"
    assert hasattr(translator, "model_manager"), "Translator missing model_manager"
    assert hasattr(translator, "config"), "Translator missing config"

@pytest.mark.asyncio
async def test_direct_translation(load_translator):
    """Test direct translation using the Translator class."""
    translator = load_translator
    
    if translator is None:
        pytest.skip("Translator initialization failed, skipping test")
    
    # Define test cases
    test_cases = [
        {
            "text": "Hello, how are you?",
            "source_language": "en",
            "target_language": "es",
            "expected_contains": ["Hola", "cómo", "estás"]
        },
        {
            "text": "The weather is nice today.",
            "source_language": "en",
            "target_language": "fr",
            "expected_contains": ["temps", "beau", "aujourd"]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        source_language = test_case["source_language"]
        target_language = test_case["target_language"]
        expected_contains = test_case["expected_contains"]
        
        try:
            print(f"\nTest case {i+1}: Translating from {source_language} to {target_language}")
            print(f"Text: {text}")
            
            # Execute translation
            start_time = time.time()
            result = await translator.translate(
                text=text,
                source_language=source_language,
                target_language=target_language
            )
            duration = time.time() - start_time
            
            # Verify the result
            assert isinstance(result, dict), "Translation result should be a dictionary"
            assert "translated_text" in result, "Result missing 'translated_text' field"
            
            translated_text = result["translated_text"]
            print(f"Translation: {translated_text}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Check that the translation contains expected words
            matches = [word.lower() in translated_text.lower() for word in expected_contains]
            match_rate = sum(matches) / len(expected_contains)
            
            # Print match information
            print("Expected words check:")
            for word, matched in zip(expected_contains, matches):
                match_mark = "✓" if matched else "✗"
                print(f"  '{word}': {match_mark}")
            print(f"Match rate: {match_rate:.0%}")
            
            # Assert that at least 70% of expected words are found
            # (allow for some variation in translations)
            assert match_rate >= 0.7, f"Translation doesn't contain enough expected words: {translated_text}"
                
        except Exception as e:
            pytest.fail(f"Translation failed: {str(e)}")
            
        # Clean up between test cases
        gc.collect()
        await asyncio.sleep(1)  # Give the system time to release resources

@pytest.mark.asyncio
async def test_processor_translation(load_processor):
    """Test translation through the UnifiedProcessor."""
    processor = load_processor
    
    if processor is None:
        pytest.skip("Processor initialization failed, skipping test")
    
    # Ensure processor is initialized
    if not hasattr(processor, "_initialized") or not processor._initialized:
        await processor.initialize()
    
    # Define test cases
    test_cases = [
        {
            "text": "Hello, this is a test message.",
            "source_language": "en",
            "target_language": "es",
            "expected_contains": ["Hola", "mensaje", "prueba"]
        },
        {
            "text": "Please translate this sentence to German.",
            "source_language": "en",
            "target_language": "de",
            "expected_contains": ["bitte", "übersetzen", "Satz"]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        source_language = test_case["source_language"]
        target_language = test_case["target_language"]
        expected_contains = test_case["expected_contains"]
        
        try:
            print(f"\nProcessor test case {i+1}: Translating from {source_language} to {target_language}")
            print(f"Text: {text}")
            
            # Execute translation through processor
            start_time = time.time()
            result = await processor.translate(
                text=text,
                source_language=source_language,
                target_language=target_language
            )
            duration = time.time() - start_time
            
            # Verify the result
            assert isinstance(result, dict), "Processor translation result should be a dictionary"
            
            # Result might be in different formats depending on processor implementation
            # Try to extract the translated text
            translated_text = None
            if "translated_text" in result:
                translated_text = result["translated_text"]
            elif "result" in result:
                translated_text = result["result"]
            elif "text" in result:
                translated_text = result["text"]
            
            assert translated_text is not None, "Could not extract translated text from result"
            
            print(f"Translation: {translated_text}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Check that the translation contains expected words
            matches = [word.lower() in translated_text.lower() for word in expected_contains]
            match_rate = sum(matches) / len(expected_contains)
            
            # Print match information
            print("Expected words check:")
            for word, matched in zip(expected_contains, matches):
                match_mark = "✓" if matched else "✗"
                print(f"  '{word}': {match_mark}")
            print(f"Match rate: {match_rate:.0%}")
            
            # Assert that at least 70% of expected words are found
            assert match_rate >= 0.7, f"Processor translation doesn't contain enough expected words: {translated_text}"
                
        except Exception as e:
            pytest.fail(f"Processor translation failed: {str(e)}")
            
        # Clean up between test cases
        gc.collect()
        await asyncio.sleep(1)  # Give the system time to release resources

@pytest.mark.asyncio
async def test_language_detection_through_processor(load_processor):
    """Test language detection through the UnifiedProcessor."""
    processor = load_processor
    
    if processor is None:
        pytest.skip("Processor initialization failed, skipping test")
    
    # Ensure processor is initialized
    if not hasattr(processor, "_initialized") or not processor._initialized:
        await processor.initialize()
    
    # Define test cases
    test_cases = [
        {
            "text": "Hello, how are you today? I hope you're doing well.",
            "expected_language": "en"
        },
        {
            "text": "Hola, ¿cómo estás hoy? Espero que estés bien.",
            "expected_language": "es"
        },
        {
            "text": "Bonjour, comment allez-vous aujourd'hui? J'espère que vous allez bien.",
            "expected_language": "fr"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        expected_language = test_case["expected_language"]
        
        try:
            print(f"\nProcessor test case {i+1}: Detecting language")
            print(f"Text: {text}")
            
            # Execute language detection through processor
            start_time = time.time()
            result = await processor.detect_language(text=text)
            duration = time.time() - start_time
            
            # Verify the result
            assert isinstance(result, dict), "Processor language detection result should be a dictionary"
            
            # Result might be in different formats depending on processor implementation
            # Try to extract the detected language
            detected_language = None
            if "language" in result:
                detected_language = result["language"]
            elif "detected_language" in result:
                detected_language = result["detected_language"]
            elif "result" in result and "language" in result["result"]:
                detected_language = result["result"]["language"]
            
            assert detected_language is not None, "Could not extract detected language from result"
            
            print(f"Detected language: {detected_language}")
            print(f"Expected language: {expected_language}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Check if the detected language matches the expected language
            is_correct = detected_language.lower() == expected_language.lower()
            match_mark = "✓" if is_correct else "✗"
            print(f"Correct detection: {match_mark}")
            
            # Assert that the language is correctly detected
            assert is_correct, f"Language detection failed: expected {expected_language}, got {detected_language}"
                
        except Exception as e:
            pytest.fail(f"Processor language detection failed: {str(e)}")
            
        # Clean up between test cases
        gc.collect()
        await asyncio.sleep(1)  # Give the system time to release resources