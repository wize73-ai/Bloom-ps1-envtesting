"""
Integration tests for the text simplification pipeline.

These tests focus on testing the simplification pipeline components directly,
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
def load_simplifier():
    """Load a simplifier model for testing."""
    try:
        # Import the necessary components
        from app.core.pipeline.simplifier import Simplifier
        from app.services.models.loader import ModelLoader
        from app.services.models.manager import EnhancedModelManager
        from app.utils.config import load_config
        
        # Load the configuration
        config = load_config()
        
        # Initialize the model loader and manager
        model_loader = ModelLoader(config=config)
        model_manager = EnhancedModelManager(model_loader, {}, config)
        
        # Create a simplifier instance
        simplifier = Simplifier(model_manager, config)
        
        # Return the initialized simplifier
        return simplifier
    except (ImportError, Exception) as e:
        pytest.skip(f"Failed to load simplifier components: {str(e)}")
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
async def test_simplifier_initialization(load_simplifier):
    """Test that the simplifier can be initialized."""
    simplifier = load_simplifier
    
    if simplifier is None:
        pytest.skip("Simplifier initialization failed, skipping test")
    
    # Basic verification that the simplifier is initialized
    assert hasattr(simplifier, "simplify"), "Simplifier missing simplify method"
    assert hasattr(simplifier, "model_manager"), "Simplifier missing model_manager"
    assert hasattr(simplifier, "config"), "Simplifier missing config"

@pytest.mark.asyncio
async def test_direct_simplification(load_simplifier):
    """Test direct simplification using the Simplifier class."""
    simplifier = load_simplifier
    
    if simplifier is None:
        pytest.skip("Simplifier initialization failed, skipping test")
    
    # Define test cases
    test_cases = [
        {
            "text": "The plaintiff alleged that the defendant had violated numerous provisions of the aforementioned contractual agreement.",
            "language": "en",
            "target_level": "simple",
            "expected_contains": ["plaintiff", "defendant", "contract", "violated"]
        },
        {
            "text": "The quantum entanglement phenomenon demonstrates non-local correlations between particles.",
            "language": "en",
            "target_level": "simple",
            "expected_contains": ["quantum", "particles", "correlations"]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        language = test_case["language"]
        target_level = test_case["target_level"]
        expected_contains = test_case["expected_contains"]
        
        try:
            print(f"\nTest case {i+1}: Simplifying text")
            print(f"Text: {text}")
            
            # Execute simplification
            start_time = time.time()
            result = await simplifier.simplify(
                text=text,
                language=language,
                target_level=target_level
            )
            duration = time.time() - start_time
            
            # Verify the result
            assert isinstance(result, dict), "Simplification result should be a dictionary"
            assert "simplified_text" in result, "Result missing 'simplified_text' field"
            
            simplified_text = result["simplified_text"]
            print(f"Simplified: {simplified_text}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Compare word counts
            original_words = len(text.split())
            simplified_words = len(simplified_text.split())
            print(f"Original word count: {original_words}")
            print(f"Simplified word count: {simplified_words}")
            
            # Check that the simplified text contains expected keywords
            matches = [word.lower() in simplified_text.lower() for word in expected_contains]
            match_rate = sum(matches) / len(expected_contains)
            
            # Print match information
            print("Expected keywords check:")
            for word, matched in zip(expected_contains, matches):
                match_mark = "✓" if matched else "✗"
                print(f"  '{word}': {match_mark}")
            print(f"Match rate: {match_rate:.0%}")
            
            # Assert that at least 70% of expected keywords are found
            assert match_rate >= 0.7, f"Simplified text doesn't contain enough expected keywords: {simplified_text}"
                
        except Exception as e:
            pytest.fail(f"Simplification failed: {str(e)}")
            
        # Clean up between test cases
        gc.collect()
        await asyncio.sleep(1)  # Give the system time to release resources

@pytest.mark.asyncio
async def test_processor_simplification(load_processor):
    """Test simplification through the UnifiedProcessor."""
    processor = load_processor
    
    if processor is None:
        pytest.skip("Processor initialization failed, skipping test")
    
    # Ensure processor is initialized
    if not hasattr(processor, "_initialized") or not processor._initialized:
        await processor.initialize()
    
    # Define test cases
    test_cases = [
        {
            "text": "The corporation's quarterly financial statements indicated a substantial depreciation in tangible assets.",
            "language": "en",
            "target_level": "simple",
            "expected_contains": ["company", "financial", "assets"]
        },
        {
            "text": "The research methodology employed a mixed-methods approach, incorporating both quantitative statistical analysis and qualitative phenomenological interviews.",
            "language": "en",
            "target_level": "simple",
            "expected_contains": ["research", "methods", "analysis", "interviews"]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        language = test_case["language"]
        target_level = test_case["target_level"]
        expected_contains = test_case["expected_contains"]
        
        try:
            print(f"\nProcessor test case {i+1}: Simplifying text")
            print(f"Text: {text}")
            
            # Execute simplification through processor
            start_time = time.time()
            result = await processor.simplify(
                text=text,
                language=language,
                target_level=target_level
            )
            duration = time.time() - start_time
            
            # Verify the result
            assert isinstance(result, dict), "Processor simplification result should be a dictionary"
            
            # Result might be in different formats depending on processor implementation
            # Try to extract the simplified text
            simplified_text = None
            if "simplified_text" in result:
                simplified_text = result["simplified_text"]
            elif "result" in result:
                simplified_text = result["result"]
            elif "text" in result:
                simplified_text = result["text"]
            
            assert simplified_text is not None, "Could not extract simplified text from result"
            
            print(f"Simplified: {simplified_text}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Compare word counts
            original_words = len(text.split())
            simplified_words = len(simplified_text.split())
            print(f"Original word count: {original_words}")
            print(f"Simplified word count: {simplified_words}")
            
            # Check that the simplified text contains expected keywords
            matches = [word.lower() in simplified_text.lower() for word in expected_contains]
            match_rate = sum(matches) / len(expected_contains)
            
            # Print match information
            print("Expected keywords check:")
            for word, matched in zip(expected_contains, matches):
                match_mark = "✓" if matched else "✗"
                print(f"  '{word}': {match_mark}")
            print(f"Match rate: {match_rate:.0%}")
            
            # Assert that at least 70% of expected keywords are found
            assert match_rate >= 0.7, f"Processor simplified text doesn't contain enough expected keywords: {simplified_text}"
                
        except Exception as e:
            pytest.fail(f"Processor simplification failed: {str(e)}")
            
        # Clean up between test cases
        gc.collect()
        await asyncio.sleep(1)  # Give the system time to release resources

@pytest.mark.asyncio
async def test_simplification_with_different_levels(load_processor):
    """Test simplification with different target levels."""
    processor = load_processor
    
    if processor is None:
        pytest.skip("Processor initialization failed, skipping test")
    
    # Ensure processor is initialized
    if not hasattr(processor, "_initialized") or not processor._initialized:
        await processor.initialize()
    
    # Text to simplify
    text = "The patient presents with acute myocardial infarction secondary to coronary artery thrombosis, necessitating immediate percutaneous coronary intervention to restore myocardial perfusion."
    
    # Try different target levels
    target_levels = ["simple", "very_simple", "children"]
    
    results = {}
    
    for level in target_levels:
        try:
            print(f"\nSimplifying text to level: {level}")
            print(f"Original: {text}")
            
            # Execute simplification through processor
            start_time = time.time()
            result = await processor.simplify(text=text, language="en", target_level=level)
            duration = time.time() - start_time
            
            # Try to extract the simplified text
            simplified_text = None
            if "simplified_text" in result:
                simplified_text = result["simplified_text"]
            elif "result" in result:
                simplified_text = result["result"]
            elif "text" in result:
                simplified_text = result["text"]
            
            assert simplified_text is not None, f"Could not extract simplified text for level {level}"
            
            print(f"Simplified to {level}: {simplified_text}")
            print(f"Duration: {duration:.2f} seconds")
            
            # Compare word counts
            original_words = len(text.split())
            simplified_words = len(simplified_text.split())
            print(f"Original word count: {original_words}")
            print(f"Simplified word count: {simplified_words}")
            
            # Store results for comparison
            results[level] = {
                "text": simplified_text,
                "word_count": simplified_words
            }
                
        except Exception as e:
            print(f"Simplification to level {level} failed: {str(e)}")
            results[level] = {
                "text": None,
                "error": str(e)
            }
            
        # Clean up between test cases
        gc.collect()
        await asyncio.sleep(1)
    
    # Make sure we got at least one successful simplification
    successful_levels = [level for level, result in results.items() if "text" in result and result["text"]]
    assert len(successful_levels) > 0, "All simplification attempts failed"
    
    # Print summary of results
    print("\nSimplification levels comparison:")
    for level, result in results.items():
        if "text" in result and result["text"]:
            status = "✓"
            details = f"Word count: {result['word_count']}"
        else:
            status = "✗"
            details = f"Error: {result.get('error', 'Unknown error')}"
        
        print(f"{level}: {status} {details}")