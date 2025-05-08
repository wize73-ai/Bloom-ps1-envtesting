#!/usr/bin/env python3
"""
Test script to verify the MBART tokenizer fix.

This script tests the fix for MBART tokenizer warnings by:
1. Importing the TranslationModelWrapper class
2. Creating a mock tokenizer that doesn't support src_lang
3. Testing that the patched _preprocess method handles it without warnings
"""

import sys
import logging
import inspect
from unittest.mock import MagicMock, patch
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_wrapper_module_path():
    """Find the path to the module containing the TranslationModelWrapper class."""
    # Common locations to check
    potential_paths = [
        project_root / "app" / "services" / "models" / "wrapper.py",
        project_root / "app" / "models" / "wrapper.py",
        project_root / "models" / "wrapper.py",
    ]
    
    # Check each potential path
    for path in potential_paths:
        if path.exists():
            return path
    
    # If not found in the common locations, search the entire project
    logger.info("Searching for wrapper.py file in the project...")
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file == "wrapper.py":
                path = Path(os.path.join(root, file))
                if "TranslationModelWrapper" in path.read_text():
                    return path
    
    return None

def test_tokenizer_without_src_lang():
    """Test the _preprocess method with a tokenizer that doesn't support src_lang."""
    # Find and import the wrapper module
    wrapper_path = find_wrapper_module_path()
    if not wrapper_path:
        logger.error("Could not find the wrapper module containing TranslationModelWrapper.")
        return False
    
    logger.info(f"Found wrapper module at: {wrapper_path}")
    
    # Dynamic import of the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    wrapper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper_module)
    
    # Check if the module has TranslationModelWrapper
    if not hasattr(wrapper_module, "TranslationModelWrapper"):
        logger.error("TranslationModelWrapper class not found in the loaded module.")
        return False
    
    # Create a test instance of TranslationModelWrapper
    with patch.object(wrapper_module.TranslationModelWrapper, '__init__', return_value=None):
        wrapper = wrapper_module.TranslationModelWrapper()
        
        # Create a mock tokenizer that doesn't support src_lang
        mock_tokenizer = MagicMock()
        
        # Setup the call behavior to return a mock inputs object
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        
        # Remove any src_lang attribute from the tokenizer
        if hasattr(mock_tokenizer, 'src_lang'):
            delattr(mock_tokenizer, 'src_lang')
            
        if hasattr(mock_tokenizer, 'set_src_lang_special_tokens'):
            delattr(mock_tokenizer, 'set_src_lang_special_tokens')
        
        wrapper.tokenizer = mock_tokenizer
        wrapper.config = {"max_length": 512}
        
        # Create a patch for inspect.signature to return a signature without src_lang
        mock_signature = MagicMock()
        mock_signature.parameters = {'texts': MagicMock(), 'padding': MagicMock(), 'truncation': MagicMock()}
        
        with patch('inspect.signature', return_value=mock_signature):
            # Configure a log capture to check for warnings
            with patch('logging.Logger.warning') as mock_warning:
                # Call the _preprocess method
                inputs = wrapper._preprocess(["Test text"], source_lang="en")
                
                # Check that the tokenizer was called
                mock_tokenizer.assert_called_once()
                
                # Check that no warning was logged
                if mock_warning.called:
                    logger.error("❌ Warning was still logged despite the fix!")
                    for call in mock_warning.call_args_list:
                        logger.error(f"Warning: {call}")
                    return False
                
                # Check that the src_lang parameter was not used (since our mock doesn't support it)
                call_args = mock_tokenizer.call_args[1]
                if 'src_lang' in call_args:
                    logger.error("❌ The src_lang parameter was used despite the tokenizer not supporting it!")
                    return False
                
                logger.info("✅ The tokenizer was called correctly without src_lang and no warnings were logged!")
                return True

def test_tokenizer_with_src_lang():
    """Test the _preprocess method with a tokenizer that supports src_lang."""
    # Find and import the wrapper module
    wrapper_path = find_wrapper_module_path()
    if not wrapper_path:
        logger.error("Could not find the wrapper module containing TranslationModelWrapper.")
        return False
    
    # Dynamic import of the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("wrapper", wrapper_path)
    wrapper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper_module)
    
    # Create a test instance of TranslationModelWrapper
    with patch.object(wrapper_module.TranslationModelWrapper, '__init__', return_value=None):
        wrapper = wrapper_module.TranslationModelWrapper()
        
        # Create a mock tokenizer that supports src_lang
        mock_tokenizer = MagicMock()
        
        # Setup the call behavior to return a mock inputs object
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        
        # Add src_lang attribute to the tokenizer to simulate support
        mock_tokenizer.src_lang = "en"
        
        wrapper.tokenizer = mock_tokenizer
        wrapper.config = {"max_length": 512}
        
        # Create a patch for inspect.signature to return a signature with src_lang
        mock_signature = MagicMock()
        mock_signature.parameters = {
            'texts': MagicMock(), 
            'padding': MagicMock(), 
            'truncation': MagicMock(),
            'src_lang': MagicMock()
        }
        
        with patch('inspect.signature', return_value=mock_signature):
            # Call the _preprocess method
            inputs = wrapper._preprocess(["Test text"], source_lang="en")
            
            # Check that the tokenizer was called
            mock_tokenizer.assert_called_once()
            
            # Check that the src_lang parameter was used
            call_args = mock_tokenizer.call_args[1]
            if 'src_lang' not in call_args:
                logger.error("❌ The src_lang parameter was not used despite the tokenizer supporting it!")
                logger.error(f"Call args were: {call_args}")
                return False
            
            if call_args.get('src_lang') != "en":
                logger.error(f"❌ The src_lang parameter has incorrect value: {call_args.get('src_lang')}")
                return False
            
            logger.info("✅ The tokenizer was correctly called with src_lang!")
            return True

def main():
    logger.info("Testing MBART tokenizer fix...")
    
    # Run tests
    success = True
    
    logger.info("Test 1: Tokenizer without src_lang support")
    if not test_tokenizer_without_src_lang():
        success = False
    
    logger.info("Test 2: Tokenizer with src_lang support")
    if not test_tokenizer_with_src_lang():
        success = False
    
    if success:
        logger.info("✅ All tests passed! The MBART tokenizer fix is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed! The MBART tokenizer fix may not be working correctly.")
        return 1

if __name__ == "__main__":
    sys.exit(main())