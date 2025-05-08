#!/usr/bin/env python3
"""
Test script for veracity checking in the translation model wrapper.

This script tests the veracity integration for Spanish to English translation.
"""

import os
import sys
import asyncio
import json
from pprint import pprint

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import from the application
from app.services.models.wrapper import ModelInput, TranslationModelWrapper
from app.audit.veracity import VeracityAuditor
from app.utils.config import load_config

class MockModel:
    """Mock translation model for testing."""
    
    def __init__(self, name="mock_mbart"):
        self.name_or_path = name
        self.config = type('obj', (object,), {
            '_name_or_path': name
        })
    
    def generate(self, **kwargs):
        """Mock generate method."""
        # Return model output for tokenizer to decode
        import torch
        # Return tensor with shape (1, 3) containing token IDs
        return torch.tensor([[101, 102, 103]])

class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.lang_code_to_id = {
            "en_XX": 2,
            "es_XX": 8
        }
    
    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512, **kwargs):
        """Mock tokenizer call."""
        import torch
        # Return dict with input_ids tensor of shape (batch_size, seq_len)
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
    
    def batch_decode(self, outputs, skip_special_tokens=True):
        """Mock batch_decode method."""
        # For Spanish to English test case
        if isinstance(outputs, list) and len(outputs) > 0 and outputs[0] == "test_spanish":
            return ["Hello, I am very happy to meet you today."]
        return ["Hello, I am very happy to meet you today."]

async def test_direct_veracity():
    """Test direct integration of veracity with the wrapper."""
    print(f"\n{BOLD}{BLUE}Testing Direct Veracity Integration with TranslationModelWrapper{ENDC}")
    print("-" * 80)
    
    # Create mock objects
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Create veracity auditor
    config = load_config()
    veracity_auditor = VeracityAuditor(config=config)
    await veracity_auditor.initialize()
    
    # Create wrapper with veracity auditor
    wrapper = TranslationModelWrapper(
        model=model,
        tokenizer=tokenizer,
        config={"device": "cpu", "max_length": 128}
    )
    
    # Set veracity checker
    wrapper.veracity_checker = veracity_auditor
    
    # Test input in Spanish
    spanish_text = "Hola, estoy muy feliz de conocerte hoy."
    input_data = ModelInput(
        text=spanish_text,
        source_language="es",
        target_language="en"
    )
    
    # Run processing with veracity checking
    try:
        print(f"\n{BOLD}Processing input with veracity checking...{ENDC}")
        print(f"{BOLD}Original Text:{ENDC} {spanish_text}")
        result = await wrapper.process(input_data)
        
        # Print the result
        print(f"\n{BOLD}Processing completed!{ENDC}")
        if hasattr(result, 'result'):
            print(f"{BOLD}Translation:{ENDC} {result.result}")
        else:
            print(f"{BOLD}Translation:{ENDC} {result}")
        
        # Check for veracity data
        if hasattr(result, 'metadata') and 'veracity' in result.metadata:
            print(f"\n{BOLD}{BLUE}Veracity data found:{ENDC}")
            veracity_data = result.metadata['veracity']
            verified = veracity_data.get('verified', False)
            status = f"{GREEN}Verified{ENDC}" if verified else f"{RED}Not verified{ENDC}"
            print(f"{BOLD}Verification:{ENDC} {status}")
            print(f"{BOLD}Score:{ENDC} {veracity_data.get('score', 0.0)}")
            print(f"{BOLD}Confidence:{ENDC} {veracity_data.get('confidence', 0.0)}")
            
            if 'issues' in veracity_data and veracity_data['issues']:
                print(f"\n{BOLD}Issues found:{ENDC}")
                for issue in veracity_data['issues']:
                    severity = issue.get('severity', 'unknown')
                    if severity == 'critical':
                        severity_color = RED
                    elif severity == 'warning':
                        severity_color = YELLOW
                    else:
                        severity_color = ""
                    print(f"- {severity_color}{issue.get('type', 'unknown')}{ENDC}: {issue.get('message', 'No message')}")
            
            if 'metrics' in veracity_data:
                print(f"\n{BOLD}Veracity metrics:{ENDC}")
                for key, value in veracity_data['metrics'].items():
                    print(f"- {key}: {value}")
        else:
            print(f"\n{YELLOW}No veracity data found in result{ENDC}")
        
        # Check for translation quality metrics
        if hasattr(result, 'accuracy_score'):
            print(f"\n{BOLD}Accuracy score:{ENDC} {result.accuracy_score}")
        
        if hasattr(result, 'truth_score'):
            print(f"{BOLD}Truth score:{ENDC} {result.truth_score}")
            
        # Success
        print(f"\n{GREEN}✅ Veracity integration test completed successfully!{ENDC}")
    except Exception as e:
        print(f"\n{RED}❌ Error: {e}{ENDC}")
        import traceback
        traceback.print_exc()

async def test_api_veracity():
    """Test the translation endpoint with verification."""
    print(f"\n{BOLD}{BLUE}Testing API Veracity Integration{ENDC}")
    print("-" * 80)
    
    # Test server URL
    SERVER_URL = "http://localhost:8000"
    TRANSLATE_ENDPOINT = "/pipeline/translate"
    
    # Test translation
    text = "Hello, how are you doing today? I hope you are well."
    
    print(f"{BOLD}Original Text:{ENDC} {text}")
    print("-" * 80)
    
    # Import requests
    import requests
    
    # Test with different target languages
    languages = ["es", "fr", "de"]
    
    for target_lang in languages:
        print(f"\n{BOLD}Translating to {target_lang.upper()}:{ENDC}")
        
        # Prepare request with verification
        data = {
            "text": text,
            "source_language": "en",
            "target_language": target_lang,
            "verify": True
        }
        
        # Make request
        try:
            response = requests.post(
                f"{SERVER_URL}{TRANSLATE_ENDPOINT}",
                json=data
            )
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                
                # Extract results
                translation = result["data"]["translated_text"]
                model_used = result["data"]["model_used"]
                verified = result["data"].get("verified", False)
                verification_score = result["data"].get("verification_score", None)
                
                print(f"{GREEN}Model: {model_used}{ENDC}")
                print(f"Translation: {translation}")
                
                # Print verification results
                if verified is not None:
                    status = f"{GREEN}Verified{ENDC}" if verified else f"{RED}Not verified{ENDC}"
                    print(f"Verification: {status}")
                    
                    if verification_score is not None:
                        print(f"Verification Score: {verification_score}")
            else:
                print(f"{RED}Request failed: {response.status_code}{ENDC}")
                print(response.text)
        except Exception as e:
            print(f"{RED}Error: {str(e)}{ENDC}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run API test
        asyncio.run(test_api_veracity())
    else:
        # Run direct test
        asyncio.run(test_direct_veracity())