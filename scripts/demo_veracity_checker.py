"""
Demo script for the VeracityAuditor and Model Wrapper integration.

This script demonstrates how the VeracityAuditor works with model wrappers
to verify and assess the quality of model outputs.

Usage:
    python demo_veracity_checker.py

Author: Exygy Development Team
"""

import asyncio
import logging
import sys
import json
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add local app directory to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.audit.veracity import VeracityAuditor
from app.services.models.wrapper_base import ModelInput, ModelOutput, VeracityMetrics, BaseModelWrapper

# Example mock model wrapper for demonstration
class MockTranslationWrapper(BaseModelWrapper):
    """Mock translation model wrapper for demonstration."""
    
    def __init__(self):
        """Initialize with mock model and tokenizer."""
        # Call the parent class's init with our mock model and tokenizer
        super().__init__(
            model="mock_model",
            tokenizer="mock_tokenizer",
            config={}
        )
        
        # Create the veracity auditor
        from app.audit.veracity import VeracityAuditor
        self.veracity_checker = VeracityAuditor()
        
        # Initialize hooks
        self.pre_process_hook = None
        self.post_process_hook = None
    
    async def initialize(self):
        """Initialize the mock wrapper and veracity checker."""
        await self.veracity_checker.initialize()
        logger.info("Mock translation wrapper initialized")
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess the input data."""
        return {
            "text": input_data.text,
            "source_lang": input_data.source_language,
            "target_lang": input_data.target_language
        }
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> str:
        """Run mock translation inference."""
        text = preprocessed["text"]
        source_lang = preprocessed["source_lang"]
        target_lang = preprocessed["target_lang"]
        
        # Simple mock translations
        translations = {
            ("en", "es"): {
                "Hello": "Hola",
                "How are you?": "¿Cómo estás?",
                "I have 5 apples": "Tengo manzanas",  # Missing number to trigger veracity issue
                "I like to read books": "Me gusta leer libros",
                "The cost is $500": "El costo es $"  # Missing number to trigger veracity issue
            },
            ("es", "en"): {
                "Hola": "Hello",
                "¿Cómo estás?": "How are you?",
                "Tengo 5 manzanas": "I have 5 apples",
                "Me gusta leer libros": "I like to read books"
            },
            ("en", "fr"): {
                "Hello": "Bonjour",
                "How are you?": "Comment allez-vous?",
                "I have 5 apples": "J'ai 5 pommes",
                "I like to read books": "J'aime lire des livres"
            }
        }
        
        key = (source_lang, target_lang)
        
        if key in translations and text in translations[key]:
            return translations[key][text]
        
        # Default mock response for unknown inputs
        if target_lang == "es":
            return f"[ES] {text}"
        elif target_lang == "fr":
            return f"[FR] {text}"
        else:
            return f"[{target_lang}] {text}"
    
    def _postprocess(self, model_output: str, input_data: ModelInput) -> ModelOutput:
        """Postprocess the model output."""
        return ModelOutput(
            result=model_output,
            metadata={
                "source_language": input_data.source_language,
                "target_language": input_data.target_language,
                "model_type": "mock_translation"
            }
        )

# Example test cases
TEST_INPUTS = [
    {
        "text": "Hello",
        "source_language": "en",
        "target_language": "es",
        "description": "Simple greeting - should pass veracity checks"
    },
    {
        "text": "I have 5 apples",
        "source_language": "en",
        "target_language": "es",
        "description": "Text with numbers - will fail veracity checks due to missing number in translation"
    },
    {
        "text": "The cost is $500",
        "source_language": "en",
        "target_language": "es",
        "description": "Text with currency - will fail veracity checks due to missing amount"
    },
    {
        "text": "I like to read books",
        "source_language": "en",
        "target_language": "es",
        "description": "Simple text - should pass veracity checks"
    },
    {
        "text": "Hello",
        "source_language": "en",
        "target_language": "fr",
        "description": "French translation - should pass veracity checks"
    }
]

async def run_demo():
    """Run the veracity checker demonstration."""
    logger.info("Starting veracity checker demo")
    
    # Create and initialize the mock wrapper
    wrapper = MockTranslationWrapper()
    await wrapper.initialize()
    
    results = []
    
    for test_case in TEST_INPUTS:
        logger.info(f"Processing: {test_case['description']}")
        
        # Convert to ModelInput
        input_data = ModelInput(
            text=test_case["text"],
            source_language=test_case["source_language"],
            target_language=test_case["target_language"]
        )
        
        # Process the input
        result = await wrapper.process_async(input_data)
        
        # Log results
        logger.info(f"Input: {input_data.text}")
        logger.info(f"Output: {result['result']}")
        
        if result.get("metadata", {}).get("veracity"):
            veracity = result["metadata"]["veracity"]
            logger.info(f"Veracity score: {veracity['score']:.2f}")
            logger.info(f"Confidence: {veracity['confidence']:.2f}")
            
            # Format warnings and failures
            if veracity.get("warnings"):
                logger.warning(f"Warnings: {', '.join(veracity['warnings'])}")
            if veracity.get("checks_failed"):
                logger.error(f"Failed checks: {', '.join(veracity['checks_failed'])}")
        else:
            logger.warning("No veracity data available")
        
        logger.info("-" * 50)
        
        # Add to results
        results.append({
            "input": test_case,
            "output": result.get("result"),
            "veracity_score": result.get("veracity_score"),
            "veracity_data": result.get("metadata", {}).get("veracity")
        })
    
    # Generate summary
    logger.info("Veracity Check Summary:")
    logger.info("-" * 50)
    
    for i, result in enumerate(results):
        veracity_data = result["veracity_data"]
        if veracity_data:
            score = veracity_data["score"]
            checks_passed = len(veracity_data["checks_passed"])
            checks_failed = len(veracity_data["checks_failed"])
            warnings_count = len(veracity_data["warnings"])
            
            status = "PASSED" if score > 0.7 and checks_failed == 0 else "FAILED"
            logger.info(f"Test {i+1}: {status} - Score: {score:.2f}, Passed: {checks_passed}, Failed: {checks_failed}, Warnings: {warnings_count}")
    
    # Save results to a file
    with open("veracity_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to veracity_demo_results.json")

if __name__ == "__main__":
    asyncio.run(run_demo())