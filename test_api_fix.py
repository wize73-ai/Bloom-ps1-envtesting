"""
Test script to verify our fix in the pipeline.py API file.

This script simulates the rule-based simplification code that was fixed in the API endpoint.
"""
import os
import sys
from app.core.pipeline.simplifier import SimplificationPipeline

def main():
    # Create a dummy model manager
    class DummyModelManager:
        async def load_model(self, *args, **kwargs):
            pass
        
        async def run_model(self, *args, **kwargs):
            return {"result": "dummy result"}
    
    # Simulate the API endpoint code
    try:
        # Import the rule-based simplification function
        # Using sys module directly to verify it's accessible
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        # Create a temporary simplification pipeline
        simplifier = SimplificationPipeline(DummyModelManager())
        
        # Determine if legal domain based on options or parameters
        is_legal_domain = True  # Set to True to test legal domain handling
        
        # Simulate the text request
        text = "The plaintiff alleged that the defendant had violated numerous provisions of the aforementioned contractual agreement."
        level = 4  # Simple level
        language = "en"
        
        # Apply rule-based simplification
        domain = "legal" if is_legal_domain else None
        rule_based_text = simplifier._rule_based_simplify(
            text, level, language, domain
        )
        
        print("Original:", text)
        print("Simplified:", rule_based_text)
        
        assert rule_based_text != text, "Rule-based simplification should change the text"
        print("API simplification test successful!")
        
    except Exception as e:
        print(f"Error in API simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()