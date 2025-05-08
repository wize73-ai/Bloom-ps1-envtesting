"""
Test script to verify our fix for the SimplificationPipeline.
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
    
    # Create a simplification pipeline instance
    simplifier = SimplificationPipeline(DummyModelManager())
    
    # Test the rule_based_simplify method
    text = "The plaintiff alleged that the defendant had violated numerous provisions of the aforementioned contractual agreement."
    result = simplifier._rule_based_simplify(text, level=4, language="en", domain="legal")
    
    print("Original:", text)
    print("Simplified:", result)
    
    assert result != text, "Rule-based simplification should change the text"
    print("Rule-based simplification test successful!")

if __name__ == "__main__":
    main()