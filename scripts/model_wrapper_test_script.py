#!/usr/bin/env python3
"""
Direct test of model wrapper metrics functionality.
"""
import sys
import os
import json
from app.services.models.wrapper import ModelOutput, ModelInput, BaseModelWrapper

class MockModelWrapper(BaseModelWrapper):
    """Mock model wrapper for testing metrics"""
    
    def __init__(self):
        # Skip parent initialization
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.config = {}
        self._metrics = []  # Initialize metrics list
        
    def _preprocess(self, input_data):
        return {"text": input_data.text}
        
    def _run_inference(self, preprocessed):
        # Simulate model inference
        return {"result": f"Processed: {preprocessed['text']}"}
        
    def _postprocess(self, model_output, input_data):
        # Return ModelOutput with the processed result
        return ModelOutput(
            result=model_output["result"],
            metadata={
                "input_length": len(input_data.text),
                "output_length": len(model_output["result"])
            }
        )

def test_model_wrapper_metrics():
    """Test that ModelWrapper.process populates metrics fields"""
    print("\n===== Testing ModelWrapper Metrics =====")
    
    # Create mock wrapper and input
    wrapper = MockModelWrapper()
    input_data = ModelInput(text="Hello, world!")
    
    # Process the input
    output = wrapper.process(input_data)
    
    # Check the output
    print(f"Result: {output.result}")
    
    # Check metrics
    print("\nEnhanced Metrics:")
    print(f"- performance_metrics: {output.performance_metrics}")
    print(f"- memory_usage: {output.memory_usage}")
    print(f"- operation_cost: {output.operation_cost}")
    print(f"- accuracy_score: {output.accuracy_score}")
    print(f"- truth_score: {output.truth_score}")
    
    # Verify metrics are populated
    metrics_populated = (
        output.performance_metrics is not None or
        output.memory_usage is not None or
        output.operation_cost is not None
    )
    
    print(f"\nMetrics populated: {metrics_populated}")
    
    # Print full output for debugging
    print("\nFull Output:")
    print(json.dumps(output.__dict__, default=str, indent=2))
    
    return metrics_populated

def main():
    """Run the test"""
    print("Testing Model Wrapper Enhanced Metrics")
    print("=====================================")
    
    # Run test
    success = test_model_wrapper_metrics()
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"Model Wrapper Metrics: {'PASS' if success else 'FAIL'}")
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())