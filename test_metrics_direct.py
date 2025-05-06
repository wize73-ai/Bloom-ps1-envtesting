#!/usr/bin/env python3
"""
Direct test of the enhanced metrics features in ModelOutput.
"""
import sys
import time
from dataclasses import asdict
from app.services.models.wrapper import ModelOutput, BaseModelWrapper, ModelInput

class MockModelWrapper(BaseModelWrapper):
    """Mock model wrapper for testing"""
    
    def __init__(self):
        """Initialize mock wrapper"""
        self.model_name = "mock_model"
        self.model_info = {
            "name": "mock_model",
            "type": "mock",
            "parameters": 125000000,  # 125M
            "size_mb": 500
        }
    
    def _preprocess(self, input_data):
        """Mock preprocessing implementation"""
        # Just return the input as-is
        return input_data
    
    def _postprocess(self, result, input_data):
        """Mock postprocessing implementation"""
        # Just return the result as-is
        return result
    
    def _process_impl(self, input_data):
        """Mock implementation that simulates processing"""
        # Simulate work being done
        time.sleep(0.5)
        
        # Simulate output
        return f"Processed: {input_data.text}"
    
    def _get_accuracy_score(self, result):
        """Mock accuracy scorer"""
        return 0.95
    
    def _get_truth_score(self, result):
        """Mock truth scorer"""
        return 0.87

def test_model_wrapper_metrics():
    """Test that ModelWrapper.process populates metrics fields"""
    print("\n===== Testing ModelWrapper Metrics =====\n")
    
    # Create mock wrapper and input
    wrapper = MockModelWrapper()
    input_data = ModelInput(text="Hello, world!")
    
    # Process the input
    output = wrapper.process(input_data)
    
    # Print the output
    print(f"Result: {output.result}")
    
    # Print metrics
    print("\nEnhanced Metrics:")
    print(f"Performance Metrics: {output.performance_metrics}")
    print(f"Memory Usage: {output.memory_usage}")
    print(f"Operation Cost: {output.operation_cost}")
    print(f"Accuracy Score: {output.accuracy_score}")
    print(f"Truth Score: {output.truth_score}")
    
    # Convert to dict for easier inspection
    output_dict = asdict(output)
    print("\nFull ModelOutput as Dict:")
    for key, value in output_dict.items():
        print(f"{key}: {value}")
    
    # Verify metrics are present
    assert output.performance_metrics is not None, "Performance metrics should be populated"
    assert "processing_time" in output.performance_metrics, "Processing time should be measured"
    assert output.memory_usage is not None, "Memory usage should be populated"
    assert output.operation_cost is not None, "Operation cost should be calculated"
    assert output.accuracy_score is not None, "Accuracy score should be populated"
    assert output.truth_score is not None, "Truth score should be populated"
    
    print("\n✅ All metrics are properly populated!")
    return True

if __name__ == "__main__":
    success = test_model_wrapper_metrics()
    sys.exit(0 if success else 1)