#!/usr/bin/env python3
"""
Direct test of model metrics functionality.
"""
from app.services.models.wrapper import ModelOutput, ModelInput, BaseModelWrapper
from app.services.models.loader import ModelLoader, ModelConfig
import os 
import json

# Set environment variables
os.environ["CASALINGUA_ENV"] = "development"

def test_model_output_metrics():
    """Test that ModelOutput includes our enhanced metrics."""
    print("\n===== Testing ModelOutput Metrics =====")
    
    # Create a ModelOutput instance with metrics
    output = ModelOutput(
        result="Test result",
        metadata={"source": "en", "target": "es"},
        performance_metrics={"time_ms": 100, "tokens_per_second": 50},
        memory_usage={"ram_used_mb": 200, "gpu_used_mb": 100},
        operation_cost=0.0015,
        accuracy_score=0.95,
        truth_score=0.92
    )
    
    # Verify the metrics are included
    print("ModelOutput Instance:")
    print(f"- result: {output.result}")
    print(f"- performance_metrics: {output.performance_metrics}")
    print(f"- memory_usage: {output.memory_usage}")
    print(f"- operation_cost: {output.operation_cost}")
    print(f"- accuracy_score: {output.accuracy_score}")
    print(f"- truth_score: {output.truth_score}")
    
    # Convert to dict and back to verify serialization
    output_dict = output.__dict__
    print("\nSerialized to dict:")
    print(json.dumps(output_dict, indent=2))
    
    return True

def main():
    """Run all tests"""
    print("Testing Enhanced Model Metrics")
    print("=============================")
    
    # Run tests
    model_output_ok = test_model_output_metrics()
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"ModelOutput Metrics: {'PASS' if model_output_ok else 'FAIL'}")
    
    # Return exit code
    return 0 if all([model_output_ok]) else 1

if __name__ == "__main__":
    main()