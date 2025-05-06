#!/usr/bin/env python3
"""
Inspect ModelOutput class to ensure enhanced metrics fields are defined properly.
This is a minimal test that doesn't require loading models.
"""
import sys
import os
from dataclasses import asdict, fields

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required module
from app.services.models.wrapper import ModelOutput

def main():
    """Inspect ModelOutput class structure"""
    print("\n===== Inspecting ModelOutput Class Structure =====\n")
    
    # Print the field names of ModelOutput
    print("ModelOutput field names:")
    for field in fields(ModelOutput):
        print(f"- {field.name}: {field.type}")
    
    # Create a sample ModelOutput instance with enhanced metrics
    output = ModelOutput(
        result="Sample result",
        metadata={"sample": "metadata"},
        metrics={"total_time": 0.123},
        status="success",
        performance_metrics={
            "preprocess_time": 0.01,
            "inference_time": 0.1,
            "postprocess_time": 0.013,
            "total_time": 0.123,
            "tokens_processed": {
                "input_tokens_estimate": 10,
                "output_tokens_estimate": 20,
                "total_tokens_estimate": 30
            },
            "throughput": {
                "tokens_per_second": 243.9,
                "chars_per_second": 650.4
            }
        },
        memory_usage={
            "before": {"used": 1000000, "free": 2000000},
            "after": {"used": 1100000, "free": 1900000},
            "difference": {"used": 100000, "free": -100000}
        },
        operation_cost=0.000123,
        accuracy_score=0.95,
        truth_score=0.87
    )
    
    # Convert to dict and print to see full structure
    output_dict = asdict(output)
    print("\nSample ModelOutput as dict:")
    for key, value in output_dict.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sk, sv in v.items():
                        print(f"    {sk}: {sv}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {value}")
    
    print("\n===== ModelOutput Enhanced Metrics Structure =====\n")
    
    # Verify all enhanced metrics fields are present
    has_all_fields = True
    for field_name in ["performance_metrics", "memory_usage", "operation_cost", "accuracy_score", "truth_score"]:
        if hasattr(output, field_name):
            print(f"✅ Field '{field_name}' is present in ModelOutput")
        else:
            print(f"❌ Field '{field_name}' is missing from ModelOutput")
            has_all_fields = False
    
    if has_all_fields:
        print("\n✅ ModelOutput has all required enhanced metrics fields!")
        print("The class is properly defined to hold the enhanced metrics.")
        print("If metrics are not showing up in API responses, the issue is in:")
        print("1. How these fields are populated during model processing, or")
        print("2. How they're transferred from ModelOutput to the API response schema.")
    else:
        print("\n❌ ModelOutput is missing some enhanced metrics fields!")
        print("The ModelOutput class needs to be updated to include all required fields.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())