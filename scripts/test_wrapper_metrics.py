#!/usr/bin/env python3
"""
Direct test of ModelWrapper's enhanced metrics functionality.
"""
import sys
import os
import time
import json
import asyncio
import torch
from dataclasses import asdict

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
from app.services.models.wrapper import ModelOutput, ModelInput
from app.services.models.manager import EnhancedModelManager
from app.services.models.loader import ModelLoader
from app.utils.config import load_config

async def main():
    """Test model wrapper metrics directly"""
    print("\n===== Testing Model Wrapper Enhanced Metrics =====\n")
    
    # Load the configuration
    print("Loading configuration...")
    config = load_config()
    registry_config = load_config("config/model_registry.json")
    
    # Create model loader
    print("Creating model loader...")
    model_loader = ModelLoader(config, registry_config)
    
    # Create hardware info dict
    hardware_info = {
        "cpu_cores": os.cpu_count(),
        "ram_gb": 16,  # Assuming 16GB
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_memory_gb": [8] * torch.cuda.device_count() if torch.cuda.is_available() else []  # Assuming 8GB per GPU
    }
    
    # Create model manager
    print("Creating model manager...")
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    # Initialize model manager
    print("Initializing model manager...")
    await model_manager.initialize()
    
    # Load language detection model
    print("Loading language detection model...")
    await model_manager.load_model("language_detection")
    
    # Create input for language detection
    input_data = ModelInput(
        text="Hello, this is a test of the language detection model. Let's see if it properly identifies this as English text.",
        parameters={"detailed": True}
    )
    
    # Process the input
    print("\nProcessing input with enhanced metrics...")
    result = await model_manager.run_model("language_detection", "process", input_data)
    
    # Print the results
    print("\n===== Results =====\n")
    print(f"Result: {result.get('result')}")
    
    # Check if enhanced metrics are included
    print("\n===== Enhanced Metrics =====\n")
    print(f"Performance Metrics: {json.dumps(result.get('performance_metrics', {}), indent=2)}")
    print(f"Memory Usage: {json.dumps(result.get('memory_usage', {}), indent=2)}")
    print(f"Operation Cost: {result.get('operation_cost')}")
    print(f"Accuracy Score: {result.get('accuracy_score')}")
    print(f"Truth Score: {result.get('truth_score')}")
    
    # Print full result for inspection
    print("\n===== Full ModelOutput as Dict =====\n")
    for key, value in result.items():
        if key in ['performance_metrics', 'memory_usage']:
            print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")
    
    # Verify metrics are present
    success = True
    if result.get('performance_metrics') is None:
        print("\n❌ Performance metrics are missing")
        success = False
    elif "total_time" not in result.get('performance_metrics'):
        print("\n❌ Processing time not captured in performance metrics")
        success = False
    
    if result.get('memory_usage') is None:
        print("\n❌ Memory usage metrics are missing")
        success = False
    
    if result.get('operation_cost') is None:
        print("\n❌ Operation cost estimate is missing")
        success = False
    
    if success:
        print("\n✅ All enhanced metrics are properly captured!")
    else:
        print("\n⚠️ Some metrics are missing or incomplete")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)