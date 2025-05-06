#!/usr/bin/env python3
"""
Test the language detection pipeline with enhanced metrics.
"""
import sys
import os
import asyncio
import json
from pprint import pprint

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
from app.core.pipeline.processor import UnifiedProcessor
from app.utils.config import load_config
from app.audit.logger import AuditLogger
from app.audit.metrics import MetricsCollector
from app.services.models.loader import ModelLoader
from app.api.schemas.language import LanguageDetectionRequest

async def main():
    """Test language detection pipeline with enhanced metrics"""
    print("\n===== Testing Language Detection with Enhanced Metrics =====\n")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    registry_config = load_config("config/model_registry.json")
    
    # Create components
    print("Creating components...")
    audit_logger = AuditLogger(config)
    metrics = MetricsCollector(config)
    model_loader = ModelLoader(config, registry_config)
    
    # Create processor
    print("Creating unified processor...")
    processor = UnifiedProcessor(model_loader, audit_logger, metrics, config, registry_config)
    
    # Initialize processor
    print("Initializing processor...")
    await processor.initialize()
    
    # Create detection request
    request = LanguageDetectionRequest(
        text="Hello, this is a test of the language detection pipeline. Let's see if it properly identifies this as English text.",
        detailed=True
    )
    
    # Detect language
    print("\nPerforming language detection with enhanced metrics...")
    result = await processor.detect_language(
        text=request.text,
        detailed=request.detailed
    )
    
    # Print the results
    print("\n===== Detection Results =====\n")
    print(f"Detected language: {result.get('detected_language')}")
    print(f"Confidence: {result.get('confidence')}")
    
    # Check for enhanced metrics
    print("\n===== Enhanced Metrics =====\n")
    if 'performance_metrics' in result:
        print("Performance Metrics:")
        pprint(result['performance_metrics'])
    else:
        print("❌ Performance metrics not found")
    
    if 'memory_usage' in result:
        print("\nMemory Usage:")
        pprint(result['memory_usage'])
    else:
        print("❌ Memory usage not found")
    
    if 'operation_cost' in result:
        print(f"\nOperation Cost: {result['operation_cost']}")
    else:
        print("❌ Operation cost not found")
    
    if 'accuracy_score' in result:
        print(f"\nAccuracy Score: {result['accuracy_score']}")
    else:
        print("❌ Accuracy score not found")
    
    if 'truth_score' in result:
        print(f"\nTruth Score: {result['truth_score']}")
    else:
        print("❌ Truth score not found")
    
    # Print full result
    print("\n===== Full Result =====\n")
    pprint(result)
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)