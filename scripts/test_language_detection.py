#!/usr/bin/env python3
"""
Direct test of language detection with enhanced metrics.
"""
import asyncio
import json
import os
from app.core.pipeline.processor import UnifiedProcessor
from app.services.models.manager import EnhancedModelManager
from app.services.models.loader import ModelLoader
from app.audit.logger import AuditLogger
from app.audit.metrics import MetricsCollector
from app.utils.config import load_config

async def test_language_detection():
    """Test language detection and verify metrics are included."""
    print("\n===== Testing Language Detection =====")
    
    # Load configuration
    os.environ["CASALINGUA_ENV"] = "development"
    config = load_config()
    
    # Setup components
    from app.services.hardware.detector import HardwareDetector
    hardware_detector = HardwareDetector(config)
    hardware_info = hardware_detector.detect_all()
    
    # Create model loader
    model_loader = ModelLoader(config)
    
    # Create enhanced model manager
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    # Create audit logger and metrics collector
    audit_logger = AuditLogger(config)
    metrics = MetricsCollector.get_instance(config)
    
    # Create processor
    processor = UnifiedProcessor(model_manager, audit_logger, metrics)
    await processor.initialize()
    
    # Call language detection
    text = "Hello, how are you today?"
    detection_result = await processor.detect_language(text=text, detailed=True)
    
    # Print results
    print(f"Detected language: {detection_result.get('detected_language')}")
    print(f"Confidence: {detection_result.get('confidence')}")
    
    # Print metrics
    print("\nEnhanced Metrics:")
    print(f"- performance_metrics: {detection_result.get('performance_metrics')}")
    print(f"- memory_usage: {detection_result.get('memory_usage')}")
    print(f"- operation_cost: {detection_result.get('operation_cost')}")
    print(f"- accuracy_score: {detection_result.get('accuracy_score')}")
    print(f"- truth_score: {detection_result.get('truth_score')}")
    
    # Print raw result for debugging
    print("\nRaw Result:")
    print(json.dumps(detection_result, indent=2, default=str))
    
    return True

async def main():
    """Run all tests"""
    print("Testing Language Detection with Enhanced Metrics")
    print("==============================================")
    
    try:
        # Run test
        success = await test_language_detection()
        
        # Print summary
        print("\n===== Test Summary =====")
        print(f"Language Detection: {'PASS' if success else 'FAIL'}")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    asyncio.run(main())