#!/usr/bin/env python3
"""
Test script for the pipeline integration with veracity checking.

This script tests the complete pipeline with veracity checking enabled.
"""

import os
import sys
import asyncio
import json
from pprint import pprint

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Add the project directory to the Python path
project_dir = os.path.abspath(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import from the application
from app.core.pipeline.processor import Processor
from app.services.models.manager import EnhancedModelManager
from app.services.models.loader import ModelLoader
from app.utils.config import load_config
from app.audit.veracity import VeracityAuditor

async def init_processing_pipeline():
    """Initialize the processing pipeline with veracity checking."""
    print(f"{BOLD}{BLUE}Initializing processing pipeline with veracity checking...{ENDC}")
    
    # Load configuration
    config = load_config()
    
    # Create hardware info dict (simplified)
    hardware_info = {
        "cpu": {
            "cores": 8,
            "threads": 16,
            "model": "Intel Core i7",
            "supports_avx2": True
        },
        "gpu": {
            "has_gpu": False,
            "cuda_available": False,
            "mps_available": True
        },
        "memory": {
            "total": 16 * 1024 * 1024 * 1024,  # 16GB
            "available": 8 * 1024 * 1024 * 1024  # 8GB
        }
    }
    
    # Create model loader
    loader = ModelLoader(config=config)
    
    # Create model manager
    model_manager = EnhancedModelManager(loader, hardware_info, config=config)
    
    # Create veracity auditor
    veracity_auditor = VeracityAuditor(model_manager=model_manager, config=config)
    await veracity_auditor.initialize()
    
    # Create processor
    processor = Processor(model_manager=model_manager, config=config)
    
    # Set veracity auditor on processor
    processor.veracity_auditor = veracity_auditor
    
    return processor

async def test_translation_pipeline():
    """Test the translation pipeline with veracity checking."""
    print(f"\n{BOLD}{BLUE}Testing Translation Pipeline with Veracity Checking{ENDC}")
    print("-" * 80)
    
    # Initialize processor
    processor = await init_processing_pipeline()
    
    # Test input in Spanish
    spanish_text = "Hola, estoy muy feliz de conocerte hoy. Mi nombre es Juan y tengo 25 años."
    print(f"{BOLD}Original Text:{ENDC} {spanish_text}")
    
    # Translate to English with verification
    result = await processor.process_translation(
        text=spanish_text,
        source_language="es",
        target_language="en",
        verify=True
    )
    
    # Print the result
    print(f"\n{BOLD}Processing completed!{ENDC}")
    print(f"{BOLD}Translation:{ENDC} {result.get('translated_text', 'No translation available')}")
    
    # Check for verification results
    if "verified" in result:
        verified = result.get("verified", False)
        status = f"{GREEN}Verified{ENDC}" if verified else f"{RED}Not verified{ENDC}"
        print(f"{BOLD}Verification:{ENDC} {status}")
    
    if "verification_score" in result:
        print(f"{BOLD}Verification Score:{ENDC} {result.get('verification_score', 0.0)}")
    
    # Check for verification metrics
    if "veracity" in result:
        print(f"\n{BOLD}{BLUE}Veracity data:{ENDC}")
        veracity_data = result["veracity"]
        
        # Print verification details
        if "issues" in veracity_data and veracity_data["issues"]:
            print(f"\n{BOLD}Issues found:{ENDC}")
            for issue in veracity_data["issues"]:
                severity = issue.get("severity", "unknown")
                if severity == "critical":
                    severity_color = RED
                elif severity == "warning":
                    severity_color = YELLOW
                else:
                    severity_color = ""
                print(f"- {severity_color}{issue.get('type', 'unknown')}{ENDC}: {issue.get('message', 'No message')}")
        
        if "metrics" in veracity_data:
            print(f"\n{BOLD}Veracity metrics:{ENDC}")
            for key, value in veracity_data["metrics"].items():
                print(f"- {key}: {value}")
    
    # Check for model information
    if "model_used" in result:
        print(f"\n{BOLD}Model Used:{ENDC} {result.get('model_used', 'Unknown')}")
    
    if "method" in result:
        print(f"{BOLD}Method:{ENDC} {result.get('method', 'Unknown')}")
    
    # Check for performance metrics
    if "performance_metrics" in result:
        print(f"\n{BOLD}Performance Metrics:{ENDC}")
        metrics = result["performance_metrics"]
        for key, value in metrics.items():
            print(f"- {key}: {value}")
    
    # Success
    print(f"\n{GREEN}✅ Pipeline integration test completed!{ENDC}")

if __name__ == "__main__":
    asyncio.run(test_translation_pipeline())