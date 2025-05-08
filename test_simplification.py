#!/usr/bin/env python3
"""
Test script for text simplification functionality in CasaLingua.

This script directly tests the SimplificationPipeline by creating an instance
and running simplification on test cases.
"""

import os
import sys
import asyncio
from typing import Dict, Any
import json

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import required components
from app.utils.config import load_config
from app.services.models.loader import ModelLoader
from app.services.models.manager import EnhancedModelManager
from app.core.pipeline.simplifier import SimplificationPipeline

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Test cases with varying degrees of complexity
TEST_CASES = [
    {
        "name": "Legal Statement",
        "text": "The plaintiff alleged that the defendant had violated numerous provisions of the aforementioned contractual agreement.",
        "language": "en",
        "level": 3
    },
    {
        "name": "Housing Terms",
        "text": "In accordance with paragraph 12(b) of the aforesaid Lease Agreement, the Lessee is obligated to remit payment for all utilities, including but not limited to water, electricity, gas, and telecommunications services, consumed or utilized on the premises during the term of occupancy.",
        "language": "en",
        "level": 4
    },
    {
        "name": "Technical Instructions",
        "text": "Prior to commencement of the installation process, ensure that all prerequisite components have been obtained and are readily accessible for utilization.",
        "language": "en",
        "level": 5
    },
    {
        "name": "Simple Statement",
        "text": "Turn off the water before you start fixing the sink.",
        "language": "en",
        "level": 2
    }
]

async def initialize_models():
    """Initialize model components required for simplification."""
    print(f"{BLUE}Initializing models...{ENDC}")
    
    # Load configuration
    config = load_config()
    
    # Initialize model loader
    model_loader = ModelLoader(config=config)
    
    # Create a minimal hardware info for the model manager
    hardware_info = {
        "memory": {"total_gb": 8, "available_gb": 4},
        "cpu": {"count_physical": 4, "count_logical": 8},
        "gpu": {"has_gpu": False}
    }
    
    # Initialize model manager
    model_manager = EnhancedModelManager(model_loader, hardware_info, config)
    
    return config, model_manager

async def test_simplification():
    """Test simplified text functionality."""
    print(f"\n{BOLD}{BLUE}Testing Simplification Pipeline{ENDC}")
    print("-" * 80)
    
    # Initialize models
    config, model_manager = await initialize_models()
    
    # Create simplification pipeline
    simplifier = SimplificationPipeline(model_manager, config)
    
    # Initialize pipeline
    await simplifier.initialize()
    print(f"{GREEN}✓ Simplification pipeline initialized{ENDC}")
    
    # Process each test case
    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"\n{BOLD}Test Case {i+1}: {case['name']}{ENDC}")
        print(f"{BOLD}Original:{ENDC} {case['text']}")
        
        # Run simplification
        try:
            result = await simplifier.simplify(
                text=case['text'],
                language=case['language'],
                level=case['level']
            )
            
            simplified_text = result.get('simplified_text', 'No simplified text returned')
            print(f"{BOLD}Simplified (Level {case['level']}):{ENDC} {simplified_text}")
            
            # Check if the simplified text is different from the original
            is_different = simplified_text.lower() != case['text'].lower()
            success_label = f"{GREEN}SUCCESS{ENDC}" if is_different else f"{RED}FAILED{ENDC}"
            print(f"{BOLD}Result:{ENDC} {success_label}")
            
            # Print metrics if available
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"{BOLD}Metrics:{ENDC}")
                for key, value in metrics.items():
                    print(f"  - {key}: {value}")
            
            results.append({
                "name": case['name'],
                "original": case['text'],
                "simplified": simplified_text,
                "success": is_different,
                "metrics": result.get('metrics', {})
            })
            
        except Exception as e:
            print(f"{RED}Error during simplification: {str(e)}{ENDC}")
            results.append({
                "name": case['name'],
                "original": case['text'],
                "error": str(e),
                "success": False
            })
        
        print("-" * 80)
    
    # Print summary
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    print(f"\n{BOLD}Summary:{ENDC}")
    print(f"Successful simplifications: {successful}/{total} ({successful/total*100:.1f}%)")
    
    # Determine overall success
    if successful == total:
        print(f"\n{GREEN}All simplification tests passed!{ENDC}")
    elif successful >= total * 0.75:
        print(f"\n{YELLOW}Most simplification tests passed ({successful}/{total}).{ENDC}")
    else:
        print(f"\n{RED}Too many simplification tests failed ({total-successful}/{total}).{ENDC}")

async def run_tests():
    """Run all tests for simplification."""
    print(f"{BOLD}{BLUE}=== Simplified Text Pipeline Tests ==={ENDC}")
    
    try:
        await test_simplification()
    except Exception as e:
        print(f"\n{BOLD}{RED}Error during testing: {str(e)}{ENDC}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # For Windows compatibility, use the correct event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_tests())