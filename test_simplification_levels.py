#!/usr/bin/env python3
"""
Test script for text simplification levels in CasaLingua.

This script tests the different simplification levels of the SimplificationPipeline.
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

# Single test text with different simplification levels
TEST_TEXT = "The applicant must furnish documentation verifying income and employment status in accordance with the requirements delineated in section 8 of the aforementioned application procedure."

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

async def test_simplification_levels():
    """Test different simplification levels on the same text."""
    print(f"\n{BOLD}{BLUE}Testing Simplification Levels{ENDC}")
    print("-" * 80)
    
    # Initialize models
    config, model_manager = await initialize_models()
    
    # Create simplification pipeline
    simplifier = SimplificationPipeline(model_manager, config)
    
    # Initialize pipeline
    await simplifier.initialize()
    print(f"{GREEN}✓ Simplification pipeline initialized{ENDC}")
    
    # Original text
    print(f"\n{BOLD}Original Text:{ENDC}")
    print(TEST_TEXT)
    print("-" * 80)
    
    # Test each simplification level (1-5)
    results = []
    for level in range(1, 6):
        print(f"\n{BOLD}Testing Simplification Level {level}:{ENDC}")
        
        # Run simplification
        try:
            result = await simplifier.simplify(
                text=TEST_TEXT,
                language="en",
                level=level
            )
            
            simplified_text = result.get('simplified_text', 'No simplified text returned')
            print(f"{simplified_text}")
            
            # Check if the simplified text is different from the original
            is_different = simplified_text.lower() != TEST_TEXT.lower()
            status = "Different" if is_different else "Same as original"
            print(f"{BOLD}Status:{ENDC} {status}")
            
            # Print metrics if available
            if 'metrics' in result:
                metrics = result['metrics']
                grade_level = metrics.get('estimated_grade_level', 'Unknown')
                print(f"{BOLD}Grade Level:{ENDC} {grade_level}")
            
            results.append({
                "level": level,
                "text": simplified_text,
                "is_different": is_different,
                "grade_level": grade_level if 'metrics' in result else 'Unknown'
            })
            
        except Exception as e:
            print(f"{RED}Error during simplification: {str(e)}{ENDC}")
            results.append({
                "level": level,
                "error": str(e),
                "is_different": False
            })
        
        print("-" * 80)
    
    # Check if there are differences between the levels
    all_texts = [r.get('text', '') for r in results if 'text' in r]
    unique_texts = set(all_texts)
    
    print(f"\n{BOLD}Summary:{ENDC}")
    print(f"Unique simplified outputs: {len(unique_texts)} out of 5 levels")
    
    if len(unique_texts) >= 3:
        print(f"{GREEN}The simplification levels are producing different outputs as expected.{ENDC}")
    else:
        print(f"{YELLOW}The simplification levels are not producing sufficiently different outputs.{ENDC}")
    
    # Compare grade levels
    grade_levels = [r.get('grade_level', 'Unknown') for r in results if 'grade_level' in r]
    if 'Unknown' not in grade_levels:
        try:
            are_descending = all(float(grade_levels[i]) >= float(grade_levels[i+1]) for i in range(len(grade_levels)-1))
            if are_descending:
                print(f"{GREEN}Grade levels are correctly decreasing as simplification level increases.{ENDC}")
            else:
                print(f"{YELLOW}Grade levels are not consistently decreasing as expected.{ENDC}")
        except (ValueError, TypeError):
            print(f"{YELLOW}Could not compare grade levels numerically.{ENDC}")

async def run_tests():
    """Run simplification level tests."""
    print(f"{BOLD}{BLUE}=== Simplification Levels Test ==={ENDC}")
    
    try:
        await test_simplification_levels()
        print(f"\n{GREEN}Simplification levels test completed!{ENDC}")
    except Exception as e:
        print(f"\n{BOLD}{RED}Error during testing: {str(e)}{ENDC}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # For Windows compatibility, use the correct event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_tests())