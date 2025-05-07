#!/usr/bin/env python3
"""
Test Script for Enhanced Text Simplification

This script demonstrates the enhanced text simplification system with
model-specific prompt optimization and multi-level simplification.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.pipeline.simplifier import SimplificationPipeline
from app.services.models.manager import EnhancedModelManager as ModelManager
from app.utils.logging import get_logger
from app.utils.config import load_config

logger = get_logger("test_enhanced_simplification")

async def test_simplification(
    text: str, 
    level: int = 3, 
    language: str = "en",
    domain: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test the enhanced text simplification system.
    
    Args:
        text: Text to simplify
        level: Simplification level (1-5, where 5 is simplest)
        language: Language code
        domain: Specific domain (legal, medical, etc.)
        model_name: Optional model name to use
        
    Returns:
        Dictionary with simplification results
    """
    # Load configuration
    config = load_config()
    
    # Initialize model manager
    # Create hardware info
    hardware_info = {
        "gpu_count": 0,
        "cpu_count": os.cpu_count() or 4,
        "memory_gb": 16
    }
    model_manager = ModelManager(config, hardware_info)
    # Model manager initializes automatically
    
    # Initialize simplification pipeline
    simplifier = SimplificationPipeline(model_manager, config)
    await simplifier.initialize()
    
    # Set options
    options = {
        "preserve_formatting": True
    }
    
    if domain:
        options["domain"] = domain
    
    if model_name:
        options["model_name"] = model_name
    
    # Run simplification
    result = await simplifier.simplify(text, language, level, options=options)
    
    return result

async def compare_simplification_levels(text: str, language: str = "en", domain: Optional[str] = None):
    """
    Compare simplification across all 5 levels.
    
    Args:
        text: Text to simplify
        language: Language code
        domain: Specific domain (legal, medical, etc.)
    """
    print("\n========== Comparing Simplification Levels ==========\n")
    print(f"Original text:\n{text}\n")
    
    if domain:
        print(f"Domain: {domain}\n")
    
    results = []
    
    # Simplify at each level
    for level in range(1, 6):
        level_name = ["Academic", "Standard", "Simplified", "Basic", "Elementary"][level-1]
        print(f"Level {level} ({level_name}):")
        
        result = await test_simplification(text, level, language, domain)
        results.append(result)
        
        print(f"Simplified text:\n{result['simplified_text']}\n")
        
        # Print readability metrics
        if "metrics" in result:
            metrics = result["metrics"]
            print("Readability metrics:")
            print(f"  - Estimated grade level: {metrics.get('estimated_grade_level', 'N/A')}")
            print(f"  - Words per sentence: {metrics.get('words_per_sentence', 'N/A')}")
            print(f"  - Syllables per word: {metrics.get('syllables_per_word', 'N/A')}")
            
            # Check if verification metrics are available
            if "verification" in metrics:
                verification = metrics["verification"]
                print("\nVerification metrics:")
                print(f"  - Quality score: {verification.get('score', 'N/A')}")
                print(f"  - Verified: {verification.get('verified', 'N/A')}")
                print(f"  - Issues: {verification.get('issues', 'N/A')}")
        
        print("-" * 60)
    
    # Print summary
    print("\n========== Simplification Summary ==========\n")
    print("Text length comparison:")
    print(f"Original: {len(text.split())} words")
    
    for i, result in enumerate(results):
        level = i + 1
        level_name = ["Academic", "Standard", "Simplified", "Basic", "Elementary"][i]
        simplified_text = result["simplified_text"]
        word_count = len(simplified_text.split())
        reduction = (1 - word_count / len(text.split())) * 100
        
        print(f"Level {level} ({level_name}): {word_count} words ({reduction:.1f}% reduction)")
    
    print("\nProcessing time comparison:")
    for i, result in enumerate(results):
        level = i + 1
        level_name = ["Academic", "Standard", "Simplified", "Basic", "Elementary"][i]
        processing_time = result.get("processing_time", 0)
        
        print(f"Level {level} ({level_name}): {processing_time:.4f} seconds")

async def test_domain_simplification():
    """Test simplification for different domains."""
    print("\n========== Testing Domain-Specific Simplification ==========\n")
    
    domains = {
        "legal": """
        The parties hereto agree that in the event of a breach of the aforementioned provisions, 
        the non-breaching party shall be entitled to equitable relief including, but not limited to, 
        injunctive relief and specific performance, in addition to any other remedies available at law or in equity.
        """,
        
        "medical": """
        The patient presented with symptoms of myocardial infarction, including acute chest pain 
        radiating to the left arm, diaphoresis, and dyspnea. Initial electrocardiogram revealed 
        ST-segment elevation in the anterior leads, consistent with an anterior wall STEMI.
        """,
        
        "technical": """
        The system architecture implements a microservices approach utilizing containerized deployments 
        orchestrated via Kubernetes for horizontal scalability. API communication is facilitated through 
        RESTful endpoints with GraphQL integration for complex data retrieval operations.
        """,
        
        "financial": """
        The portfolio diversification strategy mitigates risk exposure through asset allocation across 
        multiple classes with varying correlation coefficients. Maximum drawdown metrics indicate 
        enhanced volatility protection during market corrections compared to benchmark indices.
        """
    }
    
    for domain, text in domains.items():
        print(f"Domain: {domain.upper()}")
        print(f"Original text:\n{text.strip()}\n")
        
        # Test at level 3 (Simplified) and level 5 (Elementary)
        for level in [3, 5]:
            level_name = "Simplified" if level == 3 else "Elementary"
            print(f"Level {level} ({level_name}):")
            
            result = await test_simplification(text, level, "en", domain)
            
            print(f"Simplified text:\n{result['simplified_text']}\n")
            
            # Print readability metrics
            if "metrics" in result:
                metrics = result["metrics"]
                print("Readability metrics:")
                print(f"  - Estimated grade level: {metrics.get('estimated_grade_level', 'N/A')}")
                print(f"  - Words per sentence: {metrics.get('words_per_sentence', 'N/A')}")
                
                # Check if verification metrics are available
                if "verification" in metrics:
                    verification = metrics["verification"]
                    print("\nVerification metrics:")
                    print(f"  - Quality score: {verification.get('score', 'N/A')}")
                    print(f"  - Verified: {verification.get('verified', 'N/A')}")
            
            print("-" * 60)
        
        print("\n")

async def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test enhanced text simplification")
    parser.add_argument("--text", type=str, help="Text to simplify")
    parser.add_argument("--level", type=int, help="Simplification level (1-5)", default=3)
    parser.add_argument("--language", type=str, help="Language code", default="en")
    parser.add_argument("--domain", type=str, help="Specific domain", default=None)
    parser.add_argument("--model", type=str, help="Model name to use", default=None)
    parser.add_argument("--compare", action="store_true", help="Compare all simplification levels")
    parser.add_argument("--domain-test", action="store_true", help="Test domain-specific simplification")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.domain_test:
        await test_domain_simplification()
    elif args.text and args.compare:
        await compare_simplification_levels(args.text, args.language, args.domain)
    elif args.text:
        result = await test_simplification(
            args.text, args.level, args.language, args.domain, args.model
        )
        
        print(f"Simplified text (level {args.level}):\n{result['simplified_text']}\n")
        
        # Print metrics if available
        if "metrics" in result:
            metrics = result["metrics"]
            print("Readability metrics:")
            for key, value in metrics.items():
                if key != "verification":
                    print(f"  - {key}: {value}")
            
            # Print verification metrics if available
            if "verification" in metrics:
                verification = metrics["verification"]
                print("\nVerification metrics:")
                for key, value in verification.items():
                    print(f"  - {key}: {value}")
        
        print(f"\nProcessing time: {result.get('processing_time', 0):.4f} seconds")
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())