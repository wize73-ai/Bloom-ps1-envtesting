#!/usr/bin/env python3
"""
Test Script for Enhanced Language Detection

This script demonstrates the enhanced language detection system with
model-specific prompt optimization.

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

from app.core.pipeline.language_detector import LanguageDetector
from app.services.models.manager import EnhancedModelManager as ModelManager
from app.utils.logging import get_logger
from app.utils.config import load_config

logger = get_logger("test_enhanced_language_detection")

async def test_language_detection(text: str, model_name: Optional[str] = None, detailed: bool = False) -> Dict[str, Any]:
    """
    Test the enhanced language detection system.
    
    Args:
        text: Text to detect language
        model_name: Optional model name to use
        detailed: Whether to include detailed information
        
    Returns:
        Dictionary with detection results
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
    
    # Initialize language detector
    detector = LanguageDetector(model_manager, config)
    await detector.initialize()
    
    # Run detection
    result = await detector.detect_language(text, detailed=detailed)
    
    return result

async def run_sample_tests():
    """Run a series of tests with different languages and text samples."""
    test_samples = [
        {
            "name": "English",
            "text": "The quick brown fox jumps over the lazy dog. This system is designed to detect language with high accuracy.",
            "expected": "en"
        },
        {
            "name": "Spanish",
            "text": "El zorro marrón rápido salta sobre el perro perezoso. Este sistema está diseñado para detectar el idioma con alta precisión.",
            "expected": "es"
        },
        {
            "name": "French",
            "text": "Le renard brun rapide saute par-dessus le chien paresseux. Ce système est conçu pour détecter la langue avec une grande précision.",
            "expected": "fr"
        },
        {
            "name": "German",
            "text": "Der schnelle braune Fuchs springt über den faulen Hund. Dieses System wurde entwickelt, um Sprache mit hoher Genauigkeit zu erkennen.",
            "expected": "de"
        },
        {
            "name": "Chinese",
            "text": "快速的棕色狐狸跳过懒狗。该系统旨在以高精度检测语言。",
            "expected": "zh"
        },
        {
            "name": "Code-mixed",
            "text": "This is a sample of text with 一些中文 mixed in to challenge the detector.",
            "expected": "en"
        },
        {
            "name": "Ambiguous",
            "text": "OK",
            "expected": "any"  # Could be any language
        }
    ]
    
    print("\n========== Testing Enhanced Language Detection ==========\n")
    
    for sample in test_samples:
        print(f"Testing: {sample['name']}")
        print(f"Text: {sample['text']}")
        
        # Test with basic detection
        result = await test_language_detection(sample['text'])
        print(f"Detected language: {result['detected_language']} (confidence: {result['confidence']:.2f})")
        
        # Test with detailed detection
        detailed_result = await test_language_detection(sample['text'], detailed=True)
        
        if detailed_result.get('possible_languages'):
            print("Possible languages:")
            for lang in detailed_result['possible_languages']:
                print(f"  - {lang['language']} (confidence: {lang['confidence']:.2f})")
        
        # Check if result matches expectation
        if sample['expected'] != "any" and result['detected_language'] != sample['expected']:
            print(f"WARNING: Expected {sample['expected']} but got {result['detected_language']}")
        
        print("\n")
    
    print("========== Metrics Detection Test ==========\n")
    
    # Test with a longer text to check metrics
    long_text = """
    Language detection is a crucial component of natural language processing systems. 
    It enables applications to determine the language of input text, which is essential for 
    subsequent processing steps like translation, text-to-speech, or content filtering.
    Modern language detection systems use a variety of techniques, from statistical models 
    to neural networks, to achieve high accuracy across a wide range of languages.
    """
    
    result = await test_language_detection(long_text, detailed=True)
    
    print(f"Detected language: {result['detected_language']} (confidence: {result['confidence']:.2f})")
    print(f"Processing time: {result.get('processing_time', 0):.4f} seconds")
    
    # Print any performance metrics if available
    if "performance_metrics" in result:
        print("\nPerformance metrics:")
        for key, value in result["performance_metrics"].items():
            print(f"  - {key}: {value}")
    
    # Print any accuracy metrics if available
    if "accuracy_score" in result:
        print(f"\nAccuracy score: {result['accuracy_score']:.2f}")
    if "truth_score" in result:
        print(f"Truth score: {result['truth_score']:.2f}")
    
    print("\nTest completed successfully!")

async def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test enhanced language detection")
    parser.add_argument("--text", type=str, help="Text to detect language")
    parser.add_argument("--detailed", action="store_true", help="Include detailed information")
    parser.add_argument("--model", type=str, help="Model name to use", default=None)
    parser.add_argument("--samples", action="store_true", help="Run sample tests")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.samples:
        await run_sample_tests()
    elif args.text:
        result = await test_language_detection(args.text, model_name=args.model, detailed=args.detailed)
        print(f"Detected language: {result['detected_language']} (confidence: {result['confidence']:.2f})")
        
        if args.detailed and result.get('possible_languages'):
            print("\nPossible languages:")
            for lang in result['possible_languages']:
                print(f"  - {lang['language']} (confidence: {lang['confidence']:.2f})")
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())