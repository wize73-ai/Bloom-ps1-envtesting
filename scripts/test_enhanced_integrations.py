#!/usr/bin/env python3
"""
Test script for the Enhanced CasaLingua Integrations.

This script tests all the enhanced components together:
- Language detector with prompt enhancement
- Simplifier with 5-level prompt enhancement
- Metrics collector with veracity and audit score fixes

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logging import get_logger
from app.core.enhanced_integrations import setup_enhanced_components, verify_enhanced_components

logger = get_logger(__name__)

async def test_enhanced_integrations():
    """Test all enhanced CasaLingua integrations."""
    logger.info("Starting enhanced integrations test")
    
    # Setup all enhanced components
    logger.info("Setting up enhanced components...")
    setup_status = await setup_enhanced_components()
    
    # Print setup status
    logger.info("Enhanced components setup status:")
    for component, status in setup_status.items():
        logger.info(f"  - {component}: {'✅ Success' if status else '❌ Failed'}")
    
    # Verify components
    logger.info("Verifying enhanced components...")
    verification_results = await verify_enhanced_components()
    
    # Print verification results
    logger.info("Enhanced components verification results:")
    for component, result in verification_results.items():
        status_str = "✅ Verified" if result["status"] else "❌ Not verified"
        logger.info(f"  - {component}: {status_str}")
        
        # Print details if there was an error
        if "error" in result["details"]:
            logger.warning(f"    Error: {result['details']['error']}")
    
    # Test language detection if available
    if setup_status["language_detector"]:
        await test_language_detection()
    
    # Test simplification if available
    if setup_status["simplifier"]:
        await test_simplification()
    
    # Test metrics reporting if available
    if setup_status["metrics_collector"]:
        await test_metrics_reporting()
    
    logger.info("Enhanced integrations test completed")
    
    # Write verification results to a file
    with open("enhanced_integration_verification.json", "w") as f:
        json.dump(verification_results, f, indent=2)
    
    logger.info("Verification results written to enhanced_integration_verification.json")
    
    return setup_status, verification_results

async def test_language_detection():
    """Test enhanced language detection."""
    logger.info("Testing language detection with prompt enhancement...")
    
    try:
        from app.core.pipeline.language_detector import LanguageDetector
        
        # Create language detector
        detector = LanguageDetector()
        
        # Test with different text samples
        test_texts = [
            ("Hello, how are you today?", "en"),
            ("Hola, ¿cómo estás hoy?", "es"),
            ("Bonjour, comment allez-vous aujourd'hui?", "fr"),
            ("Guten Tag, wie geht es Ihnen heute?", "de"),
            ("今日は元気ですか？", "ja"),
            ("你今天好吗？", "zh"),
            ("Code-mixed text with English and un poco de español", "mixed")
        ]
        
        for text, expected_lang in test_texts:
            try:
                # Detect language with enhanced detector
                detected = await detector.detect_language(text)
                
                # Log result
                logger.info(f"Text: '{text[:30]}...' - Detected: {detected['language']} (Expected: {expected_lang})")
                
                # Check for enhanced fields
                has_enhanced_fields = "confidence" in detected and "details" in detected
                if has_enhanced_fields:
                    logger.info(f"  - Enhanced fields present: confidence={detected.get('confidence', 0)}")
                else:
                    logger.warning("  - No enhanced fields found in language detection result")
                    
            except Exception as e:
                logger.error(f"Error detecting language for '{text[:30]}...': {e}")
        
        logger.info("Language detection test completed")
    except Exception as e:
        logger.error(f"Error testing language detection: {e}")

async def test_simplification():
    """Test enhanced text simplification with 5 levels."""
    logger.info("Testing text simplification with 5-level prompt enhancement...")
    
    try:
        from app.core.pipeline.simplifier import TextSimplifier
        
        # Create simplifier
        simplifier = TextSimplifier()
        
        # Test text (a complex academic passage)
        test_text = """
        The implementation of computational linguistics approaches to semantic analysis
        involves the algorithmic interpretation of natural language through vectorized 
        representation models, thereby facilitating the extraction of conceptual meaning 
        from lexical structures. Contemporary methodologies employ transformer-based 
        architectures that leverage attention mechanisms to contextualize word embeddings 
        within their syntactic and semantic environments.
        """
        
        # Test with different simplification levels
        for level in range(1, 6):
            try:
                # Simplify text with specific level
                simplified = await simplifier.simplify_text(
                    text=test_text,
                    language="en",
                    level=level
                )
                
                # Log result (truncated for readability)
                logger.info(f"Level {level} simplification result:")
                logger.info(f"  - '{simplified[:100]}...'")
                
                # Calculate simplification metrics
                original_words = len(test_text.split())
                simplified_words = len(simplified.split())
                ratio = simplified_words / original_words if original_words > 0 else 0
                
                logger.info(f"  - Metrics: Original={original_words} words, Simplified={simplified_words} words, Ratio={ratio:.2f}")
                
            except Exception as e:
                logger.error(f"Error simplifying text at level {level}: {e}")
        
        logger.info("Simplification test completed")
    except Exception as e:
        logger.error(f"Error testing simplification: {e}")

async def test_metrics_reporting():
    """Test enhanced metrics reporting with veracity and audit scores."""
    logger.info("Testing metrics reporting with veracity and audit score fixes...")
    
    try:
        from app.audit.metrics import MetricsCollector
        from app.audit.metrics_fix import EnhancedMetricsCollector
        
        # Get metrics collector instance
        metrics = MetricsCollector.get_instance()
        
        # Verify it's the enhanced version
        if not isinstance(metrics, EnhancedMetricsCollector):
            logger.warning("Metrics collector is not the enhanced version")
            return
        
        # Record a translation with veracity data
        logger.info("Recording translation metrics with veracity data...")
        translation_veracity_data = {
            "verified": True,
            "score": 0.92,
            "confidence": 0.89,
            "issues": [
                {"type": "minor_punctuation", "severity": "warning", "message": "Minor punctuation differences"}
            ]
        }
        
        metrics.record_translation_metrics(
            source_language="en",
            target_language="es",
            text_length=1500,
            processing_time=0.75,
            model_id="gpt-4",
            veracity_data=translation_veracity_data
        )
        
        # Record a simplification with veracity data
        logger.info("Recording simplification metrics with veracity data...")
        simplification_veracity_data = {
            "verified": True,
            "score": 0.88,
            "confidence": 0.85,
            "issues": [
                {"type": "slight_meaning_change", "severity": "warning", "message": "Slight meaning changes detected"}
            ]
        }
        
        for level in range(1, 6):
            metrics.record_simplification_metrics(
                language="en",
                text_length=2000,
                simplified_length=1500,
                level=str(level),
                processing_time=0.65,
                model_id="gpt-4",
                veracity_data=simplification_veracity_data
            )
        
        # Record audit scores
        logger.info("Recording audit scores...")
        
        metrics.record_audit_score(
            operation="translation",
            language="en",
            target_language="es",
            score=0.93
        )
        
        for level in range(1, 6):
            metrics.record_audit_score(
                operation="simplification",
                language="en",
                target_language=None,
                score=0.85 + (level * 0.02),
                metadata={"level": str(level)}
            )
        
        # Wait a bit to allow async tasks to complete
        await asyncio.sleep(1)
        
        # Get metrics
        logger.info("Retrieving metrics...")
        all_metrics = metrics.get_all_metrics()
        
        # Check for enhanced metrics sections
        has_veracity = "veracity" in all_metrics
        has_audit = "audit" in all_metrics
        
        logger.info(f"Enhanced metrics available: veracity={has_veracity}, audit={has_audit}")
        
        # Write metrics to a file for inspection
        with open("enhanced_metrics_test_output.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info("Metrics written to enhanced_metrics_test_output.json")
        
    except Exception as e:
        logger.error(f"Error testing metrics reporting: {e}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_enhanced_integrations())