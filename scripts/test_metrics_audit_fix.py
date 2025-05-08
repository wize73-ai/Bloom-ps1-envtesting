#!/usr/bin/env python3
"""
Test Script for Metrics and Audit System Fixes

This script demonstrates the enhanced metrics and audit system with
fixes for veracity and audit score reporting issues.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audit.metrics import MetricsCollector
from app.audit.logger import AuditLogger
from app.audit.veracity import VeracityAuditor
from app.audit.metrics_fix import EnhancedMetricsCollector, setup_enhanced_metrics
from app.utils.logging import get_logger, setup_logging
from app.utils.config import load_config

logger = get_logger("test_metrics_audit_fix")

async def test_metrics_system(fix_enabled: bool = True) -> Dict[str, Any]:
    """
    Test the metrics system with or without the fix.
    
    Args:
        fix_enabled: Whether to enable the metrics fix
        
    Returns:
        Dictionary with test results
    """
    # Load configuration
    config = load_config()
    
    # Initialize metrics collector
    if fix_enabled:
        # Setup enhanced metrics
        logger.info("Setting up enhanced metrics collector")
        await setup_enhanced_metrics()
        metrics = EnhancedMetricsCollector(config)
    else:
        logger.info("Using standard metrics collector")
        metrics = MetricsCollector.get_instance(config)
    
    # Record some sample metrics
    logger.info("Recording sample metrics")
    
    # API request metrics
    metrics.record_request(
        endpoint="/api/translate",
        success=True,
        duration=0.5,
        status_code=200,
        payload_size=1024
    )
    
    metrics.record_request(
        endpoint="/api/simplify",
        success=True,
        duration=0.3,
        status_code=200,
        payload_size=768
    )
    
    # Translation metrics
    metrics.record_translation_metrics(
        source_language="en",
        target_language="es",
        text_length=500,
        processing_time=0.8,
        model_id="mbart-large-en-es"
    )
    
    metrics.record_translation_metrics(
        source_language="es",
        target_language="en",
        text_length=450,
        processing_time=0.7,
        model_id="mbart-large-es-en"
    )
    
    # Simplification metrics
    metrics.record_simplification_metrics(
        language="en",
        text_length=600,
        simplified_length=400,
        level="3",
        processing_time=0.6,
        model_id="simplifier-model"
    )
    
    # Get all metrics
    all_metrics = metrics.get_all_metrics()
    
    # Check if veracity metrics are included
    has_veracity = "veracity" in all_metrics
    
    return {
        "has_veracity_metrics": has_veracity,
        "metrics": all_metrics,
        "fix_enabled": fix_enabled
    }

async def test_audit_logger() -> Dict[str, Any]:
    """
    Test the audit logger system.
    
    Returns:
        Dictionary with test results
    """
    # Initialize audit logger
    audit_logger = AuditLogger()
    
    # Log various operations
    logger.info("Logging sample audit events")
    
    # Log language detection
    lang_detection_id = await audit_logger.log_language_detection(
        text_length=200,
        detected_language="fr",
        confidence=0.92,
        model_id="language-detection-model",
        processing_time=0.15,
        metadata={"enhanced_prompt": True}
    )
    
    # Log translation
    translation_id = await audit_logger.log_translation(
        text_length=350,
        source_language="en",
        target_language="de",
        model_id="mbart-large-en-de",
        processing_time=0.65,
        quality_score=0.88,
        metadata={"enhanced_prompt": True}
    )
    
    # Log simplification
    simplify_id = await audit_logger.log_simplification(
        text_length=500,
        simplified_length=350,
        language="en",
        level="4",
        model_id="simplifier-model",
        processing_time=0.45,
        metadata={
            "grade_level": 6,
            "domain": "educational",
            "enhanced_prompt": True,
            "readability_metrics": {
                "estimated_grade_level": 5.8,
                "words_per_sentence": 8.3
            }
        }
    )
    
    # Flush logs
    await audit_logger.flush()
    
    # Search for recently logged events
    recent_logs = await audit_logger.search_logs(
        log_type="translation",
        limit=10
    )
    
    return {
        "logged_events": {
            "language_detection": lang_detection_id,
            "translation": translation_id,
            "simplification": simplify_id
        },
        "recent_logs": recent_logs,
        "logs_found": len(recent_logs)
    }

async def test_veracity_auditor() -> Dict[str, Any]:
    """
    Test the veracity auditor system.
    
    Returns:
        Dictionary with test results
    """
    # Initialize veracity auditor
    veracity = VeracityAuditor()
    await veracity.initialize()
    
    # Sample translation verification
    translation_result = await veracity.verify_translation(
        source_text="This is a sample text to test the translation verification system.",
        translation="Este es un texto de muestra para probar el sistema de verificación de traducción.",
        source_lang="en",
        target_lang="es"
    )
    
    # Sample simplification verification
    simplification_result = await veracity._verify_simplification(
        original_text="The implementation of the fiscal policies necessitated extensive deliberation concerning potential macroeconomic ramifications.",
        simplified_text="The fiscal policy changes required careful consideration of possible effects on the economy.",
        language="en",
        metadata={"level": 3, "grade_level": 8}
    )
    
    # Get quality statistics
    quality_stats = veracity.get_quality_statistics()
    
    return {
        "translation_verification": {
            "verified": translation_result.get("verified", False),
            "score": translation_result.get("score", 0.0),
            "confidence": translation_result.get("confidence", 0.0),
            "issue_count": len(translation_result.get("issues", []))
        },
        "simplification_verification": {
            "verified": simplification_result.get("verified", False),
            "score": simplification_result.get("score", 0.0),
            "confidence": simplification_result.get("confidence", 0.0),
            "issue_count": len(simplification_result.get("issues", []))
        },
        "quality_statistics": quality_stats
    }

async def compare_metrics(save_results: bool = False) -> Dict[str, Any]:
    """
    Compare metrics with and without the fix.
    
    Args:
        save_results: Whether to save results to a file
        
    Returns:
        Dictionary with comparison results
    """
    # Test with standard metrics
    standard_results = await test_metrics_system(fix_enabled=False)
    
    # Test with enhanced metrics
    enhanced_results = await test_metrics_system(fix_enabled=True)
    
    # Check for differences
    has_veracity_standard = standard_results["has_veracity_metrics"]
    has_veracity_enhanced = enhanced_results["has_veracity_metrics"]
    
    # Get API metrics for comparison
    standard_api = standard_results["metrics"].get("api", {})
    enhanced_api = enhanced_results["metrics"].get("api", {})
    
    # Compare language metrics
    standard_language = standard_results["metrics"].get("language", {})
    enhanced_language = enhanced_results["metrics"].get("language", {})
    
    # Check if enhanced metrics has veracity data
    veracity_data = enhanced_results["metrics"].get("veracity", {})
    
    comparison = {
        "standard_has_veracity": has_veracity_standard,
        "enhanced_has_veracity": has_veracity_enhanced,
        "endpoint_count_match": len(standard_api.get("endpoints", {})) == len(enhanced_api.get("endpoints", {})),
        "language_pair_count_match": len(standard_language.get("language_pairs", {})) == len(enhanced_language.get("language_pairs", {})),
        "veracity_data_available": bool(veracity_data),
        "timestamp": datetime.now().isoformat()
    }
    
    if save_results:
        # Save results to file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "metrics")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"metrics_comparison_{timestamp}.json")
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "comparison": comparison,
                "standard_results": standard_results["metrics"],
                "enhanced_results": enhanced_results["metrics"]
            }, f, indent=2)
            
        comparison["results_saved"] = filename
    
    return comparison

async def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test metrics and audit fixes")
    parser.add_argument("--metrics", action="store_true", help="Test metrics system")
    parser.add_argument("--audit", action="store_true", help="Test audit logger")
    parser.add_argument("--veracity", action="store_true", help="Test veracity auditor")
    parser.add_argument("--compare", action="store_true", help="Compare metrics with and without fix")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.all or args.metrics:
        print("\n========== Testing Metrics System ==========\n")
        metrics_results = await test_metrics_system(fix_enabled=True)
        
        print(f"Enhanced metrics enabled: {metrics_results['fix_enabled']}")
        print(f"Has veracity metrics: {metrics_results['has_veracity_metrics']}")
        
        api_metrics = metrics_results["metrics"].get("api", {})
        print(f"\nTotal API requests: {api_metrics.get('overall', {}).get('total_requests', 0)}")
        
        if "veracity" in metrics_results["metrics"]:
            veracity_metrics = metrics_results["metrics"]["veracity"]
            print("\nVeracity metrics available:")
            print(f"  - Enhanced tracking: {veracity_metrics.get('has_enhanced_tracking', False)}")
            print(f"  - Tracked language pairs: {len(veracity_metrics.get('veracity_metrics', {}))}")
    
    if args.all or args.audit:
        print("\n========== Testing Audit Logger ==========\n")
        audit_results = await test_audit_logger()
        
        print(f"Events logged:")
        for event_type, event_id in audit_results["logged_events"].items():
            print(f"  - {event_type}: {event_id}")
        
        print(f"\nFound {audit_results['logs_found']} recent log entries")
    
    if args.all or args.veracity:
        print("\n========== Testing Veracity Auditor ==========\n")
        veracity_results = await test_veracity_auditor()
        
        print("Translation verification:")
        translation = veracity_results["translation_verification"]
        print(f"  - Verified: {translation['verified']}")
        print(f"  - Score: {translation['score']:.2f}")
        print(f"  - Issues: {translation['issue_count']}")
        
        print("\nSimplification verification:")
        simplification = veracity_results["simplification_verification"]
        print(f"  - Verified: {simplification['verified']}")
        print(f"  - Score: {simplification['score']:.2f}")
        print(f"  - Issues: {simplification['issue_count']}")
        
        quality_stats = veracity_results["quality_statistics"]
        overall = quality_stats.get("overall", {})
        print("\nQuality statistics:")
        print(f"  - Total operations: {overall.get('total_count', 0)}")
        print(f"  - Verified count: {overall.get('verified_count', 0)}")
        if overall.get('total_count', 0) > 0:
            print(f"  - Verification rate: {overall.get('verification_rate', 0):.2f}")
    
    if args.all or args.compare:
        print("\n========== Comparing Metrics Systems ==========\n")
        comparison = await compare_metrics(save_results=args.save)
        
        print("Comparison results:")
        print(f"  - Standard metrics has veracity: {comparison['standard_has_veracity']}")
        print(f"  - Enhanced metrics has veracity: {comparison['enhanced_has_veracity']}")
        print(f"  - Endpoint counts match: {comparison['endpoint_count_match']}")
        print(f"  - Language pair counts match: {comparison['language_pair_count_match']}")
        print(f"  - Veracity data available in enhanced: {comparison['veracity_data_available']}")
        
        if args.save and "results_saved" in comparison:
            print(f"\nResults saved to: {comparison['results_saved']}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())