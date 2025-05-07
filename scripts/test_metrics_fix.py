#!/usr/bin/env python3
"""
Test script for the Enhanced Metrics Collector with Veracity and Audit Score Fixes.

This script demonstrates the usage of the enhanced metrics collector and verifies
that veracity and audit score reporting work correctly after the fixes.

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
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logging import get_logger
from app.audit.metrics import MetricsCollector
from app.audit.metrics_fix import EnhancedMetricsCollector, setup_enhanced_metrics_collector
from app.audit.veracity import VeracityAuditor
from app.audit.logger import AuditLogger

logger = get_logger(__name__)

async def test_metrics_fix():
    """Run tests for the enhanced metrics collector with veracity and audit score fixes."""
    logger.info("Starting metrics fix test")
    
    # Setup enhanced metrics collector
    enhanced_metrics = setup_enhanced_metrics_collector()
    
    # Verify it's correctly installed
    assert MetricsCollector.get_instance() is enhanced_metrics
    logger.info("Enhanced metrics collector installed correctly")
    
    # Setup veracity auditor and audit logger
    veracity = VeracityAuditor()
    audit_logger = AuditLogger()
    
    # Test recording translation metrics with veracity data
    logger.info("Testing translation metrics with veracity data")
    
    # Create mock veracity data
    translation_veracity_data = {
        "verified": True,
        "score": 0.92,
        "confidence": 0.89,
        "issues": [
            {"type": "minor_punctuation", "severity": "warning", "message": "Punctuation differs"}
        ],
        "metrics": {
            "semantic_similarity": 0.95,
            "entity_preservation": 0.98
        }
    }
    
    # Record a translation with veracity data
    enhanced_metrics.record_translation_metrics(
        source_language="en",
        target_language="es",
        text_length=1500,
        processing_time=0.75,
        model_id="gpt-4",
        veracity_data=translation_veracity_data
    )
    
    # Test recording simplification metrics with veracity data
    logger.info("Testing simplification metrics with veracity data")
    
    # Create mock veracity data
    simplification_veracity_data = {
        "verified": True,
        "score": 0.88,
        "confidence": 0.85,
        "issues": [
            {"type": "slight_meaning_change", "severity": "warning", "message": "Slight meaning change detected"}
        ],
        "metrics": {
            "semantic_similarity": 0.91,
            "readability_improvement": 0.82
        }
    }
    
    # Record a simplification with veracity data
    enhanced_metrics.record_simplification_metrics(
        language="en",
        text_length=2000,
        simplified_length=1500,
        level="3",
        processing_time=0.65,
        model_id="gpt-4",
        veracity_data=simplification_veracity_data
    )
    
    # Test recording audit scores
    logger.info("Testing audit score recording")
    
    # Record audit scores for different operations
    enhanced_metrics.record_audit_score(
        operation="translation",
        language="en",
        target_language="es",
        score=0.93,
        metadata={"review_type": "automatic"}
    )
    
    enhanced_metrics.record_audit_score(
        operation="simplification",
        language="en",
        target_language=None,
        score=0.89,
        metadata={"review_type": "automatic", "level": "3"}
    )
    
    # Wait a bit to allow async tasks to complete
    await asyncio.sleep(1)
    
    # Fetch and display veracity metrics
    logger.info("Retrieving veracity metrics")
    veracity_metrics = enhanced_metrics.get_veracity_metrics()
    print_metrics("Veracity Metrics", veracity_metrics)
    
    # Fetch and display audit metrics
    logger.info("Retrieving audit metrics")
    audit_metrics = enhanced_metrics.get_audit_metrics()
    print_metrics("Audit Metrics", audit_metrics)
    
    # Fetch and display all metrics together
    logger.info("Retrieving all metrics")
    all_metrics = enhanced_metrics.get_all_metrics()
    
    # Verify metrics contain our new sections
    assert "veracity" in all_metrics, "Veracity metrics section missing"
    assert "audit" in all_metrics, "Audit metrics section missing"
    assert all_metrics.get("enhanced", False), "Enhanced flag not set"
    
    logger.info("Enhanced metrics fix test completed successfully")
    
    # Write metrics to a file for inspection
    with open("test_metrics_output.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info("Metrics written to test_metrics_output.json")
    
    return True

def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    """Print metrics in a readable format."""
    print("\n" + "=" * 40)
    print(f"{title}")
    print("=" * 40)
    print(json.dumps(metrics, indent=2))
    print("=" * 40 + "\n")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_metrics_fix())