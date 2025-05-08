"""
Enhanced Integrations Module for CasaLingua

This module provides integration functions for the enhanced language models
in CasaLingua, including the language detector and simplifier enhancements,
and the metrics collection fixes. It provides a simple way to set up all
enhanced components together.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union

from app.utils.logging import get_logger
from app.utils.config import load_config, get_config_value
from app.audit.metrics_fix import setup_enhanced_metrics_collector
from app.services.models.language_detector_prompt_enhancer import LanguageDetectorPromptEnhancer
from app.services.models.simplifier_prompt_enhancer import SimplifierPromptEnhancer

logger = get_logger(__name__)

async def setup_enhanced_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """
    Setup all enhanced components for CasaLingua.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary with status of each component setup
    """
    logger.info("Setting up enhanced CasaLingua components")
    
    # Load config if not provided
    if config is None:
        config = load_config()
        
    # Track setup status
    setup_status = {
        "metrics_collector": False,
        "language_detector": False,
        "simplifier": False
    }
    
    # Setup enhanced metrics collector
    try:
        enhanced_metrics = setup_enhanced_metrics_collector(config)
        if enhanced_metrics:
            setup_status["metrics_collector"] = True
            logger.info("Enhanced metrics collector setup successfully")
        else:
            logger.warning("Failed to setup enhanced metrics collector")
    except Exception as e:
        logger.error(f"Error setting up enhanced metrics collector: {str(e)}")
    
    # Setup language detector prompt enhancer
    try:
        # Only import these modules if they exist
        from app.core.pipeline.language_detector import LanguageDetector
        
        # Create and inject the prompt enhancer
        language_detector_enhancer = LanguageDetectorPromptEnhancer()
        
        # Try to patch the detector class
        LanguageDetector.prompt_enhancer = language_detector_enhancer
        
        # If we got here without errors, mark as successful
        setup_status["language_detector"] = True
        logger.info("Language detector prompt enhancer setup successfully")
    except Exception as e:
        logger.error(f"Error setting up language detector enhancer: {str(e)}")
    
    # Setup simplifier prompt enhancer
    try:
        # Only import these modules if they exist
        from app.core.pipeline.simplifier import TextSimplifier
        
        # Create and inject the prompt enhancer
        simplifier_enhancer = SimplifierPromptEnhancer()
        
        # Try to patch the simplifier class
        TextSimplifier.prompt_enhancer = simplifier_enhancer
        
        # If we got here without errors, mark as successful
        setup_status["simplifier"] = True
        logger.info("Simplifier prompt enhancer setup successfully")
    except Exception as e:
        logger.error(f"Error setting up simplifier enhancer: {str(e)}")
    
    # Log overall status
    successful = sum(1 for v in setup_status.values() if v)
    total = len(setup_status)
    
    logger.info(f"Enhanced components setup complete: {successful}/{total} successful")
    
    return setup_status

def is_setup_successful(setup_status: Dict[str, bool]) -> bool:
    """
    Check if setup was successful.
    
    Args:
        setup_status: Setup status dictionary
        
    Returns:
        True if all components were setup successfully
    """
    return all(setup_status.values())

async def verify_enhanced_components() -> Dict[str, Dict[str, Any]]:
    """
    Verify that enhanced components are working correctly.
    
    Returns:
        Dictionary with verification results for each component
    """
    verification_results = {
        "metrics_collector": {
            "status": False,
            "details": {}
        },
        "language_detector": {
            "status": False,
            "details": {}
        },
        "simplifier": {
            "status": False,
            "details": {}
        }
    }
    
    # Verify metrics collector
    try:
        from app.audit.metrics import MetricsCollector
        from app.audit.metrics_fix import EnhancedMetricsCollector
        
        metrics = MetricsCollector.get_instance()
        
        # Check if it's the enhanced version
        if isinstance(metrics, EnhancedMetricsCollector):
            all_metrics = metrics.get_all_metrics()
            
            # Check for enhanced features
            has_enhanced = all_metrics.get("enhanced", False)
            has_veracity = "veracity" in all_metrics
            has_audit = "audit" in all_metrics
            
            verification_results["metrics_collector"]["status"] = has_enhanced and has_veracity
            verification_results["metrics_collector"]["details"] = {
                "is_enhanced_instance": isinstance(metrics, EnhancedMetricsCollector),
                "has_enhanced_flag": has_enhanced,
                "has_veracity_metrics": has_veracity,
                "has_audit_metrics": has_audit
            }
    except Exception as e:
        verification_results["metrics_collector"]["details"]["error"] = str(e)
    
    # Verify language detector
    try:
        from app.core.pipeline.language_detector import LanguageDetector
        from app.services.models.language_detector_prompt_enhancer import LanguageDetectorPromptEnhancer
        
        # Check if prompt enhancer is attached
        has_enhancer = hasattr(LanguageDetector, "prompt_enhancer")
        
        if has_enhancer:
            enhancer = LanguageDetector.prompt_enhancer
            is_correct_type = isinstance(enhancer, LanguageDetectorPromptEnhancer)
            
            verification_results["language_detector"]["status"] = is_correct_type
            verification_results["language_detector"]["details"] = {
                "has_prompt_enhancer": has_enhancer,
                "is_correct_type": is_correct_type
            }
    except Exception as e:
        verification_results["language_detector"]["details"]["error"] = str(e)
    
    # Verify simplifier
    try:
        from app.core.pipeline.simplifier import TextSimplifier
        from app.services.models.simplifier_prompt_enhancer import SimplifierPromptEnhancer
        
        # Check if prompt enhancer is attached
        has_enhancer = hasattr(TextSimplifier, "prompt_enhancer")
        
        if has_enhancer:
            enhancer = TextSimplifier.prompt_enhancer
            is_correct_type = isinstance(enhancer, SimplifierPromptEnhancer)
            
            verification_results["simplifier"]["status"] = is_correct_type
            verification_results["simplifier"]["details"] = {
                "has_prompt_enhancer": has_enhancer,
                "is_correct_type": is_correct_type
            }
    except Exception as e:
        verification_results["simplifier"]["details"]["error"] = str(e)
    
    # Return results
    return verification_results