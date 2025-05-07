"""
Enhanced Metrics Collection Module for CasaLingua

This module extends the base MetricsCollector with improved tracking for
veracity metrics and audit scores, ensuring comprehensive reporting of
quality assessment metrics across language processing operations.

Author: Exygy Development Team
Version: 1.1.0
License: MIT
"""

import time
import os
import json
import threading
import math
import statistics
import asyncio
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Local imports
from app.utils.config import load_config, get_config_value
from app.utils.logging import get_logger
from app.audit.metrics import MetricsCollector
from app.audit.veracity import VeracityAuditor
from app.audit.logger import AuditLogger

logger = get_logger(__name__)

class EnhancedMetricsCollector(MetricsCollector):
    """
    Enhanced metrics collector that ensures proper reporting of
    veracity and audit scores by fixing inconsistencies in the metrics system.
    """
    
    _enhanced_instance = None
    _instance_lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> 'EnhancedMetricsCollector':
        """
        Get or create the singleton instance of EnhancedMetricsCollector.
        
        Args:
            config: Optional configuration to pass when creating the instance
            
        Returns:
            The singleton EnhancedMetricsCollector instance
        """
        with cls._instance_lock:
            if cls._enhanced_instance is None:
                cls._enhanced_instance = cls(config)
                # Set as the global instance to replace standard collector
                MetricsCollector._instance = cls._enhanced_instance
            return cls._enhanced_instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced metrics collector."""
        super().__init__(config)
        
        # Initialize veracity metrics storage
        self.veracity_metrics: Dict[str, Dict[str, Any]] = {}
        self.audit_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Hook registrations
        self._pre_operation_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._post_operation_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Register veracity tracking hooks
        self._register_veracity_hooks()
        
        logger.info("Enhanced metrics collector initialized")
    
    def _register_veracity_hooks(self) -> None:
        """Register hooks for tracking veracity metrics on operations."""
        # Register post-operation hooks for operations that can have veracity scores
        self.register_post_operation_hook(
            "translation", self._track_translation_veracity
        )
        self.register_post_operation_hook(
            "simplification", self._track_simplification_veracity
        )
    
    def register_pre_operation_hook(self, operation: str, hook: Callable) -> None:
        """
        Register a hook to run before an operation is recorded.
        
        Args:
            operation: Operation type
            hook: Hook function to call
        """
        self._pre_operation_hooks[operation].append(hook)
    
    def register_post_operation_hook(self, operation: str, hook: Callable) -> None:
        """
        Register a hook to run after an operation is recorded.
        
        Args:
            operation: Operation type
            hook: Hook function to call
        """
        self._post_operation_hooks[operation].append(hook)

    async def install_hooks(self) -> bool:
        """Install hooks to ensure metrics are properly reported."""
        try:
            # Get original instances 
            original_metrics = MetricsCollector.get_instance()
            
            # Create veracity auditor
            veracity = VeracityAuditor()
            
            # Create audit logger
            audit_logger = AuditLogger()
            
            # Store original functions
            original_record_translation = original_metrics.record_translation_metrics
            original_record_simplification = original_metrics.record_simplification_metrics
            
            # Replace with enhanced versions 
            def enhanced_record_translation_metrics(*args, **kwargs):
                # Call original function
                result = original_record_translation(*args, **kwargs)
                
                # Ensure metrics are logged to audit system
                try:
                    # Extract parameters
                    source_language = kwargs.get('source_language', args[0] if len(args) > 0 else 'unknown')
                    target_language = kwargs.get('target_language', args[1] if len(args) > 1 else 'unknown')
                    text_length = kwargs.get('text_length', args[2] if len(args) > 2 else 0)
                    processing_time = kwargs.get('processing_time', args[3] if len(args) > 3 else 0.0)
                    model_id = kwargs.get('model_id', args[4] if len(args) > 4 else 'unknown')
                    veracity_data = kwargs.get('veracity_data', None)
                    
                    # Create metadata with veracity information
                    metadata = {}
                    if veracity_data:
                        metadata["veracity"] = veracity_data
                    
                    # Track metrics in veracity system
                    asyncio.create_task(self._track_translation_veracity({
                        "source_lang": source_language,
                        "target_lang": target_language,
                        "operation": "translation",
                        "duration": processing_time,
                        "input_size": text_length,
                        "output_size": text_length,
                        "success": True,
                        "metadata": metadata
                    }))
                except Exception as e:
                    logger.warning(f"Failed to track enhanced translation metrics: {str(e)}")
                
                return result
                
            def enhanced_record_simplification_metrics(*args, **kwargs):
                # Call original function
                result = original_record_simplification(*args, **kwargs)
                
                # Ensure metrics are logged to audit system
                try:
                    # Extract parameters
                    language = kwargs.get('language', args[0] if len(args) > 0 else 'unknown')
                    text_length = kwargs.get('text_length', args[1] if len(args) > 1 else 0)
                    simplified_length = kwargs.get('simplified_length', args[2] if len(args) > 2 else 0)
                    level = kwargs.get('level', args[3] if len(args) > 3 else 'unknown')
                    processing_time = kwargs.get('processing_time', args[4] if len(args) > 4 else 0.0)
                    model_id = kwargs.get('model_id', args[5] if len(args) > 5 else 'unknown')
                    veracity_data = kwargs.get('veracity_data', None)
                    
                    # Create metadata with veracity information
                    metadata = {"level": level}
                    if veracity_data:
                        metadata["veracity"] = veracity_data
                        
                    # Track metrics in veracity system
                    asyncio.create_task(self._track_simplification_veracity({
                        "source_lang": language,
                        "target_lang": language,
                        "operation": "simplification",
                        "duration": processing_time,
                        "input_size": text_length,
                        "output_size": simplified_length,
                        "success": True,
                        "metadata": metadata
                    }))
                except Exception as e:
                    logger.warning(f"Failed to track enhanced simplification metrics: {str(e)}")
                
                return result
            
            # Apply the patches
            original_metrics.record_translation_metrics = enhanced_record_translation_metrics
            original_metrics.record_simplification_metrics = enhanced_record_simplification_metrics
            
            logger.info("Enhanced metrics collection hooks installed")
            return True
        except Exception as e:
            logger.error(f"Failed to install enhanced metrics hooks: {str(e)}")
            return False
    
    def _run_pre_operation_hooks(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run pre-operation hooks and return potentially modified data.
        
        Args:
            operation: Operation type
            data: Operation data
            
        Returns:
            Potentially modified operation data
        """
        modified_data = data.copy()
        
        for hook in self._pre_operation_hooks.get(operation, []):
            try:
                result = hook(modified_data)
                if result is not None:
                    modified_data = result
            except Exception as e:
                logger.error(f"Error in pre-operation hook for {operation}: {str(e)}")
                
        return modified_data
    
    def _run_post_operation_hooks(self, operation: str, data: Dict[str, Any]) -> None:
        """
        Run post-operation hooks.
        
        Args:
            operation: Operation type
            data: Operation data
        """
        for hook in self._post_operation_hooks.get(operation, []):
            try:
                hook(data)
            except Exception as e:
                logger.error(f"Error in post-operation hook for {operation}: {str(e)}")
    
    def record_language_operation(
        self,
        source_lang: str,
        target_lang: Optional[str],
        operation: str,
        duration: float,
        input_size: int,
        output_size: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record language operation metrics with enhanced tracking.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code (if applicable)
            operation: Operation performed
            duration: Execution time in seconds
            input_size: Size of input in characters/tokens
            output_size: Size of output in characters/tokens
            success: Whether operation was successful
            metadata: Additional metadata for the operation
        """
        if not self.enabled:
            return
        
        # Prepare operation data to be passed to hooks
        operation_data = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "operation": operation,
            "duration": duration,
            "input_size": input_size,
            "output_size": output_size,
            "success": success,
            "metadata": metadata or {}
        }
        
        # Run pre-operation hooks
        modified_data = self._run_pre_operation_hooks(operation, operation_data)
        
        # Call parent implementation with potentially modified data
        super().record_language_operation(
            source_lang=modified_data["source_lang"],
            target_lang=modified_data["target_lang"],
            operation=modified_data["operation"],
            duration=modified_data["duration"],
            input_size=modified_data["input_size"],
            output_size=modified_data["output_size"],
            success=modified_data["success"]
        )
        
        # Run post-operation hooks
        self._run_post_operation_hooks(operation, modified_data)
    
    def _track_translation_veracity(self, data: Dict[str, Any]) -> None:
        """
        Track veracity metrics for translation operations.
        
        Args:
            data: Translation operation data
        """
        try:
            metadata = data.get("metadata", {})
            if not metadata:
                return
            
            # Extract veracity information from metadata if available
            veracity_data = metadata.get("veracity", {})
            if not veracity_data:
                return
            
            source_lang = data["source_lang"]
            target_lang = data["target_lang"]
            lang_pair = f"{source_lang}-{target_lang}"
            
            # Initialize veracity metrics for this language pair if needed
            if lang_pair not in self.veracity_metrics:
                self.veracity_metrics[lang_pair] = {
                    "translations_verified": 0,
                    "translations_total": 0,
                    "average_score": 0.0,
                    "average_confidence": 0.0,
                    "issue_counts": defaultdict(int),
                    "scores": []
                }
            
            # Update metrics
            metrics = self.veracity_metrics[lang_pair]
            metrics["translations_total"] += 1
            
            # Track verification status
            if veracity_data.get("verified", False):
                metrics["translations_verified"] += 1
            
            # Track score and confidence
            score = veracity_data.get("score", 0.0)
            confidence = veracity_data.get("confidence", 0.0)
            
            # Update averages
            metrics["scores"].append(score)
            metrics["average_score"] = statistics.mean(metrics["scores"])
            
            # Update confidence (weighted average based on text length)
            old_confidence = metrics["average_confidence"]
            old_total = metrics["translations_total"] - 1
            metrics["average_confidence"] = (
                (old_confidence * old_total + confidence) / 
                metrics["translations_total"]
            )
            
            # Track issues
            for issue in veracity_data.get("issues", []):
                issue_type = issue.get("type", "unknown")
                metrics["issue_counts"][issue_type] += 1
            
            # Add to time series
            self._record_time_series("veracity", {
                "operation": "translation",
                "lang_pair": lang_pair,
                "score": score,
                "confidence": confidence,
                "verified": veracity_data.get("verified", False),
                "issue_count": len(veracity_data.get("issues", []))
            })
            
            # Log to audit system asynchronously
            try:
                async def log_to_audit():
                    audit_logger = AuditLogger()
                    await audit_logger.log_translation(
                        text_length=data.get("input_size", 0),
                        source_language=source_lang,
                        target_language=target_lang,
                        model_id=metadata.get("model_id", "unknown"),
                        processing_time=data.get("duration", 0.0),
                        quality_score=score,
                        metadata={
                            "tracked_by_enhanced_metrics": True,
                            "veracity_verified": veracity_data.get("verified", False)
                        }
                    )
                
                asyncio.create_task(log_to_audit())
            except Exception as e:
                logger.debug(f"Could not log translation to audit system: {str(e)}")
            
            logger.debug(f"Tracked translation veracity for {lang_pair}: score={score:.3f}")
            
        except Exception as e:
            logger.warning(f"Error tracking translation veracity metrics: {str(e)}")
    
    def _track_simplification_veracity(self, data: Dict[str, Any]) -> None:
        """
        Track veracity metrics for simplification operations.
        
        Args:
            data: Simplification operation data
        """
        try:
            metadata = data.get("metadata", {})
            if not metadata:
                return
            
            # Extract veracity information from metadata if available
            veracity_data = metadata.get("veracity", {})
            if not veracity_data:
                return
            
            language = data["source_lang"]
            level = metadata.get("level", "unknown")
            simplification_key = f"{language}:{level}"
            
            # Initialize veracity metrics for this simplification pair if needed
            if simplification_key not in self.veracity_metrics:
                self.veracity_metrics[simplification_key] = {
                    "simplifications_verified": 0,
                    "simplifications_total": 0,
                    "average_score": 0.0,
                    "average_confidence": 0.0,
                    "issue_counts": defaultdict(int),
                    "scores": []
                }
            
            # Update metrics
            metrics = self.veracity_metrics[simplification_key]
            metrics["simplifications_total"] += 1
            
            # Track verification status
            if veracity_data.get("verified", False):
                metrics["simplifications_verified"] += 1
            
            # Track score and confidence
            score = veracity_data.get("score", 0.0)
            confidence = veracity_data.get("confidence", 0.0)
            
            # Update averages
            metrics["scores"].append(score)
            metrics["average_score"] = statistics.mean(metrics["scores"])
            
            # Update confidence (weighted average based on text length)
            old_confidence = metrics["average_confidence"]
            old_total = metrics["simplifications_total"] - 1
            metrics["average_confidence"] = (
                (old_confidence * old_total + confidence) / 
                metrics["simplifications_total"]
            )
            
            # Track issues
            for issue in veracity_data.get("issues", []):
                issue_type = issue.get("type", "unknown")
                metrics["issue_counts"][issue_type] += 1
            
            # Add to time series
            self._record_time_series("veracity", {
                "operation": "simplification",
                "language": language,
                "level": level,
                "score": score,
                "confidence": confidence,
                "verified": veracity_data.get("verified", False),
                "issue_count": len(veracity_data.get("issues", []))
            })
            
            # Log to audit system asynchronously
            try:
                async def log_to_audit():
                    audit_logger = AuditLogger()
                    await audit_logger.log_simplification(
                        text_length=data.get("input_size", 0),
                        simplified_length=data.get("output_size", 0),
                        language=language,
                        level=str(level),
                        model_id=metadata.get("model_id", "unknown"),
                        processing_time=data.get("duration", 0.0),
                        metadata={
                            "tracked_by_enhanced_metrics": True,
                            "veracity_verified": veracity_data.get("verified", False),
                            "veracity_score": score
                        }
                    )
                
                asyncio.create_task(log_to_audit())
            except Exception as e:
                logger.debug(f"Could not log simplification to audit system: {str(e)}")
            
            logger.debug(f"Tracked simplification veracity for {language} (level {level}): score={score:.3f}")
            
        except Exception as e:
            logger.warning(f"Error tracking simplification veracity metrics: {str(e)}")
    
    def record_translation_metrics(
        self,
        source_language: str,
        target_language: str,
        text_length: int,
        processing_time: float,
        model_id: str = "unknown",
        veracity_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record translation metrics with veracity data.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            text_length: Length of the processed text
            processing_time: Processing time in seconds
            model_id: Model used for translation
            veracity_data: Veracity assessment data
        """
        if not self.enabled:
            return
        
        # Create metadata with veracity information
        metadata = {}
        if veracity_data:
            metadata["veracity"] = veracity_data
            metadata["model_id"] = model_id
        
        # Use the language_operation method to record metrics
        self.record_language_operation(
            source_lang=source_language,
            target_lang=target_language,
            operation="translation",
            duration=processing_time,
            input_size=text_length,
            output_size=text_length,  # Approximation, we don't have the actual output length
            success=True,
            metadata=metadata
        )
        
        # Record model usage if we have a model ID
        if model_id and model_id != "unknown":
            # Approximate token counts based on text length
            # This is a rough estimate - 1 token ≈ 4 characters for English
            input_tokens = max(1, text_length // 4)
            output_tokens = max(1, text_length // 4)
            
            self.record_model_usage(
                model_id=model_id,
                operation="translation",
                duration=processing_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True
            )
    
    def record_simplification_metrics(
        self,
        language: str,
        text_length: int,
        simplified_length: int,
        level: str,
        processing_time: float,
        model_id: str = "unknown",
        veracity_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record text simplification metrics with veracity data.
        
        Args:
            language: Language code
            text_length: Length of the original text
            simplified_length: Length of the simplified text
            level: Simplification level (e.g., "simple", "medium")
            processing_time: Processing time in seconds
            model_id: Model used for simplification
            veracity_data: Veracity assessment data
        """
        if not self.enabled:
            return
        
        # Create metadata with veracity information and level
        metadata = {"level": level, "model_id": model_id}
        if veracity_data:
            metadata["veracity"] = veracity_data
        
        # Use the language_operation method to record metrics
        self.record_language_operation(
            source_lang=language,
            target_lang=language,  # Same language for simplification
            operation="simplification",
            duration=processing_time,
            input_size=text_length,
            output_size=simplified_length,
            success=True,
            metadata=metadata
        )
        
        # Record model usage if we have a model ID
        if model_id and model_id != "unknown":
            # Approximate token counts based on text length
            # This is a rough estimate - 1 token ≈ 4 characters for English
            input_tokens = max(1, text_length // 4)
            output_tokens = max(1, simplified_length // 4)
            
            self.record_model_usage(
                model_id=model_id,
                operation="simplification",
                duration=processing_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True
            )
    
    def record_audit_score(
        self,
        operation: str,
        language: str,
        target_language: Optional[str],
        score: float,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an audit score for an operation.
        
        Args:
            operation: Type of operation (translation, simplification, etc.)
            language: Source language code
            target_language: Target language code (if applicable)
            score: Audit score (0-1)
            timestamp: Timestamp for the score (default: current time)
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = time.time()
        
        # Create audit key based on operation and languages
        audit_key = operation
        if target_language and language != target_language:
            audit_key = f"{operation}:{language}-{target_language}"
        else:
            audit_key = f"{operation}:{language}"
        
        # Initialize audit metrics for this key if needed
        if audit_key not in self.audit_metrics:
            self.audit_metrics[audit_key] = {
                "scores": [],
                "timestamps": [],
                "average_score": 0.0,
                "min_score": 1.0,
                "max_score": 0.0,
                "count": 0,
                "operation": operation,
                "language": language,
                "target_language": target_language
            }
        
        # Update metrics
        metrics = self.audit_metrics[audit_key]
        metrics["scores"].append(score)
        metrics["timestamps"].append(timestamp)
        metrics["count"] += 1
        metrics["average_score"] = statistics.mean(metrics["scores"])
        metrics["min_score"] = min(metrics["min_score"], score)
        metrics["max_score"] = max(metrics["max_score"], score)
        
        # Limit the size of scores and timestamps lists
        max_samples = get_config_value(self.metrics_config, "max_samples", 1000)
        if len(metrics["scores"]) > max_samples:
            metrics["scores"] = metrics["scores"][-max_samples:]
            metrics["timestamps"] = metrics["timestamps"][-max_samples:]
        
        # Record time series data
        self._record_time_series("audit", {
            "operation": operation,
            "language": language,
            "target_language": target_language,
            "score": score,
            "metadata": metadata or {}
        })
        
        logger.debug(f"Recorded audit score for {audit_key}: {score:.3f}")
    
    def get_veracity_metrics(self) -> Dict[str, Any]:
        """
        Get veracity metrics for all operations.
        
        Returns:
            Dictionary with veracity metrics
        """
        if not self.enabled or not self.veracity_metrics:
            return {
                "veracity_metrics": self.veracity_metrics,
                "has_enhanced_tracking": True
            }
        
        result = {
            "overall": {
                "verified_ratio": 0.0,
                "average_score": 0.0,
                "average_confidence": 0.0,
                "top_issues": []
            },
            "operations": {},
            "has_enhanced_tracking": True
        }
        
        # Process translation metrics
        translation_metrics = {
            "verified_count": 0,
            "total_count": 0,
            "average_score": 0.0,
            "language_pairs": {}
        }
        
        # Process simplification metrics
        simplification_metrics = {
            "verified_count": 0,
            "total_count": 0,
            "average_score": 0.0,
            "languages": {}
        }
        
        # Collect all issues
        all_issues = defaultdict(int)
        
        # Process all metrics
        for key, metrics in self.veracity_metrics.items():
            if "translations_total" in metrics:
                # This is a translation language pair
                translation_metrics["total_count"] += metrics["translations_total"]
                translation_metrics["verified_count"] += metrics["translations_verified"]
                
                # Add language pair metrics
                translation_metrics["language_pairs"][key] = {
                    "verified_count": metrics["translations_verified"],
                    "total_count": metrics["translations_total"],
                    "verified_ratio": metrics["translations_verified"] / max(1, metrics["translations_total"]),
                    "average_score": metrics["average_score"],
                    "average_confidence": metrics["average_confidence"],
                    "top_issues": self._get_top_issues(metrics["issue_counts"], 3)
                }
            
            elif "simplifications_total" in metrics:
                # This is a simplification language/level
                simplification_metrics["total_count"] += metrics["simplifications_total"]
                simplification_metrics["verified_count"] += metrics["simplifications_verified"]
                
                # Add language/level metrics
                language, level = key.split(":", 1) if ":" in key else (key, "unknown")
                
                if language not in simplification_metrics["languages"]:
                    simplification_metrics["languages"][language] = {
                        "verified_count": 0,
                        "total_count": 0,
                        "levels": {}
                    }
                
                lang_metrics = simplification_metrics["languages"][language]
                lang_metrics["verified_count"] += metrics["simplifications_verified"]
                lang_metrics["total_count"] += metrics["simplifications_total"]
                
                # Add level-specific metrics
                lang_metrics["levels"][level] = {
                    "verified_count": metrics["simplifications_verified"],
                    "total_count": metrics["simplifications_total"],
                    "verified_ratio": metrics["simplifications_verified"] / max(1, metrics["simplifications_total"]),
                    "average_score": metrics["average_score"],
                    "average_confidence": metrics["average_confidence"],
                    "top_issues": self._get_top_issues(metrics["issue_counts"], 3)
                }
            
            # Collect all issues
            for issue_type, count in metrics.get("issue_counts", {}).items():
                all_issues[issue_type] += count
        
        # Calculate overall metrics
        total_count = translation_metrics["total_count"] + simplification_metrics["total_count"]
        verified_count = translation_metrics["verified_count"] + simplification_metrics["verified_count"]
        
        if total_count > 0:
            result["overall"]["verified_ratio"] = verified_count / total_count
        
        # Calculate average scores
        if translation_metrics["total_count"] > 0:
            translation_metrics["average_score"] = (
                sum(m["average_score"] * m["total_count"] for m in translation_metrics["language_pairs"].values()) /
                translation_metrics["total_count"]
            )
        
        if simplification_metrics["total_count"] > 0:
            # Calculate simplification average score across all languages and levels
            level_scores = []
            for lang_data in simplification_metrics["languages"].values():
                for level_data in lang_data["levels"].values():
                    level_scores.extend([level_data["average_score"]] * level_data["total_count"])
            
            if level_scores:
                simplification_metrics["average_score"] = statistics.mean(level_scores)
        
        # Calculate overall average score
        if total_count > 0:
            result["overall"]["average_score"] = (
                (translation_metrics["average_score"] * translation_metrics["total_count"] +
                 simplification_metrics["average_score"] * simplification_metrics["total_count"]) /
                total_count
            )
        
        # Add top issues
        result["overall"]["top_issues"] = self._get_top_issues(all_issues, 5)
        
        # Add operation-specific metrics
        result["operations"]["translation"] = translation_metrics
        result["operations"]["simplification"] = simplification_metrics
        
        return result
    
    def _get_top_issues(self, issue_counts: Dict[str, int], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top issues by count.
        
        Args:
            issue_counts: Dictionary of issue counts
            limit: Maximum number of issues to return
            
        Returns:
            List of top issues
        """
        return [
            {"type": issue_type, "count": count}
            for issue_type, count in sorted(
                issue_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
        ]
    
    def get_audit_metrics(self) -> Dict[str, Any]:
        """
        Get audit metrics for all operations.
        
        Returns:
            Dictionary with audit metrics
        """
        if not self.enabled or not self.audit_metrics:
            return {}
        
        result = {
            "overall": {
                "average_score": 0.0,
                "operations": {}
            },
            "by_language": {},
            "by_language_pair": {}
        }
        
        # Calculate overall metrics
        all_scores = []
        operation_scores = defaultdict(list)
        
        for key, metrics in self.audit_metrics.items():
            all_scores.extend(metrics["scores"])
            operation = metrics["operation"]
            operation_scores[operation].extend(metrics["scores"])
        
        # Calculate overall average score
        if all_scores:
            result["overall"]["average_score"] = statistics.mean(all_scores)
        
        # Calculate operation-specific averages
        for operation, scores in operation_scores.items():
            if scores:
                result["overall"]["operations"][operation] = {
                    "average_score": statistics.mean(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "count": len(scores)
                }
        
        # Group by language and language pair
        for key, metrics in self.audit_metrics.items():
            operation = metrics["operation"]
            language = metrics["language"]
            target_language = metrics["target_language"]
            
            # Add to by_language metrics
            if language not in result["by_language"]:
                result["by_language"][language] = {
                    "operations": {}
                }
            
            if operation not in result["by_language"][language]["operations"]:
                result["by_language"][language]["operations"][operation] = {
                    "average_score": 0.0,
                    "count": 0
                }
            
            lang_op_metrics = result["by_language"][language]["operations"][operation]
            lang_op_metrics["average_score"] = metrics["average_score"]
            lang_op_metrics["count"] = metrics["count"]
            
            # Add to by_language_pair metrics for translation
            if target_language and language != target_language:
                lang_pair = f"{language}-{target_language}"
                
                if lang_pair not in result["by_language_pair"]:
                    result["by_language_pair"][lang_pair] = {
                        "source_language": language,
                        "target_language": target_language,
                        "operations": {}
                    }
                
                if operation not in result["by_language_pair"][lang_pair]["operations"]:
                    result["by_language_pair"][lang_pair]["operations"][operation] = {
                        "average_score": 0.0,
                        "count": 0
                    }
                
                pair_op_metrics = result["by_language_pair"][lang_pair]["operations"][operation]
                pair_op_metrics["average_score"] = metrics["average_score"]
                pair_op_metrics["count"] = metrics["count"]
        
        return result
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics including veracity and audit metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if not self.enabled:
            return {}
        
        # Get base metrics
        metrics = super().get_all_metrics()
        
        # Add enhanced metrics
        metrics["veracity"] = self.get_veracity_metrics()
        metrics["audit"] = self.get_audit_metrics()
        metrics["enhanced"] = True
        
        return metrics


# Helper function to setup the enhanced metrics collector
async def setup_enhanced_metrics() -> bool:
    """
    Setup the enhanced metrics collector.
    
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Create enhanced metrics collector
        enhanced_metrics = EnhancedMetricsCollector.get_instance()
        
        # Install hooks
        result = await enhanced_metrics.install_hooks()
        
        # Log result
        if result:
            logger.info("Enhanced metrics collector successfully setup")
        else:
            logger.warning("Failed to setup enhanced metrics collector")
            
        return result
    except Exception as e:
        logger.error(f"Error setting up enhanced metrics collector: {str(e)}")
        return False

def setup_enhanced_metrics_collector(config: Optional[Dict[str, Any]] = None) -> EnhancedMetricsCollector:
    """
    Set up the enhanced metrics collector as the global instance.
    
    Args:
        config: Application configuration
        
    Returns:
        Enhanced metrics collector instance
    """
    # Get or create the enhanced metrics collector
    collector = EnhancedMetricsCollector.get_instance(config)
    
    # Replace the base class instance with our enhanced instance
    MetricsCollector._instance = collector
    
    # Schedule hook installation
    asyncio.create_task(collector.install_hooks())
    
    logger.info("Enhanced metrics collector set up as global instance")
    
    return collector