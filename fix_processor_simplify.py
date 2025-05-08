#!/usr/bin/env python3
"""
Fix for adding missing simplify_text method to UnifiedProcessor

This script adds the missing simplify_text method to the UnifiedProcessor class
to address API integration issues with the CasaLingua demo.
"""

import os
import sys
import re
import shutil
from typing import Dict, Any, List, Optional

def apply_fix():
    """Apply the fix to add the simplify_text method to UnifiedProcessor."""
    print("Adding simplify_text method to UnifiedProcessor...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    processor_py_path = os.path.join(script_dir, "app/core/pipeline/processor.py")
    processor_py_backup = os.path.join(script_dir, "app/core/pipeline/processor.py.bak")
    
    # Create backup
    if os.path.exists(processor_py_path):
        shutil.copy2(processor_py_path, processor_py_backup)
        print(f"Created backup: {processor_py_backup}")
    else:
        print(f"Error: {processor_py_path} does not exist")
        return False
    
    # Read the processor.py file
    with open(processor_py_path, 'r') as f:
        content = f.read()
    
    # Define the new simplify_text method
    new_method = """
    async def simplify_text(
        self, 
        text: str, 
        target_level: str = "simple",
        source_language: str = "en",
        model_id: Optional[str] = None,
        options: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        \"\"\"
        Simplify text to the specified level.
        
        Args:
            text: Text to simplify
            target_level: Target simplification level (simple, medium, complex)
            source_language: Text language 
            model_id: Specific model to use
            options: Additional options
            user_id: Optional user ID for metrics
            request_id: Optional request ID for correlation
            
        Returns:
            Dict with simplified text and metadata
        \"\"\"
        # Initialize if needed
        if not self.initialized:
            await self.initialize()
            
        if not self.simplifier:
            logger.warning("Simplifier not initialized, initializing now")
            await self._initialize_simplifier()
        
        options = options or {}
        
        # Start timing
        start_time = time.time()
        
        # Auto-detect language if not provided
        detect_lang = source_language
        if not detect_lang:
            try:
                if not self.language_detector:
                    await self._initialize_language_detector()
                
                detection_result = await self.language_detector.detect_language(text)
                detect_lang = detection_result["detected_language"]
                logger.debug(f"Auto-detected language for simplification: {detect_lang}")
            except Exception as e:
                logger.warning(f"Language detection failed, defaulting to English: {str(e)}")
                detect_lang = "en"
        
        # Map target_level to level parameter
        level = 3  # Default to middle level
        if isinstance(target_level, int) or target_level.isdigit():
            # Convert string digit to int if needed
            level = int(target_level) if isinstance(target_level, str) else target_level
            level = max(1, min(5, level))  # Ensure level is between 1-5
        elif target_level.lower() == "simple":
            level = 4
        elif target_level.lower() == "medium":
            level = 3
        elif target_level.lower() == "complex":
            level = 2
        
        # Run simplification
        try:
            # Start with the result dict
            simplification_result = {
                "text": text,
                "language": detect_lang,
                "target_level": target_level,
                "level": level,
                "model_used": "simplifier"
            }
            
            # Get source text metrics for evaluation
            original_length = len(text)
            original_word_count = len(text.split())
            
            # Call the simplifier with the correct method signature
            simplified_text = await self.simplifier.simplify(
                text=text,
                language=detect_lang,
                level=level,
                options=options
            )
            
            # Extract simplified text
            if isinstance(simplified_text, dict):
                simplification_result.update(simplified_text)
            else:
                simplification_result["simplified_text"] = simplified_text
            
            # Ensure simplified_text is in the result
            if "simplified_text" not in simplification_result:
                logger.warning("Simplified text not found in result, using rule-based simplification")
                if hasattr(self.simplifier, "_rule_based_simplify"):
                    simplification_result["simplified_text"] = self.simplifier._rule_based_simplify(
                        text=text, 
                        level=level, 
                        language=detect_lang,
                        domain=options.get("domain")
                    )
                    simplification_result["model_used"] = f"rule_based_simplifier_level_{level}"
                else:
                    logger.error("Rule-based simplification not available")
                    simplification_result["simplified_text"] = text
                    simplification_result["model_used"] = "fallback"
            
            # Get simplified text metrics
            simplified_length = len(simplification_result["simplified_text"])
            simplified_word_count = len(simplification_result["simplified_text"].split())
            
            # Add metrics
            simplification_ratio = simplified_length / original_length if original_length > 0 else 1.0
            word_reduction_ratio = simplified_word_count / original_word_count if original_word_count > 0 else 1.0
            
            # Calculate approximate quality scores (mock data for demo purposes)
            readability_score = 0.85
            simplicity_score = max(0.5, 1.0 - (level * 0.1))
            
            # Add scores to result
            simplification_result["simplification_ratio"] = simplification_ratio
            simplification_result["word_reduction_ratio"] = word_reduction_ratio
            simplification_result["readability_score"] = readability_score
            simplification_result["simplicity_score"] = simplicity_score
            
        except Exception as e:
            logger.error(f"Error in simplification: {str(e)}", exc_info=True)
            # Fallback to rule-based simplification in case of error
            if hasattr(self.simplifier, "_rule_based_simplify"):
                simplified_text = self.simplifier._rule_based_simplify(
                    text=text, 
                    level=level, 
                    language=detect_lang,
                    domain=options.get("domain")
                )
                simplification_result = {
                    "text": text,
                    "simplified_text": simplified_text,
                    "language": detect_lang,
                    "target_level": target_level,
                    "level": level,
                    "model_used": f"rule_based_simplifier_level_{level}",
                    "error": str(e)
                }
            else:
                # If rule-based simplification also fails, return original text
                simplification_result = {
                    "text": text,
                    "simplified_text": text,
                    "language": detect_lang,
                    "target_level": target_level,
                    "level": level,
                    "model_used": "error_fallback",
                    "error": str(e)
                }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        simplification_result["processing_time"] = processing_time
        
        # Add performance metrics
        simplification_result["performance_metrics"] = {
            "characters_per_second": original_length / processing_time if processing_time > 0 else 0,
            "words_per_second": original_word_count / processing_time if processing_time > 0 else 0,
            "latency_ms": processing_time * 1000
        }
        
        # Add memory usage metrics (mock data for now)
        simplification_result["memory_usage"] = {
            "peak_mb": 160.0,
            "allocated_mb": 130.0,
            "util_percent": 70.0
        }
        
        # Add operation cost (mock data for now)
        operation_cost = 0.02 * (original_length / 1000)  # $0.02 per 1000 characters
        simplification_result["operation_cost"] = operation_cost
        
        # Audit and metrics
        if user_id and hasattr(self, "audit_logger") and self.audit_logger:
            await self.audit_logger.log_simplification(
                user_id=user_id,
                request_id=request_id or str(uuid4()),
                language=detect_lang,
                level=str(level),
                text_length=original_length,
                simplified_length=len(simplification_result["simplified_text"]),
                model_id=simplification_result.get("model_used", "unknown"),
                processing_time=processing_time,
                metadata={
                    "operation_cost": operation_cost,
                    "readability_score": readability_score,
                    "simplicity_score": simplicity_score,
                    "target_level": target_level
                }
            )
            
            # Collect metrics
            if hasattr(self, "metrics") and self.metrics:
                self.metrics.record_simplification_metrics(
                    language=detect_lang,
                    text_length=original_length,
                    simplified_length=len(simplification_result["simplified_text"]),
                    level=str(level),
                    processing_time=processing_time,
                    model_id=simplification_result.get("model_used", "unknown")
                )
            elif hasattr(self, "metrics_collector") and self.metrics_collector:
                self.metrics_collector.record_simplification_metrics(
                    language=detect_lang,
                    text_length=original_length,
                    simplified_length=len(simplification_result["simplified_text"]),
                    level=str(level),
                    processing_time=processing_time,
                    model_id=simplification_result.get("model_used", "unknown")
                )
        
        return simplification_result
    """
    
    # Check if method already exists
    if "async def simplify_text" in content:
        print("simplify_text method already exists. Updating instead of adding new method.")
        # Replace existing method
        simplify_text_pattern = r'async def simplify_text.*?memory_usage": memory_usage\s+}\s+\)'
        if re.search(simplify_text_pattern, content, re.DOTALL):
            content = re.sub(simplify_text_pattern, new_method.strip(), content, flags=re.DOTALL)
        else:
            # If specific pattern not found but method exists, find a better pattern
            print("Couldn't find exact method pattern. Skipping update.")
            return False
    else:
        # Insert method after summarize_text method
        if "async def summarize_text" in content:
            # Find the end of summarize_text method
            summarize_text_end_pattern = r'(return summarization_result\s+\n)'
            content = re.sub(summarize_text_end_pattern, r'\1\n' + new_method, content, flags=re.DOTALL)
        else:
            # If summarize_text not found, insert before anonymize_text
            if "async def anonymize_text" in content:
                content = content.replace("async def anonymize_text", new_method + "\n\n    async def anonymize_text")
            else:
                print("Could not find a suitable place to insert simplify_text method. Aborting.")
                return False
    
    # Write the modified file
    with open(processor_py_path, 'w') as f:
        f.write(content)
    
    print("Successfully added simplify_text method to UnifiedProcessor!")
    print("Please restart the server for changes to take effect.")
    return True

if __name__ == "__main__":
    apply_fix()