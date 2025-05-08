"""
Base Model Wrapper for CasaLingua
This module contains the base wrapper class for models to avoid circular imports.
The wrapper provides standardized interfaces for model operations, preprocessing,
and postprocessing, with support for model stability monitoring and veracity checks.
"""

import logging
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

class ModelInput(BaseModel):
    """
    Input model for standardized wrapper inputs.
    Provides a consistent interface for all model types.
    """
    text: Union[str, List[str]]
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    context: Optional[Union[str, List[str]]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    user_id: Optional[str] = None

class ModelOutput(BaseModel):
    """
    Output model for standardized wrapper outputs.
    Includes the model result and metadata for analysis and monitoring.
    """
    result: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    veracity_score: Optional[float] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    stability_metrics: Dict[str, Any] = Field(default_factory=dict)
    
class VeracityMetrics(BaseModel):
    """
    Metrics for tracking the veracity (truthfulness) of model outputs.
    Used for monitoring and improving model accuracy.
    """
    score: float = 1.0  # Scale of 0-1 where 1 is highest veracity
    confidence: float = 1.0  # Model's confidence in its output
    checks_passed: List[str] = Field(default_factory=list)
    checks_failed: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    source_verification: Optional[Dict[str, Any]] = None  # References to source material

class StabilityMetrics(BaseModel):
    """
    Metrics for monitoring model stability and performance.
    Used to detect and prevent model degradation.
    """
    memory_usage_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    execution_time_ms: Optional[float] = None
    error_rate: Optional[float] = 0.0
    retry_count: int = 0
    warnings: List[str] = Field(default_factory=list)

def monitor_stability(func):
    """
    Decorator to monitor model stability metrics during execution.
    Captures performance data and handles recovery from common failure modes.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        stability_metrics = StabilityMetrics()
        
        # Capture initial memory usage if possible
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            initial_memory = None
        
        # Execute with retry logic
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                result = func(self, *args, **kwargs)
                
                # Capture final memory metrics if possible
                try:
                    if initial_memory is not None:
                        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                        stability_metrics.memory_usage_mb = current_memory - initial_memory
                        stability_metrics.peak_memory_mb = current_memory
                except Exception:
                    pass
                
                # Calculate execution time
                stability_metrics.execution_time_ms = (time.time() - start_time) * 1000
                stability_metrics.retry_count = retry_count
                
                # Attach stability metrics to result if it's a ModelOutput
                if isinstance(result, ModelOutput):
                    result.stability_metrics = stability_metrics.dict()
                
                return result
            
            except (torch.cuda.OutOfMemoryError, torch.mps.OutOfMemoryError) as e:
                # Critical OOM error - no retry
                stability_metrics.error_rate = 1.0
                stability_metrics.warnings.append(f"OOM Error: {str(e)}")
                logger.error(f"Out of memory error in model execution: {str(e)}")
                raise
            
            except (RuntimeError, ValueError) as e:
                # Potentially recoverable error
                retry_count += 1
                last_error = e
                stability_metrics.retry_count = retry_count
                logger.warning(f"Model error (attempt {retry_count}/{max_retries}): {str(e)}")
                
                # Exponential backoff
                time.sleep(0.5 * (2 ** retry_count))
        
        # All retries failed
        stability_metrics.error_rate = 1.0
        if last_error:
            stability_metrics.warnings.append(f"Failed after {max_retries} attempts: {str(last_error)}")
        
        # Raise the last error
        if last_error:
            raise last_error
        
        # Should never reach here
        return None
    
    return wrapper

class BaseModelWrapper(ABC):
    """
    Base class for all model wrappers in CasaLingua.
    Provides a standardized interface and common functionality for all model types.
    """
    
    def __init__(self, model, tokenizer, config: Dict[str, Any] = None, **kwargs):
        """
        Initialize the model wrapper with the provided model and tokenizer.
        The device should be set by the loader, not the wrapper.
        
        Args:
            model: The underlying model instance
            tokenizer: The tokenizer for the model
            config: Configuration parameters for the wrapper
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # Device should be pre-determined by the loader
        self.device = getattr(model, "device", "cpu") if hasattr(model, "device") else "cpu"
        
        # Set model to evaluation mode if it's a PyTorch model
        if hasattr(self.model, "eval") and callable(self.model.eval):
            self.model.eval()
        
        # Initialize veracity auditor if available
        self.veracity_checker = None
        try:
            from app.audit.veracity import VeracityAuditor
            self.veracity_checker = VeracityAuditor()
            logger.info("Veracity auditor initialized for model wrapper")
        except ImportError:
            logger.info("Veracity auditor not available - veracity checks will be skipped")
        
        # Performance monitoring
        self.execution_count = 0
        self.error_count = 0
        self.total_execution_time = 0
        self.max_execution_time = 0
        
        # Initialize any provided hooks
        self.pre_process_hook = kwargs.get('pre_process_hook')
        self.post_process_hook = kwargs.get('post_process_hook')
    
    async def process_async(self, input_data: Union[Dict[str, Any], ModelInput]) -> Dict[str, Any]:
        """
        Process input asynchronously with stability monitoring.
        
        Args:
            input_data: The input data to process
            
        Returns:
            Dict containing the result and metadata
        """
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # Convert to ModelInput if needed
            if isinstance(input_data, ModelInput):
                model_input = input_data
            else:
                model_input = ModelInput(**input_data)
            
            # Apply pre-process hook if defined
            if self.pre_process_hook and callable(self.pre_process_hook):
                model_input = self.pre_process_hook(model_input)
            
            # Run processing pipeline with stability monitoring
            try:
                preprocessed = self._preprocess(model_input)
                raw_output = self._run_inference(preprocessed)
                result = self._postprocess(raw_output, model_input)
            except Exception as e:
                logger.error(f"Error in model processing: {str(e)}", exc_info=True)
                self.error_count += 1
                raise
            
            # Apply veracity checks if available
            if self.veracity_checker and not isinstance(result.result, (bytes, bytearray)):
                try:
                    veracity_metrics = await self._check_veracity(result.result, model_input)
                    result.veracity_score = veracity_metrics.score
                    result.metadata["veracity"] = veracity_metrics.dict()
                except Exception as e:
                    logger.warning(f"Error in veracity checking: {str(e)}")
                    result.metadata["veracity_error"] = str(e)
            
            # Record execution time
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.max_execution_time = max(self.max_execution_time, execution_time)
            
            # Add execution time to result
            result.processing_time = execution_time
            
            # Apply post-process hook if defined
            if self.post_process_hook and callable(self.post_process_hook):
                result = self.post_process_hook(result, model_input)
            
            # Convert to dictionary
            if isinstance(result, ModelOutput):
                output_dict = result.dict()
            else:
                # Legacy model wrapper support
                output_dict = {
                    "result": result,
                    "processing_time": execution_time
                }
            
            return output_dict
            
        except Exception as e:
            # Log and propagate error
            logger.error(f"Model processing error: {str(e)}", exc_info=True)
            self.error_count += 1
            
            # Return error response
            return {
                "error": str(e),
                "result": None,
                "processing_time": time.time() - start_time
            }
    
    @monitor_stability
    def process(self, input_data: Union[Dict[str, Any], ModelInput]) -> Dict[str, Any]:
        """
        Synchronous version of process_async with stability monitoring.
        
        Args:
            input_data: The input data to process
            
        Returns:
            Dict containing the result and metadata
        """
        start_time = time.time()
        self.execution_count += 1
        
        try:
            # Convert to ModelInput if needed
            if isinstance(input_data, ModelInput):
                model_input = input_data
            else:
                model_input = ModelInput(**input_data)
            
            # Apply pre-process hook if defined
            if self.pre_process_hook and callable(self.pre_process_hook):
                model_input = self.pre_process_hook(model_input)
            
            # Run processing pipeline with stability monitoring
            try:
                preprocessed = self._preprocess(model_input)
                raw_output = self._run_inference(preprocessed)
                result = self._postprocess(raw_output, model_input)
            except Exception as e:
                logger.error(f"Error in model processing: {str(e)}", exc_info=True)
                self.error_count += 1
                raise
            
            # Run synchronous veracity checks if available
            if self.veracity_checker and not isinstance(result.result, (bytes, bytearray)):
                try:
                    veracity_metrics = self._check_veracity_sync(result.result, model_input)
                    result.veracity_score = veracity_metrics.score
                    result.metadata["veracity"] = veracity_metrics.dict()
                except Exception as e:
                    logger.warning(f"Error in veracity checking: {str(e)}")
                    result.metadata["veracity_error"] = str(e)
            
            # Record execution time
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.max_execution_time = max(self.max_execution_time, execution_time)
            
            # Add execution time to result
            result.processing_time = execution_time
            
            # Apply post-process hook if defined
            if self.post_process_hook and callable(self.post_process_hook):
                result = self.post_process_hook(result, model_input)
            
            # Convert to dictionary
            if isinstance(result, ModelOutput):
                output_dict = result.dict()
            else:
                # Legacy model wrapper support
                output_dict = {
                    "result": result,
                    "processing_time": execution_time
                }
            
            return output_dict
            
        except Exception as e:
            # Log and propagate error
            logger.error(f"Model processing error: {str(e)}", exc_info=True)
            self.error_count += 1
            
            # Return error response
            return {
                "error": str(e),
                "result": None,
                "processing_time": time.time() - start_time
            }
    
    async def _check_veracity(self, result: Any, input_data: ModelInput) -> VeracityMetrics:
        """
        Check the veracity of model outputs using the veracity auditor.
        
        Args:
            result: The model output to check
            input_data: The original input data
            
        Returns:
            VeracityMetrics containing veracity information
        """
        if not self.veracity_checker:
            # Return default metrics if no checker available
            return VeracityMetrics()
        
        try:
            # Convert result to string if needed for verification
            result_text = str(result) if not isinstance(result, str) else result
            input_text = str(input_data.text) if not isinstance(input_data.text, str) else input_data.text
            
            # Prepare options for verification
            options = {
                "source_language": input_data.source_language,
                "target_language": input_data.target_language,
                "request_id": input_data.request_id,
            }
            
            # Determine operation type based on available information
            if input_data.source_language and input_data.target_language and input_data.source_language != input_data.target_language:
                options["operation"] = "translation"
            elif input_data.parameters.get("simplify", False):
                options["operation"] = "simplification"
            else:
                options["operation"] = "generic"
            
            # Use the veracity auditor to check the content
            verification_result = await self.veracity_checker.check(
                input_text,
                result_text,
                options
            )
            
            # Extract information for VeracityMetrics
            checks_passed = []
            checks_failed = []
            warnings = []
            
            # Process issues and categorize them
            for issue in verification_result.get("issues", []):
                if issue["severity"] == "critical":
                    checks_failed.append(issue["type"])
                elif issue["severity"] == "warning":
                    warnings.append(issue["message"])
            
            # Add metrics as checks passed if they're good
            for metric_name, metric_value in verification_result.get("metrics", {}).items():
                if isinstance(metric_value, (int, float)) and metric_value > 0.7:
                    checks_passed.append(f"metric_{metric_name}")
            
            return VeracityMetrics(
                score=verification_result.get("score", 1.0),
                confidence=verification_result.get("confidence", 0.8),
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                warnings=warnings,
                source_verification=verification_result.get("metrics", {})
            )
        except Exception as e:
            logger.warning(f"Error in veracity checking: {str(e)}")
            return VeracityMetrics(
                score=0.5,  # Neutral score for failed verification
                confidence=0.3,  # Low confidence due to error
                warnings=[f"Verification error: {str(e)}"]
            )
    
    def _check_veracity_sync(self, result: Any, input_data: ModelInput) -> VeracityMetrics:
        """
        Synchronous version of veracity checking for when async isn't available.
        
        Args:
            result: The model output to check
            input_data: The original input data
            
        Returns:
            VeracityMetrics containing veracity information
        """
        if not self.veracity_checker:
            # Return default metrics if no checker available
            return VeracityMetrics()
        
        try:
            # Since our VeracityAuditor is primarily async-based, we'll use a more simplified
            # approach for synchronous checking based on basic rules
            result_text = str(result) if not isinstance(result, str) else result
            input_text = str(input_data.text) if not isinstance(input_data.text, str) else input_data.text
            
            checks_passed = []
            checks_failed = []
            warnings = []
            
            # Content length check
            if len(result_text) > 0:
                checks_passed.append("content_present")
            else:
                checks_failed.append("empty_content")
            
            # Check if output is identical to input
            if result_text.strip() == input_text.strip():
                if input_data.source_language != input_data.target_language:
                    checks_failed.append("untranslated_content")
                    warnings.append("Output is identical to input, suggesting no processing occurred")
            
            # Check length ratio between input and output (for translation)
            if input_data.source_language and input_data.target_language and input_data.source_language != input_data.target_language:
                input_words = len(input_text.split())
                output_words = len(result_text.split())
                ratio = output_words / max(1, input_words)
                
                # Extremely high or low ratios could indicate problems
                if ratio < 0.3 or ratio > 3.0:
                    warnings.append(f"Unusual length ratio: {ratio:.2f} (output/input)")
            
            # Hallucination indicators check
            hallucination_phrases = [
                "I don't have enough information",
                "I don't have access to",
                "I cannot provide",
                "I'm not able to",
                "As an AI language model"
            ]
            
            for phrase in hallucination_phrases:
                if phrase.lower() in result_text.lower():
                    warnings.append(f"potential_hallucination: {phrase}")
            
            # Check for numbers in translation
            if input_data.source_language and input_data.target_language and input_data.source_language != input_data.target_language:
                import re
                # Simple pattern to match numbers in text
                number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?\b'
                
                input_numbers = re.findall(number_pattern, input_text)
                output_numbers = re.findall(number_pattern, result_text)
                
                # If input has numbers but output doesn't have a similar amount
                if len(input_numbers) > 0 and len(output_numbers) < len(input_numbers) * 0.7:
                    warnings.append(f"Missing numbers: Found {len(input_numbers)} in input but only {len(output_numbers)} in output")
            
            # Calculate basic veracity score
            if checks_failed:
                score = max(0.0, 1.0 - (len(checks_failed) / (len(checks_passed) + len(checks_failed))))
            else:
                score = 1.0
            
            # Adjust for warnings
            if warnings:
                score = score * (1.0 - min(0.5, 0.1 * len(warnings)))
            
            metrics = {
                "input_length": len(input_text),
                "output_length": len(result_text),
                "warnings_count": len(warnings),
                "failed_checks": len(checks_failed)
            }
            
            return VeracityMetrics(
                score=score,
                confidence=0.6,  # Lower confidence for sync checks
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                warnings=warnings,
                source_verification=metrics
            )
        except Exception as e:
            logger.warning(f"Error in sync veracity checking: {str(e)}")
            return VeracityMetrics(
                score=0.5,  # Neutral score for failed verification
                confidence=0.3,  # Low confidence due to error
                warnings=[f"Verification error: {str(e)}"]
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this model wrapper.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.execution_count == 0:
            avg_execution_time = 0
        else:
            avg_execution_time = self.total_execution_time / self.execution_count
        
        return {
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.execution_count),
            "avg_execution_time": avg_execution_time,
            "max_execution_time": self.max_execution_time,
            "device": self.device
        }
    
    @abstractmethod
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """
        Preprocess input data - must be implemented by subclasses.
        
        Args:
            input_data: The input data to preprocess
            
        Returns:
            Dictionary of preprocessed data ready for model inference
        """
        raise NotImplementedError("Subclasses must implement _preprocess")
    
    @abstractmethod
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """
        Run model inference - must be implemented by subclasses.
        
        Args:
            preprocessed: The preprocessed data
            
        Returns:
            Raw model output
        """
        raise NotImplementedError("Subclasses must implement _run_inference")
    
    @abstractmethod
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """
        Postprocess model output - must be implemented by subclasses.
        
        Args:
            model_output: The raw model output
            input_data: The original input data
            
        Returns:
            ModelOutput containing processed results and metadata
        """
        raise NotImplementedError("Subclasses must implement _postprocess")